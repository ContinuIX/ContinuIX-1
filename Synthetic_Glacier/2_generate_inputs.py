#!/usr/bin/env python3

import os
import re
import struct
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# === 1. Domain parameters & raster grid -- Half-pipe ===

from domain_config import (
    n, w, max_altitude, max_depth, slope, glacier_length,
    rows, cols, x0, y0, dx, dy, x_coords, y_coords, X, Y,
    bed, ela, max_a, da_ds,
)

print(f"Grid: ({rows}, {cols}), dx={dx:.4f} m")
print(f"Bed range: {np.nanmin(bed):.1f} – {np.nanmax(bed):.1f} m")

# === 2. VTU readers -- Helper functions ===

def _parse_vtu(path):
    """
    Low-level binary parser for Firedrake VTU files (appended-raw encoding).

    Locates the appended-data block, reads the XML header to build a registry
    of all DataArrays (name -> byte offset, numpy dtype, number of components),
    and returns a read_array() closure that decodes any named array on demand.

    Parameters
    ----------
    path : str -- path to the .vtu file

    Returns
    -------
    raw        : bytes    -- full file contents
    read_array : callable -- read_array(name) -> numpy array
    n_points   : int      -- number of mesh nodes
    arrays     : dict     -- {name: (offset, dtype, n_components)}
    """
    with open(path, "rb") as f:
        raw = f.read()

    marker = b'<AppendedData encoding="raw">\n_'
    offset_base = raw.index(marker) + len(marker)

    header_bytes = raw[:raw.index(b"<AppendedData")]
    header_text = header_bytes.decode("utf-8", errors="replace")

    n_points = int(re.search(r'NumberOfPoints="(\d+)"', header_text).group(1))

    da_pattern = re.compile(
        r'<DataArray\s+'
        r'Name="([^"]+)"\s+'
        r'type="([^"]+)"\s+'
        r'NumberOfComponents="([^"]*)"\s+'
        r'format="appended"\s+'
        r'offset="(\d+)"'
    )

    arrays = {}
    for m in da_pattern.finditer(header_text):
        name = m.group(1)
        dtype_str = m.group(2)
        nc = int(m.group(3)) if m.group(3) else 1
        offset = int(m.group(4))
        np_dtype = {
            "Float64": np.float64, "Float32": np.float32,
            "Int32": np.int32, "Int64": np.int64, "UInt8": np.uint8,
        }[dtype_str]
        arrays[name] = (offset, np_dtype, nc)

    def read_array(name):
        """
        Decode a named DataArray from the appended binary block.

        Reads the 4-byte little-endian byte-count header at the given offset,
        then reads that many bytes and returns them as a numpy array of the
        registered dtype.

        Parameters
        ----------
        name : str -- key in the parent arrays dict

        Returns
        -------
        numpy array of the appropriate dtype and shape
        """
        offset, dtype, nc = arrays[name]
        pos = offset_base + offset
        nbytes = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return np.frombuffer(raw, dtype=dtype,
                             count=nbytes // np.dtype(dtype).itemsize, offset=pos)

    return raw, read_array, n_points, arrays


def read_vtu_scalar(path, field_name=None):
    """
    Read one scalar field from a Firedrake VTU file.

    If field_name is None, auto-selects the first non-topology scalar
    DataArray found in the header (n_components == 1).

    Parameters
    ----------
    path       : str      -- path to the .vtu file
    field_name : str|None -- name of the DataArray to extract; auto-detected
                             when None

    Returns
    -------
    pts  : (n_points, 3) float array -- mesh node coordinates (x, y, z)
    vals : (n_points,)   float array -- scalar values at each node
    """
    _, read_array, n_points, arrays = _parse_vtu(path)
    pts = read_array("firedrake_default_coordinates").reshape(n_points, 3)

    if field_name is None:
        skip = {"firedrake_default_coordinates", "connectivity", "offsets", "types"}
        candidates = [k for k in arrays if k not in skip and arrays[k][2] == 1]
        if not candidates:
            raise KeyError(f"No scalar field found. Available: {list(arrays.keys())}")
        field_name = candidates[0]

    vals = read_array(field_name)
    return pts, vals


def read_vtu_vector(path):
    """
    Read the first 3-component vector field from a Firedrake VTU file.

    Skips coordinate and topology arrays (firedrake_default_coordinates,
    connectivity, offsets, types). The z-component is discarded because
    the mesh is 2-D.

    Parameters
    ----------
    path : str -- path to the .vtu file

    Returns
    -------
    pts : (n_points, 3) float array -- mesh node coordinates
    vx  : (n_points,)  float array -- x-component of the vector
    vy  : (n_points,)  float array -- y-component of the vector
    """
    _, read_array, n_points, arrays = _parse_vtu(path)
    pts = read_array("firedrake_default_coordinates").reshape(n_points, 3)

    skip = {"firedrake_default_coordinates", "connectivity", "offsets", "types"}
    vec_name = None
    for k, (off, dt, nc) in arrays.items():
        if k not in skip and nc == 3:
            vec_name = k
            break
    if vec_name is None:
        raise KeyError(f"No vector field found. Available: {list(arrays.keys())}")

    vec = read_array(vec_name).reshape(n_points, 3)
    return pts, vec[:, 0], vec[:, 1]


def last_vtu(directory, prefix):
    """
    Return the path to the last (highest time-step index) VTU file in a
    Firedrake PVD output directory.

    Parameters
    ----------
    directory : str -- directory containing <prefix>_<index>.vtu files
    prefix    : str -- filename stem (e.g. "h", "flux_div")

    Returns
    -------
    str -- full path to the highest-index matching .vtu file
    """
    pattern = re.compile(rf"{prefix}_(\d+)\.vtu")
    files = [(f, int(pattern.search(f).group(1)))
             for f in os.listdir(directory)
             if f.endswith(".vtu") and pattern.search(f)]
    files.sort(key=lambda x: x[1])
    return os.path.join(directory, files[-1][0])


def grid_scalar(path, field_name=None):
    """
    Read a scalar VTU and linearly interpolate onto the raster grid (X, Y).

    Thin wrapper: read_vtu_scalar -> scipy.interpolate.griddata (linear).
    Grid X, Y are defined in the domain-parameters cell.

    Parameters
    ----------
    path       : str      -- path to the .vtu file
    field_name : str|None -- DataArray name; auto-detected when None

    Returns
    -------
    (rows, cols) float array -- field interpolated onto the raster grid
    """
    pts, vals = read_vtu_scalar(path, field_name)
    return griddata(pts[:, :2], vals, (X, Y), method="linear")

def grid_vector(path):
    """
    Read a vector VTU and interpolate both horizontal components onto the
    raster grid (X, Y).

    Parameters
    ----------
    path : str -- path to the .vtu file

    Returns
    -------
    vx_g : (rows, cols) float array -- x-component on raster grid
    vy_g : (rows, cols) float array -- y-component on raster grid
    """
    pts, vx, vy = read_vtu_vector(path)
    vx_g = griddata(pts[:, :2], vx, (X, Y), method="linear")
    vy_g = griddata(pts[:, :2], vy, (X, Y), method="linear")
    return vx_g, vy_g

# === 3. Read the last VTU from each output directory ===

last_h_path  = last_vtu("./output/h/", "h")
last_u_path  = last_vtu("./output/u/", "u")
last_fd_path = last_vtu("./output/flux_div/", "flux_div")
last_dh_path = last_vtu("./output/dhdt/", "dhdt")

print(f"Last h:        {last_h_path}")
print(f"Last u:        {last_u_path}")
print(f"Last flux_div: {last_fd_path}")
print(f"Last dhdt:     {last_dh_path}")

# ── Grid all fields ──
h_grid  = grid_scalar(last_h_path)
fd_grid = grid_scalar(last_fd_path)
dh_grid = grid_scalar(last_dh_path)
ux_grid, uy_grid = grid_vector(last_u_path)

# ── Glacier mask from thickness ──
glacier_mask = ~np.isnan(bed)

# ── Clamp thickness ──
# griddata can produce small negative values near the mesh boundary due to linear
# interpolation overshoots; clamp before any derived computation
h_grid = np.maximum(np.nan_to_num(h_grid, nan=0.0), 0.0)
h_grid[~glacier_mask] = np.nan

# ── Mask all fields: NaN where no ice or outside domain ──
ice_mask = glacier_mask & (h_grid > 0)
for arr in [ux_grid, uy_grid, fd_grid, dh_grid]:
    arr[~ice_mask] = np.nan

# ── Derived fields ──
vel_mag = np.sqrt(np.nan_to_num(ux_grid, 0)**2 + np.nan_to_num(uy_grid, 0)**2)
vel_mag[~ice_mask] = np.nan

surface = bed + np.nan_to_num(h_grid, nan=0.0)
surface[~glacier_mask] = np.nan

print(f"Thickness: {np.nanmin(h_grid):.2f} – {np.nanmax(h_grid):.2f} m")
print(f"Velocity:  {np.nanmin(vel_mag):.2f} – {np.nanmax(vel_mag):.2f} m/yr")
print(f"Flux div:  {np.nanmin(fd_grid):.4f} – {np.nanmax(fd_grid):.4f} m/yr")
print(f"dh/dt:     {np.nanmin(dh_grid):.4f} – {np.nanmax(dh_grid):.4f} m/yr")
print(f"Surface:   {np.nanmin(surface):.1f} – {np.nanmax(surface):.1f} m")

# === 4. SMB ===
smb = np.minimum((surface - ela) * da_ds, max_a).astype(np.float32)
smb[~ice_mask] = np.nan
print(f"SMB: {np.nanmin(smb):.3f} – {np.nanmax(smb):.3f} m/yr")

# === 5. Uncertainties ===

rng = np.random.default_rng(42)  # kept for reproducibility; perturbation draws use rng_pert below

# ── Uncertainties ──
vel_unc = 0.05 * np.abs(vel_mag)
thick_unc = 0.15 * np.nan_to_num(h_grid, 0)
smb_unc = np.full((rows, cols), 0.3)
dhdt_unc = np.full((rows, cols), 0.2)

# Flux divergence uncertainty propagated from ∇·(hu) ≈ (h·δv + v·δh) / δ
# delta = geometric-mean pixel size as the finite-difference length scale
# sqrt(2) combines the independent x and y flux-divergence terms in quadrature
delta = np.sqrt(dx * np.abs(dy))
flux_div_unc = np.sqrt(
    (np.nan_to_num(h_grid, 0) * vel_unc / delta)**2 +
    (np.nan_to_num(vel_mag, 0) * thick_unc / delta)**2
) * np.sqrt(2)

for arr in [vel_unc, thick_unc, smb_unc, dhdt_unc, flux_div_unc]:
    arr[~glacier_mask] = np.nan

print("Done.")

# === 6. Elevation bins (10 m step) ===

bin_step = 10.0
elev_min = np.floor(np.nanmin(surface) / bin_step) * bin_step
elev_max = np.ceil(np.nanmax(surface) / bin_step) * bin_step
bin_edges = np.arange(elev_min, elev_max + bin_step, bin_step)

elevation_bin = np.full((rows, cols), np.nan)
elevation_bin[glacier_mask] = np.digitize(surface[glacier_mask], bin_edges) - 1

n_bins = int(np.nanmax(elevation_bin)) + 1
print(f"{n_bins} bins from {elev_min:.0f} to {elev_max:.0f} m (step {bin_step:.0f} m)")

# === 7. Quick-look plots ===

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

# ELA contour mask
ela_bin = np.digitize([ela], bin_edges)[0] - 1
ela_mask = elevation_bin == ela_bin

fields = [
    ("Bed",                 bed,            "terrain",  False, False),
    ("Surface",             surface,        "terrain",  False, True),
    ("Thickness",           h_grid,         "viridis",  False, False),
    ("Velocity",            vel_mag,        "inferno",  False, False),
    ("SMB",                 smb,            "RdBu",     True,  True),
    ("dh/dt (icepack)",     dh_grid,        "RdBu",     True,  False),
    ("Flux div (icepack)",  fd_grid,        "RdBu_r",   True,  False),
    ("Elevation bin",       elevation_bin,  "tab20",    False, False),
]

for ax, (title, data, cmap, sym_clip, show_ela) in zip(axes.flat, fields):
    if sym_clip:
        vmax = np.nanpercentile(np.abs(data), 95)
        vmin = -vmax
    else:
        vmin, vmax = None, None

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # ELA contour only on panels where surface elevation is meaningful
    if show_ela:
        ela_overlay = np.full(data.shape + (4,), 0.0)  # RGBA, all transparent
        ela_overlay[ela_mask, 3] = 1.0  # black, fully opaque
        ax.imshow(ela_overlay, aspect="auto")

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/quick_look.png", dpi=150, bbox_inches="tight")
plt.close()

# === 8. Perturbations ===

from scipy.ndimage import zoom, gaussian_filter

rng_pert = np.random.default_rng(42)

# ── 1. Velocity speckle: additive uniform noise ∈ [-15, 15] m/a per pixel ───
speckle_x = rng_pert.uniform(-15.0, 15.0, size=(rows, cols))
speckle_y = rng_pert.uniform(-15.0, 15.0, size=(rows, cols))
speckle_x[~glacier_mask] = np.nan
speckle_y[~glacier_mask] = np.nan

vel_x_speckle = np.where(glacier_mask,
                          np.nan_to_num(ux_grid, 0) + np.nan_to_num(speckle_x, 0),
                          np.nan)
vel_y_speckle = np.where(glacier_mask,
                          np.nan_to_num(uy_grid, 0) + np.nan_to_num(speckle_y, 0),
                          np.nan)

# ── 2. Velocity bias: multiplicative ramp 10 → 0% along glacier length (x) — higher up is more uncertain ──
x_frac    = (X - x_coords.min()) / (x_coords.max() - x_coords.min())
bias_ramp = 0.10 * (1.0 - x_frac)
bias_x    = np.where(glacier_mask, np.nan_to_num(ux_grid, 0) * bias_ramp, np.nan)
bias_y    = np.where(glacier_mask, np.nan_to_num(uy_grid, 0) * bias_ramp, np.nan)

vel_x_biased = np.where(glacier_mask,
                         np.nan_to_num(ux_grid, 0) * (1.0 + bias_ramp), np.nan)
vel_y_biased = np.where(glacier_mask,
                         np.nan_to_num(uy_grid, 0) * (1.0 + bias_ramp), np.nan)

# ── 3. Velocity autocorrelated: same GRF factor on both components ────────────
# Single shared factor preserves the velocity direction; only speed is perturbed
# Mean-centered on the glacier mask so the factor averages to zero over the ice
# sigma in pixels = corr_m / dx (gaussian_filter takes pixel units)
corr_vel_m  = glacier_length / 2.0
_raw_vel    = rng_pert.standard_normal((rows, cols))
_smooth_vel = gaussian_filter(_raw_vel, sigma=corr_vel_m / dx)
_on_mask    = _smooth_vel[glacier_mask]
vel_grf_factor = (_smooth_vel - _on_mask.mean()) / _on_mask.std() * 0.10

vel_x_autocorr = np.where(glacier_mask,
                           np.nan_to_num(ux_grid, 0) * (1.0 + vel_grf_factor), np.nan)
vel_y_autocorr = np.where(glacier_mask,
                           np.nan_to_num(uy_grid, 0) * (1.0 + vel_grf_factor), np.nan)
vel_grf_factor[~glacier_mask] = np.nan

# ── 4. Coarsened: all fields downsampled 10× ─────────────────────────────────
_f = 0.1

def _coarsen(arr, fill=0.0):
    filled = np.where(glacier_mask, np.nan_to_num(arr, nan=fill), fill)
    return zoom(filled, _f, order=1)

rows_c     = int(round(rows * _f))
cols_c     = int(round(cols * _f))
dx_c       = dx / _f
dy_c       = dy / _f
x_coords_c = x0 + (np.arange(cols_c) + 0.5) * dx_c
y_coords_c = y0 + (np.arange(rows_c) + 0.5) * dy_c
mask_c     = zoom(glacier_mask.astype(np.float32), _f, order=1) >= 0.5  # majority-vote threshold

bed_c     = _coarsen(bed);     bed_c[~mask_c]     = np.nan
surface_c = _coarsen(surface); surface_c[~mask_c] = np.nan
h_c       = _coarsen(h_grid);  h_c[~mask_c]       = np.nan
ux_c      = _coarsen(ux_grid); ux_c[~mask_c]      = np.nan
uy_c      = _coarsen(uy_grid); uy_c[~mask_c]      = np.nan
mag_c     = np.sqrt(np.nan_to_num(ux_c)**2 + np.nan_to_num(uy_c)**2)
mag_c[~mask_c] = np.nan
smb_c     = _coarsen(smb);     smb_c[~mask_c]     = np.nan
dhdt_c    = _coarsen(dh_grid); dhdt_c[~mask_c]    = np.nan
fd_c      = _coarsen(fd_grid); fd_c[~mask_c]      = np.nan

# ── 5. Thickness bias: combined x·y multiplicative ramp at 25% and 50% ──────
h_clean = np.nan_to_num(h_grid, nan=0.0)
y_frac  = (Y - y_coords.min()) / (y_coords.max() - y_coords.min())

h_bias_25 = np.where(glacier_mask, h_clean * (1.0 + 0.125 * x_frac) * (1.0 + 0.125 * y_frac), np.nan)
h_bias_50 = np.where(glacier_mask, h_clean * (1.0 + 0.25  * x_frac) * (1.0 + 0.25  * y_frac), np.nan)

# ── 6. Thickness autocorrelated perturbation, oscillating 5–25% of local h ──
def _h_grf_factor(corr_m):
    raw = rng_pert.standard_normal((rows, cols))
    smoothed = gaussian_filter(raw, sigma=corr_m / dx)
    _on = smoothed[glacier_mask]
    smoothed = (smoothed - _on.mean()) / _on.std()  # mean-centre + unit std within ice
    return smoothed * 0.15

h_pert_factor_short = _h_grf_factor(100.0)
h_pert_factor_long  = _h_grf_factor(1000.0)

h_pert_autocorr_short = np.maximum(h_clean * (1.0 + h_pert_factor_short), 0.0)
h_pert_autocorr_long  = np.maximum(h_clean * (1.0 + h_pert_factor_long),  0.0)
h_pert_autocorr_short[~glacier_mask] = np.nan
h_pert_autocorr_long[~glacier_mask]  = np.nan

print(f"Vel GRF autocorr: corr={corr_vel_m:.0f} m, "
      f"factor mean={np.nanmean(vel_grf_factor):.4f}  std={np.nanstd(vel_grf_factor)*100:.1f}%")
print(f"Coarse grid: {rows_c}×{cols_c}  (dx={dx_c:.0f} m, dy={abs(dy_c):.0f} m)")

# === 8b. Perturbation quick-look plots ===

_speckle_mag = np.where(glacier_mask,
    np.sqrt(np.nan_to_num(speckle_x)**2 + np.nan_to_num(speckle_y)**2), np.nan)
_dh_bias_25       = np.where(glacier_mask, h_bias_25            - h_clean, np.nan)
_dh_bias_50       = np.where(glacier_mask, h_bias_50            - h_clean, np.nan)
_dh_autocorr_100  = np.where(glacier_mask, h_pert_autocorr_short - h_clean, np.nan)
_dh_autocorr_1000 = np.where(glacier_mask, h_pert_autocorr_long  - h_clean, np.nan)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Perturbation fields (seed=42)", fontsize=13)

_pert_fields = [
    ("Vel speckle magnitude (|vx|+|vy| noise ∈[-15,15])", _speckle_mag,               "Oranges", False, "m/yr"),
    ("Vel bias ramp",                  np.where(glacier_mask, bias_ramp * 100, np.nan),"RdBu_r",  False, "%"),
    ("Thickness bias 25%  (Δh)",       _dh_bias_25,                                   "RdBu_r",  True,  "m"),
    ("Thickness bias 50%  (Δh)",       _dh_bias_50,                                   "RdBu_r",  True,  "m"),
    ("Thickness autocorr 100 m  (Δh)", _dh_autocorr_100,                              "RdBu",    True,  "m"),
    ("Thickness autocorr 1000 m (Δh)", _dh_autocorr_1000,                             "RdBu",    True,  "m"),
]

for ax, (title, data, cmap, sym, units) in zip(axes.flat, _pert_fields):
    vmax = np.nanpercentile(np.abs(data), 95) if sym else None
    vmin = -vmax if sym else None
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7, label=units)

plt.tight_layout()
plt.savefig("./figures/perturbations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → ./figures/perturbations.png")

# === 9. Save to NetCDF ===

_coords_binned = {
    "x":              (("x",),              x_coords,           {"units": "m", "long_name": "Easting"}),
    "y":              (("y",),              y_coords,           {"units": "m", "long_name": "Northing"}),
    "elevation_band": (("elevation_band",), bin_edges[:n_bins], {"units": "m", "bin_step_m": bin_step}),
}

_attrs = {
    "title": "Synthetic half-pipe glacier — steady-state inputs from icepack",
    "glacier_length_m": glacier_length, "max_altitude_m": max_altitude,
    "max_depth_m": max_depth, "slope": slope, "ela_m": ela,
    "max_a_m_yr": max_a, "da_ds": da_ds, "glen_n": 3, "T_ice_K": 272.15,
    "dx_m": dx, "dy_m": dy, "crs": "EPSG:3857",
    "elevation_bin_step_m": bin_step, "perturbation_seed": 42,
}

def _v(arr, units, long_name):
    return (("y", "x"), arr.astype(np.float32), {"units": units, "long_name": long_name})

def _make_base_ds(extra_attrs=None):
    attrs = {**_attrs, **(extra_attrs or {})}
    return xr.Dataset(
        coords=_coords_binned,
        data_vars={
            "BED":           _v(bed,          "m",    "Bed elevation"),
            "DEM":       _v(surface,      "m",    "Ice surface elevation"),
            "THK":     _v(h_grid,       "m",    "Ice thickness"),
            "VX":    _v(ux_grid,      "m/yr", "Velocity x-component"),
            "VY":    _v(uy_grid,      "m/yr", "Velocity y-component"),
            "VMAG":  _v(vel_mag,      "m/yr", "Velocity magnitude"),
            "SMB":           _v(smb,          "m/yr", "Surface mass balance"),
            "DHDT":          _v(dh_grid,      "m/yr", "dh/dt"),
            "FDIV":      _v(fd_grid,      "m/yr", "Flux divergence"),
            "UNCT_VX":       _v(vel_unc,      "m/yr", "Standard deviation of VX"),
            "UNCT_VY":       _v(vel_unc,      "m/yr", "Standard deviation of VY"),
            "UNCT_THK": _v(thick_unc,    "m",    "Thickness uncertainty (1σ)"),
            "UNCT_SMB":       _v(smb_unc,      "m/yr", "SMB uncertainty (1σ)"),
            "UNCT_DHDT":      _v(dhdt_unc,     "m/yr", "dh/dt uncertainty (1σ)"),
            "UNCT_FDIV":  _v(flux_div_unc, "m/yr", "Flux divergence uncertainty (1σ)"),
            "elevation_bin": _v(elevation_bin,"1",    "Elevation band index"),
        },
        attrs=attrs,
    )

# ── 1. unperturbed.nc ────────────────────────────────────────────────────────
os.makedirs("./netcdfs", exist_ok=True)
_make_base_ds().to_netcdf("./netcdfs/unperturbed.nc")
print(f"unperturbed.nc    →  {os.path.getsize('./netcdfs/unperturbed.nc')/1e6:.0f} MB  "
      f"({len(xr.open_dataset('./netcdfs/unperturbed.nc').data_vars)} vars)")

# ── 2. perturbed.nc  (15 base + 10 perturbation variants = 25 vars) ──────────
vel_mag_speckle = np.sqrt(np.nan_to_num(vel_x_speckle)**2 + np.nan_to_num(vel_y_speckle)**2)
vel_mag_speckle[~glacier_mask] = np.nan

_coords_xy = {k: v for k, v in _coords_binned.items() if k in ("x", "y")}

ds_pert_extra = xr.Dataset(
    coords=_coords_xy,
    data_vars={
        "VX_speckle":        _v(vel_x_speckle,         "m/yr", "vx + speckle (seed=42, U[-15,15] m/a)"),
        "VY_speckle":        _v(vel_y_speckle,         "m/yr", "vy + speckle (seed=42, U[-15,15] m/a)"),
        "VX_biased":         _v(vel_x_biased,          "m/yr", "vx × (1 + bias_ramp, 10→0% head to toe)"),
        "VY_biased":         _v(vel_y_biased,          "m/yr", "vy × (1 + bias_ramp, 10→0% head to toe)"),
        "VX_autocorr":       _v(vel_x_autocorr,        "m/yr", "vx × (1 + GRF, corr=2500 m, std=10%, same factor as vy)"),
        "VY_autocorr":       _v(vel_y_autocorr,        "m/yr", "vy × (1 + GRF, corr=2500 m, std=10%, same factor as vx)"),
        "THK_bias_25":         _v(h_bias_25,             "m",    "h × ((1+0.125·x_frac)×(1+0.125·y_frac))"),
        "THK_bias_50":         _v(h_bias_50,             "m",    "h × ((1+0.25·x_frac)×(1+0.25·y_frac))"),
        "THK_autocorr_100m":   _v(h_pert_autocorr_short, "m",    "h × (1 + GRF, corr=100 m, std=15%)"),
        "THK_autocorr_1000m":  _v(h_pert_autocorr_long,  "m",    "h × (1 + GRF, corr=1000 m, std=15%)"),
    }
)
xr.merge([_make_base_ds(extra_attrs={"description": "base fields + all perturbation variants"}),
          ds_pert_extra]).to_netcdf("./netcdfs/perturbed.nc")
print(f"perturbed.nc      →  {os.path.getsize('./netcdfs/perturbed.nc')/1e6:.0f} MB  "
      f"({len(xr.open_dataset('./netcdfs/perturbed.nc').data_vars)} vars)")

# ── 3. perturbations.nc  (individual perturbation factor arrays) ──────────────
_bias_ramp_out = np.where(glacier_mask, bias_ramp, np.nan)

xr.Dataset(
    coords=_coords_xy,
    data_vars={
        "speckle_x":               _v(speckle_x,     "m/yr", "Additive speckle noise on vx, U[-15,15] (seed=42)"),
        "speckle_y":               _v(speckle_y,     "m/yr", "Additive speckle noise on vy, U[-15,15] (seed=42)"),
        "VX_biased":       _v(vel_x_biased,  "m/yr", "vx × (1 + bias_ramp, 10→0% head to toe)"),
        "VY_biased":       _v(vel_y_biased,  "m/yr", "vy × (1 + bias_ramp, 10→0% head to toe)"),
        "bias_ramp":               _v(_bias_ramp_out, "1",   "Velocity bias multiplier (0.10 → 0 head to toe)"),
        "vel_grf_factor":          _v(vel_grf_factor, "1",   "GRF factor applied to both vx and vy (corr=2500 m, std=10%)"),
        "VX_autocorr":     _v(vel_x_autocorr, "m/yr","vx × (1 + vel_grf_factor)"),
        "VY_autocorr":     _v(vel_y_autocorr, "m/yr","vy × (1 + vel_grf_factor)"),
        "x_frac":                  _v(np.where(glacier_mask, x_frac, np.nan), "1", "Along-glacier fraction (0→1)"),
        "y_frac":                  _v(np.where(glacier_mask, y_frac, np.nan), "1", "Cross-glacier fraction (0→1)"),
        "THK_bias_25":       _v(h_bias_25,      "m",   "h × ((1+0.125·x_frac)×(1+0.125·y_frac))"),
        "THK_bias_50":       _v(h_bias_50,      "m",   "h × ((1+0.25·x_frac)×(1+0.25·y_frac))"),
        "h_grf_factor_100m":       _v(np.where(glacier_mask, h_pert_factor_short, np.nan),
                                      "1", "GRF multiplicative factor, corr=100 m (seed=42)"),
        "h_grf_factor_1000m":      _v(np.where(glacier_mask, h_pert_factor_long,  np.nan),
                                      "1", "GRF multiplicative factor, corr=1000 m (seed=42)"),
        "THK_autocorr_100m": _v(h_pert_autocorr_short, "m", "h × (1 + GRF, corr=100 m, std=15%)"),
        "THK_autocorr_1000m":_v(h_pert_autocorr_long,  "m", "h × (1 + GRF, corr=1000 m, std=15%)"),
    },
    attrs={**_attrs, "description": "All individual perturbation arrays (seed=42)"}
).to_netcdf("./netcdfs/perturbations.nc")
print(f"perturbations.nc  →  {os.path.getsize('./netcdfs/perturbations.nc')/1e6:.0f} MB  "
      f"({len(xr.open_dataset('./netcdfs/perturbations.nc').data_vars)} vars)")

# ── 4. coarsened.nc  (unperturbed at 10× pixel size) ─────────────────────────
xr.Dataset(
    coords={"x": (("x",), x_coords_c, {"units": "m"}),
            "y": (("y",), y_coords_c, {"units": "m"})},
    data_vars={
        "BED":         _v(bed_c,     "m",    "Bed elevation"),
        "DEM":     _v(surface_c, "m",    "Ice surface elevation"),
        "THK":   _v(h_c,       "m",    "Ice thickness"),
        "VX":  _v(ux_c,      "m/yr", "Velocity x-component"),
        "VY":  _v(uy_c,      "m/yr", "Velocity y-component"),
        "VMAG":_v(mag_c,     "m/yr", "Velocity magnitude"),
        "SMB":         _v(smb_c,     "m/yr", "SMB"),
        "DHDT":        _v(dhdt_c,    "m/yr", "dh/dt"),
        "FDIV":    _v(fd_c,      "m/yr", "Flux divergence"),
    },
    attrs={**_attrs, "dx_m": dx_c, "dy_m": abs(dy_c),
           "description": "Unperturbed fields coarsened 10× (zoom, order=1)"}
).to_netcdf("./netcdfs/coarsened.nc")
print(f"coarsened.nc      →  {os.path.getsize('./netcdfs/coarsened.nc')/1e6:.0f} MB  "
      f"({len(xr.open_dataset('./netcdfs/coarsened.nc').data_vars)} vars)")
