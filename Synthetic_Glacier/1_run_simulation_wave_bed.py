#!/usr/bin/env python3

import os
import numpy as np
import geoutils as gu
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from pyproj import CRS
import fiona
import gmsh

# === Functions to create the inclined half pipe model ===

from domain_config import (
    n, w, max_altitude, max_depth, slope, glacier_length,
    rows, cols, x0, y0, dx, dy, x_coords, y_coords, X, Y,
    bed_wave as bed, ela, max_a, da_ds,
    create_inclined_pipe_matrix, n_dt, n_timesteps,
    wave_amplitude as _wave_amplitude, wave_sigma as _wave_sigma,
    wave_bump_centers as _bump_centers,
)

# === Plot bed configuration before running icepack ===
os.makedirs("./figures", exist_ok=True)

_mid_row  = rows // 2
_mid_col  = cols // 2
_gl_rows  = np.where(~np.isnan(bed[:, _mid_col]))[0]
_gl_cols  = np.where(~np.isnan(bed[_mid_row, :]))[0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Wave bed configuration  (A={_wave_amplitude} m, σ={_wave_sigma} m, bumps at x={_bump_centers} m)", y=1.02)

# ── 2D map ──
im = axes[0].imshow(bed, cmap="terrain", aspect="auto")
axes[0].axhline(_mid_row, color="red", lw=1.2, ls="--", label=f"centreline (row {_mid_row})")
axes[0].set_title("Bed elevation (2D)")
axes[0].set_xlabel("Along-flow (col)"); axes[0].set_ylabel("Cross-flow (row)")
axes[0].legend(fontsize=8)
plt.colorbar(im, ax=axes[0], label="Elevation (m)", shrink=0.8)

# ── Cross-flow profile at mid-glacier ──
axes[1].plot(y_coords[_gl_rows], bed[_gl_rows, _mid_col], color="steelblue", lw=1.5)
axes[1].axvline(y_coords[_mid_row], color="red", lw=1, ls="--", label="centreline")
axes[1].set_xlabel("Cross-flow y (m)"); axes[1].set_ylabel("Bed elevation (m)")
axes[1].set_title(f"Cross-flow profile at x ≈ {x_coords[_mid_col]:.0f} m")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# ── Along-flow centreline profile (shows bumps) ──
axes[2].plot(x_coords[_gl_cols], bed[_mid_row, _gl_cols], color="darkorange", lw=1.5)
axes[2].set_xlabel("Along-flow x (m)"); axes[2].set_ylabel("Bed elevation (m)")
axes[2].set_title("Centreline profile — wave bumps")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./figures/bed_config_wave.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → ./figures/bed_config_wave.png")


def create_pipe_outline_shapefile(matrix, dx, dy, x0, y0, crs):
    """
    Build a GeoDataFrame containing the glacier outline polygon.

    Finds the bounding box of valid (non-NaN) cells in the bed matrix,
    shrinks it by one cell on each side so the mesh sits safely inside
    the DEM extent, and returns a single-row GeoDataFrame with that
    rectangle as a Polygon.

    Parameters
    ----------
    matrix : (rows, cols) float array -- bed elevation; NaN outside the pipe
    dx     : float -- pixel width  (m, positive)
    dy     : float -- pixel height (m, negative)
    x0     : float -- x-coordinate of the top-left corner (m)
    y0     : float -- y-coordinate of the top-left corner (m)
    crs    : pyproj.CRS -- coordinate reference system for the output

    Returns
    -------
    gdf : GeoDataFrame -- one row, geometry = outline Polygon
    """
    valid_rows = np.where(~np.all(np.isnan(matrix), axis=1))[0]
    valid_cols = np.where(~np.all(np.isnan(matrix), axis=0))[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise ValueError("Matrix contains no valid pipe cells")

    rmin, rmax = valid_rows.min(), valid_rows.max()
    cmin, cmax = valid_cols.min(), valid_cols.max()

    # Shrink by 1 cell so the mesh boundary sits inside the DEM
    rmin += 1; rmax -= 1
    cmin += 1; cmax -= 1

    if rmin >= rmax or cmin >= cmax:
        raise ValueError("Pipe too small to shrink by 1 cell")

    x_left   = x0 + cmin * dx
    x_right  = x0 + (cmax + 1) * dx
    y_top    = y0 + rmin * dy
    y_bottom = y0 + (rmax + 1) * dy

    polygon = Polygon([
        (x_left,  y_top),
        (x_right, y_top),
        (x_right, y_bottom),
        (x_left,  y_bottom),
        (x_left,  y_top)
    ])

    return gpd.GeoDataFrame({"type": ["pipe_outline"]}, geometry=[polygon], crs=crs)

# === Create the model domain ===

# Define WGS84 CRS
wgs84 = CRS.from_epsg(3857)

# Construct raster transform from domain_config grid parameters
transform = (dx, 0.0, x0, 0.0, dy, y0)

# Create template raster and save DEM (used by Icepack to sample bed elevation)
os.makedirs("./data", exist_ok=True)
template = gu.Raster.from_array(bed, transform=transform, crs=wgs84)
template.save('./data/dem.tif')

print(f"Template raster created:")
print(f"  Shape: {template.shape}")
print(f"  Bounds: {template.bounds}")
print(f"  CRS: {template.crs}")
print(f"  Transform: {template.transform}")

# Create glacier outline polygon (used by Gmsh to build the mesh)
gdf = create_pipe_outline_shapefile(bed, dx, dy, x0, y0, wgs84)

# === Create the meshfile ===

# Stitch together the polygons (safe measure in case there are more than 1 polygons)
union_polygon = gdf.geometry.unary_union

# Get coordinates and remove the 3rd dimension (altitude)
coords = np.array(union_polygon.exterior.coords)
coords = coords[:, :2]  # Keep only x, y (remove z)

# Mesh resolution
lc = 50

# Trim last point (is identical to first)
coords = coords[:-1, :]

# Call Gmsh to create the mesh
gmsh.initialize()

gmsh.model.add("domain")

points = []

# Gather the points coordinates
for x, y in coords:
    points.append(gmsh.model.geo.addPoint(x, y, 0, lc))

# Create lines for the curves
lines = []
for i in range(len(points)):
    lines.append(gmsh.model.geo.addLine(points[i - 1], points[i]))

cl = gmsh.model.geo.addCurveLoop(lines)

ps = gmsh.model.geo.addPlaneSurface([cl], 1)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)

os.makedirs("./output_wave", exist_ok=True)
gmsh.write("./output_wave/synthetic.msh")

gmsh.finalize()

# === Icepack initialization functions and variables ===

import icepack
import firedrake
import rasterio
import numpy as np
import tqdm

# Import firedrake mesh from file
mesh = firedrake.Mesh("./output_wave/synthetic.msh")

# Define function spaces on the mesh
Q = firedrake.FunctionSpace(mesh, family="CG", degree=1)  # Scalar function space (continuous Galerkin, degree 1)
V = firedrake.VectorFunctionSpace(mesh, family="CG", degree=1)  # Vector function space for velocity
W = firedrake.VectorFunctionSpace(mesh, family="CG", degree=1)  # Vector function space for coordinates

# Extract mesh coordinates
x = firedrake.assemble(firedrake.interpolate(mesh.coordinates, W))
meshx = x.dat.data_ro[:, 0]  # x-coordinates of mesh nodes
meshy = x.dat.data_ro[:, 1]  # y-coordinates of mesh nodes

# Initialize bed elevation function
b = firedrake.Function(Q)

# Sample DEM at mesh node locations to get bed elevation
with rasterio.open("./data/dem.tif") as src:
    b.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))])
    # Print mesh extent and DEM metadata for verification
    print(meshx.min(), meshx.max())
    print(meshy.min(), meshy.max())
    print(src.bounds)
    print(src.crs)

# Write bed elevation to VTK file for visualization
firedrake.VTKFile("./output_wave/b.pvd").write(b)

# Initialize ice thickness to zero
h0 = firedrake.Function(Q).interpolate(firedrake.Constant(0.0))

# Calculate initial surface elevation (surface = bed + thickness)
s0 = firedrake.Function(Q).interpolate(b + h0)

# Set up shallow ice approximation model and solver
model = icepack.models.ShallowIce()
solver = icepack.solvers.FlowSolver(model)

# Define ice temperature and calculate rate factor (fluidity)
T = firedrake.Constant(273.15 - 1)  # Temperature in Kelvin (-1°C)
A = icepack.rate_factor(T)  # Glen's flow law rate factor

# Initialize velocity field
u0 = firedrake.Function(V)

# Copy initial thickness
h = h0.copy(deepcopy=True)

# Solve for initial velocity field (diagnostic solve)
u = solver.diagnostic_solve(
    velocity=u0,
    thickness=h,
    surface=s0,
    fluidity=A,
)

# Define mass balance function (accumulation/ablation rate)
def mass_balance(s, max_a=0.5, da_ds=0.5 / 1000, ela=300.0):
    # Linear mass balance: rate increases with elevation above ELA
    # Capped at maximum accumulation rate
    return firedrake.min_value((s - ela) * da_ds, max_a)

# === Run Icepack ===

# ela, max_a, da_ds come from domain_config

# Calculate initial mass balance
a = mass_balance(s0, ela=ela, max_a=max_a, da_ds=da_ds)

# Write mass balance to VTK file
a_func = firedrake.Function(Q, name="a")
a_func.interpolate(a)
firedrake.VTKFile("./output_wave/a.pvd").write(a_func)

# Set up time stepping parameters
dt = n_dt  # Time step (years)
num_timesteps = n_timesteps  # Total number of time steps

# Per-step thickness diagnostics (filled inside the loop)
time_series   = np.full(num_timesteps, np.nan)
h_max_series  = np.full(num_timesteps, np.nan)
h_mean_series = np.full(num_timesteps, np.nan)
h_vol_series  = np.full(num_timesteps, np.nan)   # volume = sum(h) * cell_area (m³)
cell_area     = dx * abs(dy)

# Initialize mass balance function for time stepping
a = firedrake.Function(Q)

# Open output files for thickness, velocity, flux divergence, dh/dt
hfile = firedrake.VTKFile("./output_wave/h.pvd")
hfile.write(h0, time=0)
ufile = firedrake.VTKFile("./output_wave/u.pvd")
ufile.write(u0, time=0)
fdfile = firedrake.VTKFile("./output_wave/flux_div.pvd")
dhdtfile = firedrake.VTKFile("./output_wave/dhdt.pvd")

# Functions for flux divergence and dh/dt
flux_div_func = firedrake.Function(Q, name="flux_div")
dhdt_func = firedrake.Function(Q, name="dhdt")

# Write initial flux divergence (h=0 so div(h*u)=0)
flux_div_func.interpolate(firedrake.Constant(0.0))
fdfile.write(flux_div_func, time=0)

# Write initial dh/dt (no change yet)
dhdt_func.interpolate(firedrake.Constant(0.0))
dhdtfile.write(dhdt_func, time=0)

# Store previous thickness for dh/dt computation
h_prev = h0.copy(deepcopy=True)

# Time stepping loop
for step in tqdm.trange(num_timesteps):
    # Prognostic solve: evolve thickness forward in time
    h = solver.prognostic_solve(
        dt,
        thickness=h,
        accumulation=a,
        velocity=u,
    )

    # Ensure thickness remains non-negative
    h.interpolate(firedrake.max_value(h, 0))

    # Record thickness diagnostics
    _h_vals = h.dat.data_ro
    time_series[step]   = dt * (step + 1)
    h_max_series[step]  = _h_vals.max()
    h_mean_series[step] = _h_vals[_h_vals > 0].mean() if np.any(_h_vals > 0) else 0.0
    h_vol_series[step]  = _h_vals.sum() * cell_area

    # Update surface elevation
    s = firedrake.Function(Q).interpolate(h + b)

    # Diagnostic solve: compute velocity from updated thickness and surface
    u = solver.diagnostic_solve(
        velocity=u,
        thickness=h,
        surface=s,
        fluidity=A,
    )

    # Update mass balance based on new surface elevation
    a.interpolate(mass_balance(s, ela=ela, max_a=max_a, da_ds=da_ds))

    # Write output every 10 time steps
    if not step % 10:
        t_out = dt * (step + 1)
        hfile.write(h, time=t_out)
        ufile.write(u, time=t_out)

        # Flux divergence: ∇·(h u)
        flux_div_func.project(firedrake.div(h * u))
        fdfile.write(flux_div_func, time=t_out)

        # dh/dt from finite difference
        dhdt_func.interpolate((h - h_prev) / (10 * dt))
        dhdtfile.write(dhdt_func, time=t_out)

        # Update h_prev for next output interval
        h_prev.assign(h)

# === Ice thickness evolution plot ===
fig_ev, axes_ev = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig_ev.suptitle("Ice thickness evolution over time (wave bed)")

axes_ev[0].plot(time_series, h_max_series,  color="steelblue",  lw=1.5)
axes_ev[0].set_ylabel("Max thickness (m)")
axes_ev[0].grid(alpha=0.3)

axes_ev[1].plot(time_series, h_mean_series, color="darkorange", lw=1.5)
axes_ev[1].set_ylabel("Mean thickness\nover ice (m)")
axes_ev[1].grid(alpha=0.3)

axes_ev[2].plot(time_series, h_vol_series / 1e6, color="seagreen", lw=1.5)
axes_ev[2].set_ylabel("Ice volume (km³)")
axes_ev[2].set_xlabel("Time (yr)")
axes_ev[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./figures/thickness_evolution_wave.png", dpi=150, bbox_inches="tight")
plt.close(fig_ev)
print("Saved → ./figures/thickness_evolution_wave.png")

# === Snapshot visualization (1st, middle, last) ===
import re as _re, struct as _st
from scipy.interpolate import griddata as _gd

def _vtu_snap(path, want_vec=False):
    with open(path, "rb") as _f:
        _raw = _f.read()
    _mk = b'<AppendedData encoding="raw">\n_'
    _ob = _raw.index(_mk) + len(_mk)
    _hdr = _raw[:_raw.index(b"<AppendedData")].decode("utf-8", errors="replace")
    _npts = int(_re.search(r'NumberOfPoints="(\d+)"', _hdr).group(1))
    _pat = _re.compile(
        r'<DataArray\s+Name="([^"]+)"\s+type="([^"]+)"\s+'
        r'NumberOfComponents="([^"]*)"\s+format="appended"\s+offset="(\d+)"'
    )
    _DT = {"Float64": np.float64, "Float32": np.float32, "Int32": np.int32}
    _arrs = {}
    for _m in _pat.finditer(_hdr):
        _arrs[_m.group(1)] = (int(_m.group(4)), _DT.get(_m.group(2), np.float32),
                               int(_m.group(3)) if _m.group(3) else 1)
    def _ra(nm):
        _o, _dt, _ = _arrs[nm]
        _p = _ob + _o
        _nb = _st.unpack_from("<I", _raw, _p)[0]
        return np.frombuffer(_raw, dtype=_dt, count=_nb // np.dtype(_dt).itemsize, offset=_p + 4)
    _pts = _ra("firedrake_default_coordinates").reshape(_npts, 3)[:, :2]
    _skip = {"firedrake_default_coordinates", "connectivity", "offsets", "types"}
    if want_vec:
        _vk = next(k for k, (_, _, nc) in _arrs.items() if k not in _skip and nc == 3)
        _v = _ra(_vk).reshape(_npts, 3)
        return _pts, _v[:, 0], _v[:, 1]
    else:
        _sk = next(k for k, (_, _, nc) in _arrs.items() if k not in _skip and nc == 1)
        return _pts, _ra(_sk)

_h_dir = "./output_wave/h"
_u_dir = "./output_wave/u"
_h_files = sorted([f for f in os.listdir(_h_dir) if f.endswith(".vtu")],
                   key=lambda f: int(_re.search(r"(\d+)", f).group(1)))
_u_files = sorted([f for f in os.listdir(_u_dir) if f.endswith(".vtu")],
                   key=lambda f: int(_re.search(r"(\d+)", f).group(1)))

_nh, _nu = len(_h_files), len(_u_files)
_snap_h = [0, _nh // 2, _nh - 1]
_snap_u = [0, _nu // 2, _nu - 1]
_labels = ["1st snapshot", "Middle snapshot", "Last snapshot"]

_xs = x0 + (np.arange(cols) + 0.5) * dx
_ys = y0 + (np.arange(rows) + 0.5) * dy
_Xg, _Yg = np.meshgrid(_xs, _ys)

fig_snap, axes_snap = plt.subplots(2, 3, figsize=(18, 8))
fig_snap.suptitle("Snapshots: thickness (top) & velocity magnitude (bottom)", y=1.01)

for _col, (_hi, _ui, _lbl) in enumerate(zip(_snap_h, _snap_u, _labels)):
    _pts_h, _vals_h = _vtu_snap(os.path.join(_h_dir, _h_files[_hi]))
    _h_grid = _gd(_pts_h, _vals_h, (_Xg, _Yg), method="linear")
    _h_grid[np.isnan(bed)] = np.nan
    _vmax_h = np.nanpercentile(_h_grid, 99) if np.any(~np.isnan(_h_grid)) else 1.0
    _im_h = axes_snap[0, _col].imshow(_h_grid, origin="upper", cmap="Blues",
                                       vmin=0, vmax=_vmax_h)
    axes_snap[0, _col].set_title(_lbl)
    axes_snap[0, _col].axis("off")
    plt.colorbar(_im_h, ax=axes_snap[0, _col], label="Thickness (m)", shrink=0.7)

    _pts_u, _vx, _vy = _vtu_snap(os.path.join(_u_dir, _u_files[_ui]), want_vec=True)
    _vmag = np.hypot(_vx, _vy)
    _vm_grid = _gd(_pts_u, _vmag, (_Xg, _Yg), method="linear")
    _vm_grid[np.isnan(bed)] = np.nan
    _vmax_u = np.nanpercentile(_vm_grid, 99) if np.any(~np.isnan(_vm_grid)) else 1.0
    _im_u = axes_snap[1, _col].imshow(_vm_grid, origin="upper", cmap="Reds",
                                       vmin=0, vmax=_vmax_u)
    axes_snap[1, _col].axis("off")
    plt.colorbar(_im_u, ax=axes_snap[1, _col], label="Speed (m/a)", shrink=0.7)

axes_snap[0, 0].set_ylabel("Thickness")
axes_snap[1, 0].set_ylabel("Speed")
plt.tight_layout()
plt.savefig("./figures/snapshots_wave.png", dpi=150, bbox_inches="tight")
plt.close(fig_snap)
print("Saved → ./figures/snapshots_wave.png")

