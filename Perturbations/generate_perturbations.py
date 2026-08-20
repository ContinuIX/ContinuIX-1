"""
Perturbation generator.

Reads NetCDF named after the glacier (<name>.nc, e.g. athabasca.nc).
Outputs 3_<name>_expYY.nc where YY is the 2-digit experiment number
(exp03..exp15), written to output/<name>/.

The glacier footprint is taken as np.isfinite(thickness) — the "mask" variable
in the sample files does NOT encode a glacier boundary (it holds acquisition
years / unrelated continuous values), so it is never used for that purpose,
only passed through as-is.

Random fields are seeded deterministically from the glacier name and experiment
number, so re-running this script reproduces byte-identical output; the exact
seed and every perturbation parameter are recorded as NetCDF attributes.

Usage:
    python generate_perturbations.py athabasca [saskatchewan.nc ...]   # .nc optional
    python generate_perturbations.py            # all *.nc files in this dir
"""
import os
import sys
import glob
import zlib
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, 'output')
WP = 3  # X in "X.Y.Z" — always 3 for this work package


# ── Gaussian random field (correlation length in METRES) ──────────────────────

def _masked_gaussian(arr, valid, sigma_px):
    """Normalised (mask-aware) Gaussian blur — no bleed across the glacier edge,
    NaN-safe. sigma_px is the smoothing scale in pixels."""
    a = np.where(valid, arr, 0.0)
    num = gaussian_filter(a, sigma_px, mode='nearest')
    den = gaussian_filter(valid.astype(float), sigma_px, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 1e-9, num / den, np.nan)


def random_field(shape, valid, corr_length_m, px, seed):
    """Gaussian random field: white noise smoothed to the target correlation length (in
    metres, converted to pixels via the grid spacing px), then rescaled so its
    standard deviation over the glacier footprint is exactly 1 — multiply by
    the desired amplitude afterward."""
    sigma_px = max(corr_length_m / px, 1e-6)
    rng = np.random.default_rng(seed)
    field = _masked_gaussian(rng.standard_normal(shape), valid, sigma_px)
    s = np.nanstd(field[valid])
    return field / s if s > 0 else field


# ── grid helpers ────────────────────────────────────────────────────────────

def pixel_size(ds):
    dx = np.abs(np.diff(ds['x'].values))
    dy = np.abs(np.diff(ds['y'].values))
    px = float(np.median(np.concatenate([dx, dy])))
    if not (np.allclose(dx, px, rtol=1e-3) and np.allclose(dy, px, rtol=1e-3)):
        raise ValueError("Non-uniform or non-square grid spacing — "
                         "random_field()'s pixel-to-metres conversion assumes square pixels.")
    return px


def glacier_footprint(ds):
    """On-glacier pixels: wherever thickness is finite (see module docstring —
    the 'mask' variable does not encode this)."""
    return np.isfinite(ds['thickness'].values)


# ── provenance ──────────────────────────────────────────────────────────────

def _stamp(ds, exp_id, **params):
    ds = ds.copy()
    ds.attrs['perturbation_experiment'] = f'Exp{exp_id}'
    return ds


# ── Thickness experiments (Exp3-7) ─────────────────────────────────────────

def exp_thickness_noise(ds, exp_id, corr_factor, amp_frac, amp_relative_to, seed):
    """Additive spatially-correlated thickness noise.
    corr_factor: correlation length = corr_factor * mean glacier thickness (m).
    amp_relative_to: 'local' -> amp_frac * this pixel's own thickness (Exp1/2);
                     'mean'  -> amp_frac * glacier-mean thickness, uniform (Exp3)."""
    px = pixel_size(ds)
    th = ds['thickness'].values
    valid = glacier_footprint(ds)
    mean_th = float(np.nanmean(th[valid]))
    corr_m = corr_factor * mean_th
    g = random_field(th.shape, valid, corr_m, px, seed)
    amp = amp_frac * th if amp_relative_to == 'local' else amp_frac * mean_th
    th_new = np.where(valid, np.clip(th + g * amp, 0, None), np.nan)
    out = ds.copy()
    out['thickness'] = (ds['thickness'].dims, th_new, ds['thickness'].attrs)
    return _stamp(out, exp_id, corr_length_m=corr_m, corr_factor_x_mean_thickness=corr_factor,
                 amplitude_fraction=amp_frac, amplitude_relative_to=amp_relative_to, rng_seed=seed)


def exp_thickness_scale(ds, exp_id, factor):
    """Deterministic multiplicative thickness scaling (Exp4/5) — no randomness."""
    th = ds['thickness'].values
    valid = glacier_footprint(ds)
    th_new = np.where(valid, th * factor, np.nan)
    out = ds.copy()
    out['thickness'] = (ds['thickness'].dims, th_new, ds['thickness'].attrs)
    return _stamp(out, exp_id, scale_factor=factor)


# ── Velocity experiments (Exp8-13) ─────────────────────────────────────────

def _apply_vmag_perturbation(ds, vx, vy, valid, delta_vmag):
    """Add delta_vmag (same shape) to the velocity MAGNITUDE, rescaling vx/vy to
    preserve the original direction exactly (delta_vmag may be negative)."""
    vmag = np.hypot(vx, vy)
    vmag_new = np.clip(vmag + delta_vmag, 0, None)
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.where(vmag > 1e-9, vmag_new / vmag, 0.0)
    vx_new = np.where(valid, vx * scale, np.nan)
    vy_new = np.where(valid, vy * scale, np.nan)
    out = ds.copy()
    out['vx'] = (ds['vx'].dims, vx_new, ds['vx'].attrs)
    out['vy'] = (ds['vy'].dims, vy_new, ds['vy'].attrs)
    return out


def exp_velocity_mag_noise(ds, exp_id, corr_m, amp_abs=None, amp_frac=None, seed=0):
    """Magnitude-only noise (direction preserved): Exp6/7 (absolute amplitude,
    m/yr) or Exp8 (amplitude = amp_frac * local speed)."""
    px = pixel_size(ds)
    vx, vy = ds['vx'].values, ds['vy'].values
    valid = glacier_footprint(ds)
    g = random_field(vx.shape, valid, corr_m, px, seed)
    amp = amp_abs if amp_abs is not None else amp_frac * np.hypot(vx, vy)
    out = _apply_vmag_perturbation(ds, vx, vy, valid, g * amp)
    if amp_abs is not None:
        return _stamp(out, exp_id, corr_length_m=corr_m, amplitude_m_per_yr=amp_abs, rng_seed=seed)
    return _stamp(out, exp_id, corr_length_m=corr_m, amplitude_fraction=amp_frac, rng_seed=seed)


def exp_velocity_xy_noise(ds, exp_id, corr_m, amp_frac, seed):
    """Exp9/10: independent noise on vx and vy separately (amplitude = amp_frac
    * that component's own local value) — direction is NOT preserved."""
    px = pixel_size(ds)
    vx, vy = ds['vx'].values, ds['vy'].values
    valid = glacier_footprint(ds)
    gx = random_field(vx.shape, valid, corr_m, px, seed)
    gy = random_field(vy.shape, valid, corr_m, px, seed + 1)   # independent field
    vx_new = np.where(valid, vx + gx * amp_frac * np.abs(vx), np.nan)
    vy_new = np.where(valid, vy + gy * amp_frac * np.abs(vy), np.nan)
    out = ds.copy()
    out['vx'] = (ds['vx'].dims, vx_new, ds['vx'].attrs)
    out['vy'] = (ds['vy'].dims, vy_new, ds['vy'].attrs)
    return _stamp(out, exp_id, corr_length_m=corr_m, amplitude_fraction=amp_frac,
                 rng_seed=seed, rng_seed_y=seed + 1)


def exp_velocity_altitude_ramp(ds, exp_id, frac_low, frac_high):
    """Exp11: additive bias to velocity magnitude, linearly ramped with surface elevation (DEM) between
    frac_low * mean speed at the lowest elevation and frac_high * mean speed at the highest. Direction
    preserved."""
    vx, vy = ds['vx'].values, ds['vy'].values
    valid = glacier_footprint(ds)
    vmag = np.hypot(vx, vy)
    mean_v = float(np.nanmean(vmag[valid]))
    elev = ds['DEM'].values
    e_lo, e_hi = np.nanmin(elev[valid]), np.nanmax(elev[valid])
    t = np.clip((elev - e_lo) / max(e_hi - e_lo, 1e-9), 0.0, 1.0)
    delta = (frac_low + t * (frac_high - frac_low)) * mean_v
    out = _apply_vmag_perturbation(ds, vx, vy, valid, delta)
    return _stamp(out, exp_id, mean_velocity=mean_v, elev_min=float(e_lo), elev_max=float(e_hi),
                 frac_at_low_elev=frac_low, frac_at_high_elev=frac_high)


# ── Resolution experiments (Exp14-15) — whole dataset ──────────────────────

def _coarsen_whole(ds, factor):
    """Block-mean every variable that has both x and y dims by `factor` in each;
    variables without both dims (e.g. a scalar CRS/grid_mapping var) pass
    through untouched. NaN-safe (skipna)."""
    if factor <= 1:
        return ds.copy()
    return ds.coarsen(x=factor, y=factor, boundary='trim').mean(skipna=True)


def exp_resolution_factor(ds, exp_id, factor):
    """Exp12: reduce resolution by `factor` in x and y (block-mean)."""
    out = _coarsen_whole(ds, factor)
    return _stamp(out, exp_id, coarsen_factor=factor)


def exp_resolution_target(ds, exp_id, target_px_m):
    """Exp13: regrid the whole dataset to ~target_px_m posting via block-mean
    (nearest integer factor of the native pixel size)."""
    px = pixel_size(ds)
    factor = max(1, round(target_px_m / px))
    out = _coarsen_whole(ds, factor)
    return _stamp(out, exp_id, target_resolution_m=target_px_m, coarsen_factor=factor, native_resolution_m=px)


# ── Experiment registry (keys = experiment numbers, Exp3..Exp15) ──

def build_experiments(seed_base):
    """seed_base: deterministic per-glacier offset so each glacier gets its own
    (but reproducible) random fields."""
    return {
        3:  lambda ds: exp_thickness_noise(ds, 3, corr_factor=1, amp_frac=0.30, amp_relative_to='local', seed=seed_base + 3),
        4:  lambda ds: exp_thickness_noise(ds, 4, corr_factor=2, amp_frac=0.30, amp_relative_to='local', seed=seed_base + 4),
        5:  lambda ds: exp_thickness_noise(ds, 5, corr_factor=1, amp_frac=0.16, amp_relative_to='mean', seed=seed_base + 5),
        6:  lambda ds: exp_thickness_scale(ds, 6, factor=1.30),
        7:  lambda ds: exp_thickness_scale(ds, 7, factor=0.70),
        8:  lambda ds: exp_velocity_mag_noise(ds, 8, corr_m=100.0, amp_abs=10.0, seed=seed_base + 8),
        9:  lambda ds: exp_velocity_mag_noise(ds, 9, corr_m=1000.0, amp_abs=10.0, seed=seed_base + 9),
        10: lambda ds: exp_velocity_mag_noise(ds, 10, corr_m=100.0, amp_frac=0.10, seed=seed_base + 10),
        11: lambda ds: exp_velocity_xy_noise(ds, 11, corr_m=100.0, amp_frac=0.10, seed=seed_base + 11),
        12: lambda ds: exp_velocity_xy_noise(ds, 12, corr_m=1000.0, amp_frac=0.10, seed=seed_base + 12),
        13: lambda ds: exp_velocity_altitude_ramp(ds, 13, frac_low=0.10, frac_high=0.50),
        14: lambda ds: exp_resolution_factor(ds, 14, factor=3),
        15: lambda ds: exp_resolution_target(ds, 15, target_px_m=100.0),
    }


# ── driver ──────────────────────────────────────────────────────────────────

ENCODING_KWARGS = dict(zlib=True, complevel=4)


def _sanitize_encoding(ds):
    """The source files carry stale/conflicting FillValue + missing_value
    metadata on some variables (e.g. spatial_ref). Clear both the
    leftover encoding dict and those two attrs on every variable so to_netcdf
    works."""
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding.clear()
        for k in ('_FillValue', 'missing_value'):
            ds[name].attrs.pop(k, None)


def process_file(path):
    name = os.path.splitext(os.path.basename(path))[0]

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    ds = xr.open_dataset(path)
    for v in ('thickness', 'vx', 'vy', 'DEM'):
        if v not in ds:
            raise ValueError(f"{path}: missing required variable '{v}'")

    seed_base = zlib.crc32(name.encode()) % 10**6   # deterministic per glacier name
    experiments = build_experiments(seed_base)
    print(f"{os.path.basename(path)} -> glacier '{name}': "
          f"{ds.sizes['y']}x{ds.sizes['x']} @ {pixel_size(ds):g} m, "
          f"{glacier_footprint(ds).sum()} on-glacier px")
    for exp, fn in experiments.items():
        out = fn(ds)
        _sanitize_encoding(out)
        out_path = os.path.join(out_dir, f'{WP}_{name}_exp{exp:02d}.nc')
        encoding = {v: ENCODING_KWARGS for v in out.data_vars
                   if out[v].ndim > 0}
        out.to_netcdf(out_path, encoding=encoding)
        print(f"  Exp{exp:<2d} -> {out_path}")
    ds.close()


def _resolve_input(arg):
    """Accept a glacier name ('athabasca'), a filename ('athabasca.nc'), or a
    path; bare names/filenames are looked up next to the script."""
    for cand in (arg, arg + '.nc', os.path.join(HERE, arg), os.path.join(HERE, arg + '.nc')):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No NetCDF found for {arg!r} (tried '{arg}[.nc]' "
                            f"as given and in {HERE})")


def main():
    args = sys.argv[1:]
    if args:
        paths = [_resolve_input(a) for a in args]
    else:
        paths = sorted(glob.glob(os.path.join(HERE, '*.nc')),
                       key=lambda p: os.path.basename(p).lower())
    if not paths:
        print("No input files given and no .nc files found in this directory.")
        sys.exit(1)
    for p in paths:
        process_file(p)


if __name__ == '__main__':
    main()
