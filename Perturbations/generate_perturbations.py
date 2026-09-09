"""
Perturbation generator.

Reads NetCDF named after the glacier Exp01_<glaciername>.nc.
Filename model follows ExpNN_<glaciername>.nc

The glacier footprint is taken as np.isfinite(thickness) — the "mask" variable
in the sample files does NOT encode a glacier boundary (it holds acquisition
years / unrelated continuous values), so it is never used for that purpose,
only passed through as-is.

Random fields are seeded deterministically from glacier number and experiment
number, so re-running this script reproduces byte-identical output; the exact
seed and every perturbation parameter are recorded as NetCDF attributes.

Usage:
    python generate_perturbations.py Exp01_<glaciername>.nc
    python generate_perturbations.py            # all "Exp01_<glaciername>.nc" files in inputs dir
"""
import os
import re
import sys
import glob
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, 'output')


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
    return np.isfinite(ds['THK'].values)


def glacier_name_from_path(path):
    """Extract glacier name from an input filename like Exp01_glaciername.nc."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r'^(?i:Exp\d{2})_(.+)$', stem)
    if match is not None:
        return match.group(1)
    if stem.isdigit():
        return stem
    raise ValueError(f"'{os.path.basename(path)}': filename must be Exp01_glaciername.nc")


# ── provenance ──────────────────────────────────────────────────────────────

def _stamp(ds, exp_id, **params):
    ds = ds.copy()
    ds.attrs['perturbation_experiment'] = f'Exp{exp_id:02d}'
    return ds


# ── Thickness experiments (Exp03-07) ───────────────────────────────────────

def exp_thickness_noise(ds, exp_id, corr_factor, amp_frac, amp_relative_to, seed):
    """Additive spatially-correlated thickness noise.
    corr_factor: correlation length = corr_factor * mean glacier thickness (m).
    amp_relative_to: 'local' -> amp_frac * this pixel's own thickness (Exp1/2);
                     'mean'  -> amp_frac * glacier-mean thickness, uniform (Exp3)."""
    px = pixel_size(ds)
    th = ds['THK'].values
    valid = glacier_footprint(ds)
    mean_th = float(np.nanmean(th[valid]))
    corr_m = corr_factor * mean_th
    g = random_field(th.shape, valid, corr_m, px, seed)
    amp = amp_frac * th if amp_relative_to == 'local' else amp_frac * mean_th
    th_new = np.where(valid, np.clip(th + g * amp, 0, None), np.nan)
    out = ds.copy()
    out['THK'] = (ds['THK'].dims, th_new, ds['THK'].attrs)
    return _stamp(out, exp_id, corr_length_m=corr_m, corr_factor_x_mean_thickness=corr_factor,
                 amplitude_fraction=amp_frac, amplitude_relative_to=amp_relative_to, rng_seed=seed)


def exp_thickness_scale(ds, exp_id, factor):
    """Deterministic multiplicative thickness scaling (Exp4/5) — no randomness."""
    th = ds['THK'].values
    valid = glacier_footprint(ds)
    th_new = np.where(valid, th * factor, np.nan)
    out = ds.copy()
    out['THK'] = (ds['THK'].dims, th_new, ds['THK'].attrs)
    return _stamp(out, exp_id, scale_factor=factor)


# ── Velocity experiments (Exp08-13) ───────────────────────────────────────

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
    out['VX'] = (ds['VX'].dims, vx_new, ds['VX'].attrs)
    out['VY'] = (ds['VY'].dims, vy_new, ds['VY'].attrs)
    return out


def exp_velocity_mag_noise(ds, exp_id, corr_m, amp_abs=None, amp_frac=None, seed=0):
    """Magnitude-only noise (direction preserved): Exp6/7 (absolute amplitude,
    m/yr) or Exp8 (amplitude = amp_frac * local speed)."""
    px = pixel_size(ds)
    vx, vy = ds['VX'].values, ds['VY'].values
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
    vx, vy = ds['VX'].values, ds['VY'].values
    valid = glacier_footprint(ds)
    gx = random_field(vx.shape, valid, corr_m, px, seed)
    gy = random_field(vy.shape, valid, corr_m, px, seed + 1)   # independent field
    vx_new = np.where(valid, vx + gx * amp_frac * np.abs(vx), np.nan)
    vy_new = np.where(valid, vy + gy * amp_frac * np.abs(vy), np.nan)
    out = ds.copy()
    out['VX'] = (ds['VX'].dims, vx_new, ds['VX'].attrs)
    out['VY'] = (ds['VY'].dims, vy_new, ds['VY'].attrs)
    return _stamp(out, exp_id, corr_length_m=corr_m, amplitude_fraction=amp_frac,
                 rng_seed=seed, rng_seed_y=seed + 1)


def exp_velocity_altitude_ramp(ds, exp_id, frac_low, frac_high):
    """Exp11: additive bias to velocity magnitude, linearly ramped with surface elevation (DEM) between
    frac_low * mean speed at the lowest elevation and frac_high * mean speed at the highest. Direction
    preserved."""
    vx, vy = ds['VX'].values, ds['VY'].values
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


# ── Resolution experiments (Exp12-15) — whole dataset ─────────────────────

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


# ── Experiment registry (Z = 3..15; 16-17 reserved for future experiments) ──

def build_experiments():
    return {
        3:  lambda ds: exp_thickness_noise(ds, 3, corr_factor=1, amp_frac=0.30, amp_relative_to='local', seed=100 + 3),
        4:  lambda ds: exp_thickness_noise(ds, 4, corr_factor=2, amp_frac=0.30, amp_relative_to='local', seed=100 + 4),
        5:  lambda ds: exp_thickness_noise(ds, 5, corr_factor=1, amp_frac=0.16, amp_relative_to='mean', seed=100 + 5),
        6:  lambda ds: exp_thickness_scale(ds, 6, factor=1.30),
        7:  lambda ds: exp_thickness_scale(ds, 7, factor=0.70),
        8:  lambda ds: exp_velocity_mag_noise(ds, 8, corr_m=100.0, amp_abs=10.0, seed=100 + 8),
        9:  lambda ds: exp_velocity_mag_noise(ds, 9, corr_m=1000.0, amp_abs=10.0, seed=100 + 9),
        10: lambda ds: exp_velocity_mag_noise(ds, 10, corr_m=100.0, amp_frac=0.10, seed=100 + 10),
        11: lambda ds: exp_velocity_xy_noise(ds, 11, corr_m=100.0, amp_frac=0.10, seed=100 + 11),
        12: lambda ds: exp_velocity_xy_noise(ds, 12, corr_m=1000.0, amp_frac=0.10, seed=100 + 12),
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
    glacier_name = glacier_name_from_path(path)
    os.makedirs(OUT_ROOT, exist_ok=True)

    ds = xr.open_dataset(path)
    for v in ('THK', 'VX', 'VY', 'DEM'):
        if v not in ds:
            raise ValueError(f"{path}: missing required variable '{v}'")

    experiments = build_experiments()
    print(f"{os.path.basename(path)} -> glacier {glacier_name}: "
          f"{ds.sizes['y']}x{ds.sizes['x']} @ {pixel_size(ds):g} m, "
          f"{glacier_footprint(ds).sum()} on-glacier px")
    for z, fn in experiments.items():
        out = fn(ds)
        _sanitize_encoding(out)
        out_path = os.path.join(OUT_ROOT, f'EXP{z:02d}_{glacier_name}.nc')
        encoding = {v: ENCODING_KWARGS for v in out.data_vars
                   if out[v].ndim > 0}
        out.to_netcdf(out_path, encoding=encoding)
        print(f"  Exp{z:02d} -> {out_path}")
    ds.close()


def main():
    args = sys.argv[1:]
    if args:
        paths = args
    else:
        # inputs_dir = os.path.join(HERE, 'inputs') ## Maaike: adjusted; input folder is EXP01 folder
        inputs_dir = os.path.join(HERE, 'EXP01')    ## Maaike: adjusted; input folder is EXP01 folder
          
        paths = sorted(glob.glob(os.path.join(inputs_dir, 'Exp*.nc')) +
                       glob.glob(os.path.join(inputs_dir, 'exp*.nc')) +
                       glob.glob(os.path.join(inputs_dir, '*.nc')),
                       key=lambda p: os.path.splitext(os.path.basename(p))[0])
        paths = [p for p in paths
                if re.match(r'^(?i:Exp\d{2})_.+\.nc$', os.path.basename(p)) or
                   os.path.splitext(os.path.basename(p))[0].isdigit()]
        print(f"Found {len(paths)} input files in {inputs_dir}")
    if not paths:
        print("No input files given and no matching .nc files found in the inputs directory.")
        sys.exit(1)
    
    for p in paths:
        process_file(p)


if __name__ == '__main__':
    main()
