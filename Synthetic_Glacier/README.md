# Synthetic Glacier Pipeline

Icepack-based synthetic half-pipe glacier simulation with two bed variants (flat and wave),
plus input generation for the CONTINUIX inversion framework.

---

## Pipeline overview

```
domain_config.py          ← edit this to change geometry, run length, or wave bumps
        │
        ├── 1_run_simulation.py           → ./output/        (flat bed)
        └── 1_run_simulation_wave_bed.py  → ./output_wave/   (wave bed)
                │
                ├── 2_generate_inputs.py           reads ./output/
                └── 2_generate_inputs_wave_bed.py  reads ./output_wave/
                        │
                        └── ./netcdfs/   (unperturbed, perturbed, perturbations, coarsened)
```

Run order: script 1 first, then script 2. Both versions (wave, no wave) are independent.

---

## What to modify

### `domain_config.py` — the only file you should normally edit

| Parameter | What it controls |
|-----------|-----------------|
| `n` | Grid rows (keep odd for symmetric NaN padding) |
| `w` | Glacier half-width in cells |
| `max_altitude` | Upper rim elevation (m) |
| `max_depth` | Pipe depth below rim at centreline (m) |
| `slope` | Longitudinal slope (m/m) |
| `glacier_length` | Along-flow extent (m) |
| `ela` | Equilibrium line altitude (m) |
| `max_a` | Maximum accumulation rate (m/yr) |
| `da_ds` | Mass balance lapse rate (m/yr per m elevation) |
| `n_dt` | Time step (yr) |
| `n_timesteps` | Number of time steps |
| `wave_amplitude` | Peak height of each Gaussian bed bump (m) |
| `wave_sigma` | Gaussian half-width — controls bump spread (m) |
| `wave_bump_centers` | Along-flow x positions of bumps (m) |

Changes here propagate automatically to all four scripts.

---

## Output layout

```
./data/            dem.tif (bed raster sampled by Icepack)
./output/          flat-bed run:  synthetic.msh, b/h/u/a/flux_div/dhdt .pvd + .vtu
./output_wave/     wave-bed run:  same structure
./figures/         all plots
                     bed_config.png / bed_config_wave.png
                     thickness_evolution.png / thickness_evolution_wave.png
                     snapshots.png / snapshots_wave.png
                     quick_look.png / quick_look_wave.png
                     perturbations.png / perturbations_wave.png
./netcdfs/         flat bed:   unperturbed.nc   perturbed.nc   perturbations.nc   coarsened.nc
                   wave bed:   unperturbed_wave.nc   perturbed_wave.nc   perturbations_wave.nc   coarsened_wave.nc
```

Wave and flat outputs coexist — running one does not overwrite the other.

---

## NetCDF contents

All files use `(y, x)` float32 arrays on the full raster grid. Coordinates: `x`, `y` (m), and `elevation_band` (m, 10 m step) where applicable.

| File | Variables |
|------|-----------|
| `unperturbed[_wave].nc` | bed, surface, thickness, velocity\_x/y/mag, smb, dhdt, flux\_div + uncertainty fields (velocity\_unc, thickness\_unc, smb\_unc, dhdt\_unc, flux\_div\_unc) + elevation\_bin |
| `perturbed[_wave].nc` | All 15 unperturbed fields + 10 perturbation variants (see below) |
| `perturbations[_wave].nc` | Individual perturbation factor arrays (speckle\_x/y, bias\_ramp, vel\_grf\_factor, h\_grf\_factor\_100/1000m, x/y\_frac, + all perturbed fields) |
| `coarsened[_wave].nc` | 9 base fields (no uncertainties) at 10× pixel size |

**Perturbation variants in `perturbed.nc`:**

| Name | Type | Detail |
|------|------|--------|
| `velocity_x/y_speckle` | Additive | U[−15, 15] m/yr per pixel, independent on x and y |
| `velocity_x/y_biased` | Multiplicative ramp | 10 → 0% from head to toe (along-flow) |
| `velocity_x/y_autocorr` | GRF | Shared factor, corr = 2500 m, std = 10% |
| `thickness_bias_25/50` | Multiplicative ramp | Combined x·y ramp: 25% or 50% max corner bias |
| `thickness_autocorr_100m/1000m` | GRF | corr = 100 m or 1000 m, std = 15% |

All perturbation draws use `seed=42` for reproducibility.

---

## Quick start

```bash
# Flat bed
python3 1_run_simulation.py
python3 2_generate_inputs.py

# Wave bed
python3 1_run_simulation_wave_bed.py
python3 2_generate_inputs_wave_bed.py
```

Must be run inside the Firedrake environment (Icepack dependency).
