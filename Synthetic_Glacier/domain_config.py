"""
Shared domain configuration for the synthetic half-pipe glacier.

Single source of truth for all domain parameters, derived grid variables,
mass balance settings, and the bed array. Import this in both
1_Synthetic_glacier_run.ipynb and 2_Read_results.ipynb so that any
change to the geometry propagates to both notebooks automatically.
"""
import numpy as np

# ── Domain parameters ──────────────────────────────────────────────────────
n              = 1001    # odd so NaN padding is equal above/below (200 rows each side)
w              = 300     # glacier half-width (cells)
max_altitude   = 3000.0  # upper rim elevation (m)
max_depth      = 400.0   # pipe depth below rim at centreline (m)
slope          = 0.25    # longitudinal slope (m/m)
glacier_length = 5000.0  # along-flow extent (m)

# ── Mass balance parameters ────────────────────────────────────────────────
ela   = max_altitude - 650.0  # equilibrium line altitude (m)
max_a = 5.0                   # maximum accumulation rate (m/yr)
da_ds = 1.0 / 100.0           # mass balance lapse rate (m/yr per m elevation)

# ── Derived grid ───────────────────────────────────────────────────────────
rows = n
cols = 4 * n
x0   = 0.0
y0   = float(rows) - 0.5   # half-pixel shift so pixel centres land on integer y-values; y_coords[500] = 500.0
dx   = glacier_length / cols
dy   = -1.0   # negative: numpy origin is top-left, geo origin is bottom-left

x_coords = x0 + (np.arange(cols) + 0.5) * dx
y_coords = y0 + (np.arange(rows) + 0.5) * dy
X, Y = np.meshgrid(x_coords, y_coords)

# ── Icepack run parameters ───────────────────────────────────────────────────────────
n_dt = 0.1
n_timesteps = 4000


# ── Bed geometry ───────────────────────────────────────────────────────────
def create_inclined_pipe_matrix(n, w, max_altitude, max_depth, slope, glacier_length):
    """
    Build the 2-D bed-elevation array for the synthetic half-pipe glacier.

    The cross-section is parabolic: the pipe rim sits at
    (max_altitude - max_depth) plus the parabolic offset, deepening to
    (max_altitude - max_depth) at the centreline and rising back to
    max_altitude at the edges. A constant longitudinal slope is applied so
    the bed descends uniformly from upper-left to lower-right. Cells outside
    the pipe half-width w are left as NaN.

    Parameters
    ----------
    n              : int   -- number of rows (cross-flow direction)
    w              : int   -- half-width of the pipe in matrix cells
    max_altitude   : float -- elevation at the upper rim (m)
    max_depth      : float -- depth of the pipe centre below the rim (m)
    slope          : float -- longitudinal slope (m/m)
    glacier_length : float -- total along-flow length (m)

    Returns
    -------
    matrix : (n, 4n) float array -- bed elevation in metres; NaN outside pipe
    """
    rows_ = n
    cols_ = 4 * n
    matrix = np.full((rows_, cols_), np.nan)
    middle_row = rows_ // 2
    start_row  = middle_row - w
    end_row    = middle_row + w
    for i in range(start_row, end_row + 1):
        row_offset       = abs(i - middle_row)
        parabolic_factor = (row_offset / w) ** 2 if w > 0 else 0
        base_alt         = (max_altitude - max_depth) + parabolic_factor * max_depth
        for j in range(cols_):
            dist         = (j / (cols_ - 1)) * glacier_length if cols_ > 1 else 0
            matrix[i, j] = base_alt - dist * slope
    return matrix


bed = create_inclined_pipe_matrix(n, w, max_altitude, max_depth, slope, glacier_length)

# ── Wave bed: localised Gaussian bumps ─────────────────────────────────────
wave_amplitude = 50.0    # m  peak height of each bump
wave_sigma     = 150.0   # m  Gaussian half-width
wave_bump_centers = [1000.0, 2500.0, 3000.0]  # along-flow x positions (m)

_wave_pert = np.zeros_like(bed)
for _xc in wave_bump_centers:
    _wave_pert += wave_amplitude * np.exp(-0.5 * ((X - _xc) / wave_sigma) ** 2)

bed_wave = np.where(~np.isnan(bed), bed + _wave_pert, np.nan)
