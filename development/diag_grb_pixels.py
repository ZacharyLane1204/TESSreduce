import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, '/Users/rri38/Documents/work/code/tess/tessreduce/development')
from run_diag import adaptive_medfilt_3d, _get_segments

_dev = '/Users/rri38/Documents/work/code/tess/tessreduce/development/'
_base = "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors/FFI_Cadence/"

data = np.load(f'{_dev}test_bkg2.npy')
time = np.load(f'{_dev}test_bkg2_time.npy')
T = len(time)

_df = pd.read_csv(f"{_base}TessVectors_S034_C1_FFI.csv", comment='#')
earth_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Earth_Camera_Angle'].values)
moon_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Moon_Camera_Angle'].values)

# Central 3×3 pixels around (25,25)
cx, cy = 25, 25
pixels = [(cx+dx, cy+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

segs = _get_segments(time, gap_thresh=3.0)

def with_breaks(t, y, segments):
    t_out, y_out = [], []
    for s, e in segments:
        t_out.extend(t[s:e].tolist())
        y_out.extend(y[s:e].tolist())
        t_out.append(np.nan)
        y_out.append(np.nan)
    return np.array(t_out), np.array(y_out)

fig, axes = plt.subplots(9, 3, figsize=(20, 30))
fig.suptitle('GRB central 3×3 pixels — test_bkg2  [w_max=51, scatter_angle_thresh=50]', fontsize=13)

for idx, (px, py) in enumerate(pixels):
    cube = data[:, px:px+1, py:py+1]
    raw = cube[:, 0, 0]

    sm, win, _, _ = adaptive_medfilt_3d(
        cube, time=time, metric='deviation',
        coarse_windows=(11, 21, 51, 101),
        low_pct=5, high_pct=80,
        local_norm_window=201,
        w_min=3, w_max=51, n_jobs=1,
        window_smooth_size=1,
        earth_angle=earth_angle,
        moon_angle=moon_angle,
        scatter_angle_thresh=50.0,
    )
    sm1d = sm[:, 0, 0]
    win1d = win[:, 0, 0]
    res1d = raw - sm1d

    t_raw, raw_r = with_breaks(time, raw, segs)
    t_sm, sm_r = with_breaks(time, sm1d, segs)
    t_w, win_r = with_breaks(time, win1d.astype(float), segs)
    t_r, res_r = with_breaks(time, res1d, segs)

    row = idx
    ax0 = axes[row, 0]
    ax0.plot(t_raw, raw_r, lw=0.5, color='steelblue', alpha=0.6, label='Raw')
    ax0.plot(t_sm, sm_r, lw=1.0, color='firebrick', label='Smoothed')
    ax0.set_ylabel('Flux', fontsize=7)
    ax0.set_title(f'Pixel ({px},{py}) — raw + smoothed', fontsize=8)
    ax0.legend(fontsize=6)
    ax0.tick_params(labelsize=6)

    ax1 = axes[row, 1]
    ax1.plot(t_r, res_r, lw=0.5, color='darkgreen')
    ax1.axhline(0, color='k', lw=0.4, ls='--')
    ax1.set_ylabel('Residual', fontsize=7)
    ax1.set_title(f'Residual  (mean|res|={np.nanmean(np.abs(res1d)):.4f})', fontsize=8)
    ax1.tick_params(labelsize=6)

    ax2 = axes[row, 2]
    ax2.plot(t_w, win_r, lw=0.6, color='darkorange')
    ax2.set_ylabel('Window', fontsize=7)
    ax2.set_title(f'Window size  (mean={win1d.mean():.1f})', fontsize=8)
    ax2.tick_params(labelsize=6)

    print(f"  ({px},{py}): res={np.nanmean(np.abs(res1d)):.5f}, win_mean={win1d.mean():.1f}")

for ax in axes[-1]:
    ax.set_xlabel('Time (MJD)', fontsize=7)

plt.tight_layout()
plt.savefig(f'{_dev}diag_grb_central_pixels.png', dpi=100)
plt.close()
print("Saved diag_grb_central_pixels.png")
