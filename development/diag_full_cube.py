import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time as time_mod
import sys
sys.path.insert(0, '/Users/rri38/Documents/work/code/tess/tessreduce/development')
from run_diag import adaptive_medfilt_3d, _get_segments

_dev = '/Users/rri38/Documents/work/code/tess/tessreduce/development/'
_base = "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors/FFI_Cadence/"

data = np.load(f'{_dev}test_bkg2.npy')
time = np.load(f'{_dev}test_bkg2_time.npy')
print(f"Data shape: {data.shape}")

_df = pd.read_csv(f"{_base}TessVectors_S034_C1_FFI.csv", comment='#')
earth_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Earth_Camera_Angle'].values)
moon_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Moon_Camera_Angle'].values)

t0 = time_mod.time()
smoothed, windows, variability, _ = adaptive_medfilt_3d(
    data,
    time=time,
    metric='deviation',
    coarse_windows=(11, 21, 51, 101),
    low_pct=5,
    high_pct=80,
    local_norm_window=201,
    w_min=3,
    w_max=51,
    n_jobs=-1,
    window_smooth_size=1,
    earth_angle=earth_angle,
    moon_angle=moon_angle,
    scatter_angle_thresh=50.0,
)
elapsed = time_mod.time() - t0
print(f"Done in {elapsed:.1f}s")

residual = data - smoothed
mean_res = np.nanmean(np.abs(residual))
print(f"Full cube residual: mean={mean_res:.5f}, std={np.nanstd(residual):.5f}")
print(f"Window: mean={windows.mean():.1f}, std={windows.std():.1f}, frac_wmax={np.mean(windows==51):.3f}")

segs = _get_segments(time, gap_thresh=3.0)

# ── Figure 1: representative pixel (20,20) ─────────────────────────────────
px, py = 20, 20
raw1d = data[:, px, py]
sm1d = smoothed[:, px, py]
win1d = windows[:, px, py]
res1d = raw1d - sm1d

def with_breaks(t, y, segments):
    t_out, y_out = [], []
    for s, e in segments:
        t_out.extend(t[s:e].tolist())
        y_out.extend(y[s:e].tolist())
        t_out.append(np.nan)
        y_out.append(np.nan)
    return np.array(t_out), np.array(y_out)

t_raw, raw_r = with_breaks(time, raw1d, segs)
t_sm, sm_r = with_breaks(time, sm1d, segs)
t_w, win_r = with_breaks(time, win1d.astype(float), segs)
t_r, res_r = with_breaks(time, res1d, segs)

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('Full cube run — test_bkg2, pixel (20,20)  [w_max=51, scatter_angle_thresh=50]', fontsize=11)

ax = axes[0]
ax.plot(t_raw, raw_r, lw=0.6, color='steelblue', alpha=0.7, label='Raw')
ax.plot(t_sm, sm_r, lw=1.2, color='firebrick', label='Smoothed')
ax.set_ylabel('Flux'); ax.set_title('Raw + smoothed'); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t_r, res_r, lw=0.7, color='darkgreen', label='Residual (raw − smooth)')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_ylabel('Residual'); ax.set_title('Residual'); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(t_r, np.abs(res_r), lw=0.7, color='purple', label='|Residual|')
ax.set_ylabel('|Residual|'); ax.set_title('Absolute residual'); ax.legend(fontsize=8)

ax = axes[3]
ax.plot(t_w, win_r, lw=0.8, color='darkorange', label='Window size')
ax.set_ylabel('Window size'); ax.set_xlabel('Time (BTJD)')
ax.set_title('Adaptive window size'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{_dev}residual_fullcube_test_bkg2_px.png', dpi=120)
plt.close()
print("Saved residual_fullcube_test_bkg2_px.png")

# ── Figure 2: mean residual image and window image ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Full cube — test_bkg2 spatial summary', fontsize=12)

mean_abs_res = np.nanmean(np.abs(residual), axis=0)
mean_win = windows.mean(axis=0)
mean_var = variability.mean(axis=0)

im0 = axes[0].imshow(mean_abs_res, origin='lower', cmap='inferno')
axes[0].set_title('Mean |residual| per pixel')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(mean_win, origin='lower', cmap='viridis', vmin=3, vmax=51)
axes[1].set_title('Mean window size per pixel')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(mean_var, origin='lower', cmap='plasma')
axes[2].set_title('Mean variability metric per pixel')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f'{_dev}residual_fullcube_test_bkg2_spatial.png', dpi=120)
plt.close()
print("Saved residual_fullcube_test_bkg2_spatial.png")
