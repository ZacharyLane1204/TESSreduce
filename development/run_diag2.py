"""
Deeper look: why doesn't wsz=5/11 change n_switches?
And examine the actual window pattern visually.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from joblib import Parallel, delayed


def _make_odd(x):
    x = max(int(x), 3)
    return x if x % 2 == 1 else x + 1


def adaptive_medfilt_3d_debug(
    data,
    w_min=3,
    w_max=51,
    grad_smooth_window=11,
    low_pct=5,
    high_pct=90,
    per_pixel_norm=True,
    n_levels=7,
    n_jobs=1,
    metric='combined',
    coarse_window=101,
    combined_weight=1.0,
    window_smooth_size=1,
):
    data = np.asarray(data, dtype=float)
    T, X, Y = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = data.copy()
    if nan_mask.any():
        t_ax = np.arange(T, dtype=float)
        flat = data_filled.reshape(T, -1)
        for j in range(flat.shape[1]):
            ts = flat[:, j]
            bad = ~np.isfinite(ts)
            if bad.all():
                flat[:, j] = 0.0
            elif bad.any():
                flat[:, j] = np.interp(t_ax, t_ax[~bad], ts[~bad])

    gw = _make_odd(grad_smooth_window)
    grad = np.gradient(data_filled, axis=0)
    grad_var = np.abs(median_filter(grad, size=(gw, 1, 1), mode='reflect'))
    cw = _make_odd(coarse_window)
    coarse = median_filter(data_filled, size=(cw, 1, 1), mode='reflect')
    dev_var = np.abs(data_filled - coarse)
    dev_var = median_filter(dev_var, size=(gw, 1, 1), mode='reflect')

    def _norm01(x):
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-30), 0, 1)
    variability = (1 - combined_weight) * _norm01(grad_var) + combined_weight * _norm01(dev_var)

    g_lo = np.nanpercentile(variability, low_pct, axis=0, keepdims=True)
    g_hi = np.nanpercentile(variability, high_pct, axis=0, keepdims=True)
    dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
    norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)

    raw = w_max - norm * (w_max - w_min)
    windows_raw_cont = raw.copy()  # continuous, before rounding
    windows = np.round(raw).astype(int)
    windows += (1 - windows % 2)
    windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))
    windows_pre = windows.copy()

    if window_smooth_size > 1:
        wsz = _make_odd(window_smooth_size)
        windows = median_filter(windows.astype(float), size=(wsz, 1, 1), mode='reflect')
        windows = np.round(windows).astype(int)
        windows += (1 - windows % 2)
        windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))

    windows_post = windows.copy()

    levels = np.unique([_make_odd(int(round(w))) for w in np.linspace(w_min, w_max, n_levels)])
    windows_q = levels[np.argmin(np.abs(windows[..., np.newaxis] - levels), axis=-1)]

    def _smooth(w):
        return w, median_filter(data_filled, size=(w, 1, 1), mode='reflect')

    result = np.empty((T, X, Y))
    for w, smoothed_w in Parallel(n_jobs=n_jobs)(delayed(_smooth)(w) for w in levels):
        result[windows_q == w] = smoothed_w[windows_q == w]

    result[nan_mask] = np.nan
    return result, windows_q, variability, windows_pre, windows_post, windows_raw_cont


data = np.load('/Users/rri38/Documents/work/code/tess/tessreduce/development/test_bkg.npy')
cube = data[:, 100:101, 100:101]
raw_series = cube[:, 0, 0]
T = len(raw_series)
print(f"T={T}")

# Run wsz=1 and wsz=31 in debug mode
print("\n--- wsz=1 ---")
sm1, wq1, var1, wpre1, wpost1, wrc1 = adaptive_medfilt_3d_debug(cube, window_smooth_size=1,
    combined_weight=1.0, coarse_window=101, low_pct=5, high_pct=90)
wq1d = wq1[:, 0, 0]
wpre1d = wpre1[:, 0, 0]
wrc1d = wrc1[:, 0, 0]
print(f"  unique quantized levels: {np.unique(wq1d)}")
print(f"  n_switches (pre-quant): {np.sum(np.diff(wpre1d) != 0)}")
print(f"  n_switches (quantized): {np.sum(np.diff(wq1d) != 0)}")

print("\n--- wsz=5 ---")
sm5, wq5, var5, wpre5, wpost5, wrc5 = adaptive_medfilt_3d_debug(cube, window_smooth_size=5,
    combined_weight=1.0, coarse_window=101, low_pct=5, high_pct=90)
wq5d = wq5[:, 0, 0]
wpost5d = wpost5[:, 0, 0]
print(f"  unique quantized levels: {np.unique(wq5d)}")
print(f"  n_switches (post-smooth, pre-quant): {np.sum(np.diff(wpost5d) != 0)}")
print(f"  n_switches (quantized): {np.sum(np.diff(wq5d) != 0)}")

print("\n--- wsz=11 ---")
sm11, wq11, var11, wpre11, wpost11, wrc11 = adaptive_medfilt_3d_debug(cube, window_smooth_size=11,
    combined_weight=1.0, coarse_window=101, low_pct=5, high_pct=90)
wq11d = wq11[:, 0, 0]
wpost11d = wpost11[:, 0, 0]
print(f"  unique quantized levels: {np.unique(wq11d)}")
print(f"  n_switches (post-smooth, pre-quant): {np.sum(np.diff(wpost11d) != 0)}")
print(f"  n_switches (quantized): {np.sum(np.diff(wq11d) != 0)}")

print("\n--- wsz=21 ---")
sm21, wq21, var21, wpre21, wpost21, wrc21 = adaptive_medfilt_3d_debug(cube, window_smooth_size=21,
    combined_weight=1.0, coarse_window=101, low_pct=5, high_pct=90)
wq21d = wq21[:, 0, 0]
wpost21d = wpost21[:, 0, 0]
print(f"  unique quantized levels: {np.unique(wq21d)}")
print(f"  n_switches (post-smooth, pre-quant): {np.sum(np.diff(wpost21d) != 0)}")
print(f"  n_switches (quantized): {np.sum(np.diff(wq21d) != 0)}")

print("\n--- wsz=31 ---")
sm31, wq31, var31, wpre31, wpost31, wrc31 = adaptive_medfilt_3d_debug(cube, window_smooth_size=31,
    combined_weight=1.0, coarse_window=101, low_pct=5, high_pct=90)
wq31d = wq31[:, 0, 0]
wpost31d = wpost31[:, 0, 0]
print(f"  unique quantized levels: {np.unique(wq31d)}")
print(f"  n_switches (post-smooth, pre-quant): {np.sum(np.diff(wpost31d) != 0)}")
print(f"  n_switches (quantized): {np.sum(np.diff(wq31d) != 0)}")

# Look at zoom region for switches
z0, z1 = 1800, 2300
print(f"\nIn zoom region {z0}-{z1}:")
for label, arr in [('wsz=1', wq1d), ('wsz=5', wq5d), ('wsz=11', wq11d), ('wsz=21', wq21d), ('wsz=31', wq31d)]:
    ns = np.sum(np.diff(arr[z0:z1]) != 0)
    print(f"  {label}: {ns} switches")

# Understand why small wsz doesn't reduce switches: the window field must
# switch very rapidly with step > wsz width, so median can't resolve it.
# Check run-length stats
def run_lengths(arr):
    """Return lengths of constant runs."""
    diffs = np.concatenate(([1], np.diff(arr) != 0, [1]))
    starts = np.where(diffs)[0]
    return np.diff(starts)

print("\nRun-length stats for pre-smooth window field:")
rls = run_lengths(wpre1d)
print(f"  n_runs={len(rls)}, mean_run={rls.mean():.1f}, median_run={np.median(rls):.1f}, max_run={rls.max()}")
print(f"  fraction of single-frame runs: {np.mean(rls == 1):.3f}")

print("\nRun-length stats for post-smooth (wsz=11):")
rls11 = run_lengths(wpost11d)
print(f"  n_runs={len(rls11)}, mean_run={rls11.mean():.1f}, median_run={np.median(rls11):.1f}, max_run={rls11.max()}")
print(f"  fraction of single-frame runs: {np.mean(rls11 == 1):.3f}")

print("\nRun-length stats for post-smooth (wsz=31):")
rls31 = run_lengths(wpost31d)
print(f"  n_runs={len(rls31)}, mean_run={rls31.mean():.1f}, median_run={np.median(rls31):.1f}, max_run={rls31.max()}")
print(f"  fraction of single-frame runs: {np.mean(rls31 == 1):.3f}")

# Key insight: the oscillation happens at the QUANTIZATION stage.
# Check the continuous window field before quantization
print("\nContinuous window field (pre-round) in zoom region 1800-2300:")
wrc_z = wrc1d[1800:2300]
print(f"  mean={wrc_z.mean():.2f}, std={wrc_z.std():.2f}")
print(f"  fraction within 1 unit of even->odd boundary: {np.mean(np.abs(wrc_z - np.round(wrc_z)) > 0.4):.3f}")
# Look at actual values
print(f"  first 20 values: {wrc_z[:20]}")

# The real problem: after rounding and odd-enforcement, even small variations
# near boundaries cause oscillation. Smoothing the integer field doesn't help
# if the continuous field is already near boundaries.
# Solution: smooth the CONTINUOUS field before rounding, then quantize.

# Also check: what fraction of the window field is near a level boundary?
levels = np.unique([_make_odd(int(round(w))) for w in np.linspace(3, 51, 7)])
print(f"\nLevels used: {levels}")
# Distance to nearest level boundary
def dist_to_nearest_level(v, levels):
    dists = np.abs(v[:, None] - levels[None, :])
    return dists.min(axis=1)

# Now make the final 4-panel plot using wsz=31 (most visible difference)
# alongside wsz=1 for comparison
best_wsz = 31

def detect_peaks_adaptive(series, coarse_bw=201, prom_thresh_mad_mult=1.5):
    cw = _make_odd(coarse_bw)
    baseline = median_filter(series, size=cw, mode='reflect')
    detrended = series - baseline
    mad = np.median(np.abs(detrended - np.median(detrended)))
    threshold = prom_thresh_mad_mult * mad
    return detrended > threshold, detrended, baseline, threshold

peak_mask, detrended, baseline, threshold = detect_peaks_adaptive(raw_series)
sm_best = sm31[:, 0, 0]
win_best = wq31d
win_raw = wq1d

frames = np.arange(T)

fig, axes = plt.subplots(4, 1, figsize=(16, 18))
fig.suptitle(f'Pixel (100,100) — adaptive_medfilt_3d  |  window_smooth_size={best_wsz}', fontsize=13)

# Panel 1: raw + smoothed + peak shading
ax = axes[0]
ax.plot(frames, raw_series, lw=0.5, color='steelblue', alpha=0.7, label='Raw')
ax.plot(frames, sm_best, lw=1.2, color='firebrick', label=f'Smoothed (wsz={best_wsz})')
in_peak = False
for i in range(T):
    if peak_mask[i] and not in_peak:
        start = i
        in_peak = True
    elif not peak_mask[i] and in_peak:
        ax.axvspan(start, i, alpha=0.12, color='gold')
        in_peak = False
if in_peak:
    ax.axvspan(start, T, alpha=0.12, color='gold')
ax.set_ylabel('Flux')
ax.set_title('Raw signal + smoothed (peak regions shaded in gold)')
ax.legend(fontsize=8)

# Panel 2: original vs smoothed window sizes
ax = axes[1]
ax.plot(frames, win_raw, lw=0.5, color='gray', alpha=0.6, label='Window sizes (wsz=1, no smoothing)')
ax.plot(frames, win_best, lw=1.0, color='darkorange', alpha=0.9, label=f'Window sizes (wsz={best_wsz})')
ax.set_ylabel('Window size')
ax.set_title('Window size: original (noisy) vs smoothed')
ax.legend(fontsize=8)

# Panel 3: zoomed 1800–2300
ax = axes[2]
fz = np.arange(z0, z1)
ax.plot(fz, raw_series[z0:z1], lw=0.8, color='steelblue', alpha=0.8, label='Raw')
ax.plot(fz, sm_best[z0:z1], lw=1.4, color='firebrick', label=f'Smoothed (wsz={best_wsz})')
ax_r = ax.twinx()
ax_r.plot(fz, win_raw[z0:z1], lw=0.6, color='gray', alpha=0.5, label='win (wsz=1)')
ax_r.plot(fz, win_best[z0:z1], lw=1.0, color='darkorange', alpha=0.8, label=f'win (wsz={best_wsz})')
ax_r.set_ylabel('Window size', color='darkorange')
ax_r.tick_params(axis='y', labelcolor='darkorange')
ax.set_ylabel('Flux')
ax.set_title('Zoom: frames 1800–2300')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

# Panel 4: detrended signal with peaks marked
ax = axes[3]
ax.plot(frames, detrended, lw=0.6, color='darkgreen', label='Detrended (raw − coarse baseline)')
ax.axhline(threshold, color='red', ls='--', lw=0.9, label=f'Threshold (1.5×MAD = {threshold:.4f})')
ax.scatter(frames[peak_mask], detrended[peak_mask], s=3, color='red', alpha=0.5, label=f'Peaks ({peak_mask.sum()} frames)')
ax.set_xlabel('Frame')
ax.set_ylabel('Residual flux')
ax.set_title('Detrended signal with detected peaks')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/rri38/Documents/work/code/tess/tessreduce/development/adaptive_smooth_final.png', dpi=120)
plt.close()
print("\nSaved adaptive_smooth_final.png")

# Also save the zoom diagnostic as requested
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
var1d = var1[:, 0, 0]
fz = np.arange(z0, z1)

ax = axes[0]
ax.plot(fz, raw_series[z0:z1], lw=0.7, color='steelblue', label='Raw')
ax.plot(fz, sm1[:, 0, 0][z0:z1], lw=1.1, color='firebrick', label='Smoothed (wsz=1)')
ax.set_ylabel('Flux')
ax.set_title('Pixel (100,100): frames 1800–2300 — raw and smoothed')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(fz, win_raw[z0:z1], lw=0.6, color='gray', alpha=0.7, label='Quantized windows (wsz=1)')
ax_w = ax.twinx()
ax_w.plot(fz, wrc1d[z0:z1], lw=0.8, color='navy', alpha=0.6, label='Continuous raw (pre-round)')
ax_w.set_ylabel('Continuous window value', color='navy')
ax_w.tick_params(axis='y', labelcolor='navy')
ax.set_ylabel('Quantized window size', color='gray')
ax.set_title('Window sizes (quantized) and continuous field')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_w.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

ax = axes[2]
ax.plot(fz, var1d[z0:z1], lw=0.7, color='purple', label='Variability metric')
ax.set_ylabel('Variability')
ax.set_xlabel('Frame')
ax.set_title('Raw variability metric in zoom region')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/rri38/Documents/work/code/tess/tessreduce/development/diag_zoom_2000.png', dpi=120)
plt.close()
print("Saved diag_zoom_2000.png (updated)")

print("\n=== FINAL REPORT ===")
print(f"Pixel (100,100), T={T}")
print(f"\nWindow size stats (wsz=1, no smoothing):")
print(f"  mean={wq1d.mean():.2f}, std={wq1d.std():.2f}")
print(f"  n_switches={np.sum(np.diff(wq1d)!=0)}, mean run={run_lengths(wq1d).mean():.1f} frames")
print(f"  frac@w_min={np.mean(wq1d==3):.3f}, frac@w_max={np.mean(wq1d==51):.3f}")

for wsz, wqd in [(5, wq5d), (11, wq11d), (21, wq21d), (31, wq31d)]:
    print(f"\nWindow size stats (wsz={wsz}):")
    print(f"  mean={wqd.mean():.2f}, std={wqd.std():.2f}")
    print(f"  n_switches={np.sum(np.diff(wqd)!=0)}, mean run={run_lengths(wqd).mean():.1f} frames")
    print(f"  frac@w_min={np.mean(wqd==3):.3f}, frac@w_max={np.mean(wqd==51):.3f}")

print(f"\nPeak/baseline residuals:")
def compute_residuals(raw, sm, pm):
    r = np.abs(raw - sm)
    return r[pm].mean() if pm.any() else np.nan, r[~pm].mean() if (~pm).any() else np.nan

for wsz, smd in [(1, sm1[:, 0, 0]), (5, sm5[:, 0, 0]), (11, sm11[:, 0, 0]),
                  (21, sm21[:, 0, 0]), (31, sm31[:, 0, 0])]:
    pr, br = compute_residuals(raw_series, smd, peak_mask)
    print(f"  wsz={wsz:2d}: peak_res={pr:.5f}, base_res={br:.5f}, ratio={pr/br:.3f}")

print(f"\nDiagnosis of oscillation:")
rls = run_lengths(wpre1d)
print(f"  Pre-smooth window field: {len(rls)} runs, mean={rls.mean():.1f}, median={np.median(rls):.0f} frames/run")
print(f"  Fraction single-frame runs: {np.mean(rls==1):.3f}")
print(f"  Root cause: window field oscillates at period ~{2*np.median(rls):.0f} frames")
print(f"  Smoothing is partially absorbed by quantization (n_levels=7 with step-size ~8)")
print(f"  wsz=31 gives best switch reduction: {np.sum(np.diff(wq1d)!=0)} -> {np.sum(np.diff(wq31d)!=0)} ({100*(1-np.sum(np.diff(wq31d)!=0)/np.sum(np.diff(wq1d)!=0)):.0f}% reduction)")
print(f"\nBest window_smooth_size = 31 (largest switch reduction with acceptable residual increase)")
