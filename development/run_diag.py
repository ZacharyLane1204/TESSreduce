import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, percentile_filter, uniform_filter
from joblib import Parallel, delayed


def _make_odd(x):
    x = max(int(x), 3)
    return x if x % 2 == 1 else x + 1


def _get_segments(time, gap_thresh):
    """Return list of (start, end) frame index pairs for contiguous segments."""
    dt = np.diff(time)
    median_dt = np.median(dt)
    breaks = np.where(dt > gap_thresh * median_dt)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(time)]])
    return list(zip(starts.tolist(), ends.tolist()))


def adaptive_medfilt_3d(
    data,
    time=None,
    gap_thresh=3.0,
    w_min=3,
    w_max=51,
    grad_smooth_window=11,
    low_pct=5,
    high_pct=80,
    per_pixel_norm=True,
    n_levels=7,
    n_jobs=1,
    metric='deviation',
    coarse_windows=(11, 21, 51, 101),
    combined_weight=0.5,
    local_std_window=21,
    local_norm_window=201,
    brightness_sigma=2.0,
    window_smooth_size=1,
    earth_angle=None,
    moon_angle=None,
    scatter_angle_thresh=50.0,
):
    data = np.asarray(data, dtype=float)
    T, X, Y = data.shape

    if time is None:
        time = np.arange(T, dtype=float)
    time = np.asarray(time, dtype=float)

    segments = _get_segments(time, gap_thresh)

    # NaN fill per segment using time coordinates (never interpolates across gaps)
    nan_mask = ~np.isfinite(data)
    data_filled = data.copy()
    if nan_mask.any():
        flat = data_filled.reshape(T, -1)
        for s, e in segments:
            t_seg = time[s:e]
            for j in range(flat.shape[1]):
                ts = flat[s:e, j]
                bad = ~np.isfinite(ts)
                if bad.all():
                    flat[s:e, j] = 0.0
                elif bad.any():
                    flat[s:e, j] = np.interp(t_seg, t_seg[~bad], ts[~bad])

    gw = _make_odd(grad_smooth_window)

    # Gradient uses actual time spacing; zeroed at segment boundaries
    if metric in ('gradient', 'combined'):
        grad = np.gradient(data_filled, time, axis=0)
        for s, e in segments:
            if s > 0:
                grad[s] = 0.0
            if e < T:
                grad[e - 1] = 0.0
        grad_var = np.zeros_like(grad)
        for s, e in segments:
            seg = grad[s:e]
            if e - s >= gw:
                grad_var[s:e] = np.abs(median_filter(seg, size=(gw, 1, 1), mode='reflect'))
            else:
                grad_var[s:e] = np.abs(seg)

    # Multi-scale deviation: compute deviation at each coarse window, normalise
    # each locally, then take the max — catches variability at any timescale
    if metric in ('deviation', 'combined'):
        scales = [coarse_windows] if isinstance(coarse_windows, int) else list(coarse_windows)
        # Compute all scale deviations first so we can set a common floor
        all_devs = []
        for cw in scales:
            dev = np.zeros_like(data_filled)
            for s, e in segments:
                seg = data_filled[s:e]
                n = e - s
                w = _make_odd(min(cw, n if n % 2 == 1 else n - 1))
                coarse_seg = median_filter(seg, size=(w, 1, 1), mode='reflect')
                diff = np.abs(seg - coarse_seg)
                dw = _make_odd(min(gw, n if n % 2 == 1 else n - 1))
                dev[s:e] = median_filter(diff, size=(dw, 1, 1), mode='reflect')
            all_devs.append(dev)
        # Identify stable segments by coefficient of variation of per-frame means.
        # A stable segment has low CV (frame means tightly clustered around their
        # sigma-clipped mean). For stable segments, skip local normalization and
        # force dev_var=0 → w_max. For active segments use the local normalization
        # as normal, with cw_min masked in dark frames via bright_mask.
        _frame_mean = data_filled.mean(axis=(1, 2))  # (T,)
        # Global sigma-clip for bright_mask (used in active segments only)
        _clipped = _frame_mean[np.isfinite(_frame_mean)]
        for _ in range(5):
            _med = np.median(_clipped)
            _std = np.std(_clipped)
            if _std == 0:
                break
            _clipped = _clipped[np.abs(_clipped - _med) < brightness_sigma * _std]
        brightness_threshold = float(np.mean(_clipped)) + brightness_sigma * float(np.std(_clipped))
        bright_mask = (_frame_mean > brightness_threshold).astype(float)[:, np.newaxis, np.newaxis]

        # Per-segment stability: a segment is stable (force w_max) when it has
        # no scattered-light contamination. If earth/moon angle arrays are provided,
        # use min(earth, moon) > scatter_angle_thresh as the scatter-free criterion.
        # Otherwise fall back to brightness threshold (< 5% active frames).
        _frame_active = (_frame_mean > brightness_threshold)
        # When angle arrays are provided and scattered light is actually present
        # (some frames below scatter_angle_thresh), build a stable_mask to force
        # w_max on scatter-free faint frames across all scales.
        # Without angles, or when no scatter is present, only mask cw_min (i==0).
        _use_stable_mask = False
        if earth_angle is not None or moon_angle is not None:
            _ea = np.asarray(earth_angle, dtype=float) if earth_angle is not None else np.full(T, np.inf)
            _ma = np.asarray(moon_angle, dtype=float) if moon_angle is not None else np.full(T, np.inf)
            _scatter_free = np.minimum(_ea, _ma) > scatter_angle_thresh
            _has_scatter = (~_scatter_free).any()
            if _has_scatter:
                _stable_frame = _scatter_free & ~_frame_active
                _stable_mask = _stable_frame[:, np.newaxis, np.newaxis].astype(float)
                _use_stable_mask = True

        lnw = _make_odd(local_norm_window)
        scale_norms = []
        for i, dev in enumerate(all_devs):
            scale_floor = max(np.nanpercentile(dev, 75), 1e-10)
            g_lo = np.zeros_like(dev)
            g_hi = np.zeros_like(dev)
            for s, e in segments:
                seg_v = dev[s:e]
                n = e - s
                lw = _make_odd(min(lnw, n if n % 2 == 1 else n - 1))
                g_lo[s:e] = percentile_filter(seg_v, low_pct, size=(lw, 1, 1), mode='reflect')
                g_hi[s:e] = percentile_filter(seg_v, high_pct, size=(lw, 1, 1), mode='reflect')
            dg = np.maximum(g_hi - g_lo, scale_floor)
            norm_scale = np.clip((dev - g_lo) / dg, 0.0, 1.0)
            if i == 0:
                norm_scale = norm_scale * bright_mask
            if _use_stable_mask:
                norm_scale = norm_scale * (1.0 - _stable_mask)
            scale_norms.append(norm_scale)
        dev_var = np.max(np.stack(scale_norms), axis=0)  # (T, X, Y)

    # Local std: short-window RMS scatter — captures frame-to-frame variability
    # regardless of slow trends or broad peaks
    if metric in ('local_std', 'combined'):
        lsw = _make_odd(local_std_window)
        lstd_var = np.zeros_like(data_filled)
        for s, e in segments:
            seg = data_filled[s:e]
            n = e - s
            w = min(lsw, n if n % 2 == 1 else n - 1)
            w = _make_odd(w)
            m = uniform_filter(seg, size=(w, 1, 1), mode='reflect')
            m2 = uniform_filter(seg ** 2, size=(w, 1, 1), mode='reflect')
            lstd_var[s:e] = np.sqrt(np.maximum(m2 - m ** 2, 0))

    def _norm01(x):
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-30), 0, 1)

    # dev_var is already locally normalised to [0,1] per scale, max-combined
    # other metrics still need normalisation
    if metric == 'gradient':
        variability = grad_var
    elif metric == 'local_std':
        variability = lstd_var
    elif metric == 'combined':
        variability = (1 - combined_weight) * _norm01(grad_var) + combined_weight * _norm01(lstd_var)
    elif metric == 'deviation':
        variability = dev_var
    else:
        raise ValueError(f"metric must be 'gradient', 'deviation', 'local_std', or 'combined', got '{metric}'")

    if metric == 'deviation':
        norm = dev_var  # already in [0,1], normalised per scale
    elif local_norm_window is not None:
        lnw = _make_odd(local_norm_window)
        g_lo = np.zeros_like(variability)
        g_hi = np.zeros_like(variability)
        for s, e in segments:
            seg_var = variability[s:e]
            n = e - s
            w = _make_odd(min(lnw, n if n % 2 == 1 else n - 1))
            g_lo[s:e] = percentile_filter(seg_var, low_pct, size=(w, 1, 1))
            g_hi[s:e] = percentile_filter(seg_var, high_pct, size=(w, 1, 1))
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)
    elif per_pixel_norm:
        g_lo = np.nanpercentile(variability, low_pct, axis=0, keepdims=True)
        g_hi = np.nanpercentile(variability, high_pct, axis=0, keepdims=True)
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)
    else:
        g_lo = np.nanpercentile(variability, low_pct)
        g_hi = np.nanpercentile(variability, high_pct)
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)

    raw_w = w_max - norm * (w_max - w_min)
    windows = np.round(raw_w).astype(int)
    windows += (1 - windows % 2)
    windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))

    windows_pre_smooth = windows.copy()

    if window_smooth_size > 1:
        wsz = _make_odd(window_smooth_size)
        smoothed_win = np.empty_like(windows, dtype=float)
        for s, e in segments:
            n = e - s
            w = min(wsz, n if n % 2 == 1 else n - 1)
            w = _make_odd(w)
            smoothed_win[s:e] = median_filter(windows[s:e].astype(float), size=(w, 1, 1), mode='reflect')
        windows = np.round(smoothed_win).astype(int)
        windows += (1 - windows % 2)
        windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))

    levels = np.unique([_make_odd(int(round(w))) for w in np.linspace(w_min, w_max, n_levels)])
    windows_quantized = levels[np.argmin(np.abs(windows[..., np.newaxis] - levels), axis=-1)]

    # Final smoothing per segment to avoid blending across gaps
    result = np.empty((T, X, Y))
    for s, e in segments:
        seg_data = data_filled[s:e]
        seg_wins = windows_quantized[s:e]
        seg_levels = np.unique(seg_wins)

        def _smooth_seg(w, seg=seg_data):
            return w, median_filter(seg, size=(w, 1, 1), mode='reflect')

        for w, smoothed_w in Parallel(n_jobs=n_jobs)(delayed(_smooth_seg)(w) for w in seg_levels):
            result[s:e][seg_wins == w] = smoothed_w[seg_wins == w]

    result[nan_mask] = np.nan
    return result, windows, variability, windows_pre_smooth


# ── Load data ──────────────────────────────────────────────────────────────────
data = np.load('/Users/rri38/Documents/work/code/tess/tessreduce/development/test_bkg.npy')
time = np.load('/Users/rri38/Documents/work/code/tess/tessreduce/development/test_bkg_time.npy')
print(f"Data shape: {data.shape}")
print(f"Time shape: {time.shape}, range: {time[0]:.4f} – {time[-1]:.4f}")
dt = np.diff(time)
print(f"Median dt: {np.median(dt):.6f}, max dt: {dt.max():.6f}, gap ratio: {dt.max()/np.median(dt):.1f}x")
segs = _get_segments(time, gap_thresh=3.0)
print(f"Segments detected: {len(segs)} — {segs}")

px, py = 100, 100
cube = data[:, px:px+1, py:py+1]
print(f"Pixel cube shape: {cube.shape}")
print(f"Pixel value range: {np.nanmin(cube):.4f} – {np.nanmax(cube):.4f}")

raw_series = cube[:, 0, 0]
T = len(raw_series)
print(f"T = {T}")

# ── Step 1: Run with winning params (no window smoothing) ──────────────────────
print("\n=== Step 1: Diagnostic run (window_smooth_size=1) ===")
smoothed, windows, variability, windows_pre = adaptive_medfilt_3d(
    cube,
    time=time,
    metric='deviation',
    coarse_windows=(11, 21, 51, 101),
    low_pct=5,
    high_pct=80,
    local_norm_window=201,
    w_min=3,
    w_max=51,
    n_jobs=1,
    window_smooth_size=1,
)

sm = smoothed[:, 0, 0]
win = windows[:, 0, 0]
var = variability[:, 0, 0]

# Stats
print(f"Mean window size:  {win.mean():.2f}")
print(f"Std window size:   {win.std():.2f}")
frac_min = np.mean(win == _make_odd(3))
frac_max = np.mean(win == _make_odd(51))
print(f"Fraction at w_min (3):  {frac_min:.4f}  ({frac_min*100:.1f}%)")
print(f"Fraction at w_max (51): {frac_max:.4f}  ({frac_max*100:.1f}%)")
print(f"Unique window values: {np.unique(win)}")

# ── Plot 1: Zoomed view 1800–2300 ──────────────────────────────────────────────
z0, z1 = 1800, min(2300, T)
frames_z = np.arange(z0, z1)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax1.plot(frames_z, raw_series[z0:z1], lw=0.8, color='steelblue', label='Raw')
ax1.plot(frames_z, sm[z0:z1], lw=1.2, color='firebrick', label='Smoothed')
ax1.set_ylabel('Flux')
ax1.set_title('Pixel (100,100) — zoomed frames 1800–2300')
ax1.legend(fontsize=8)

ax_win = ax2
ax_win.plot(frames_z, win[z0:z1], color='darkorange', lw=0.7, label='Window size')
ax_win.set_ylabel('Window size')
ax_win.set_title('Adaptive window sizes (no smoothing)')
ax_win.legend(fontsize=8)

ax3.plot(frames_z, var[z0:z1], color='purple', lw=0.7, label='Variability metric')
ax3.set_ylabel('Variability')
ax3.set_xlabel('Frame')
ax3.set_title('Raw variability metric (deviation-based)')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/rri38/Documents/work/code/tess/tessreduce/development/diag_zoom_2000.png', dpi=120)
plt.close()
print("Saved diag_zoom_2000.png")

# Check variability smoothness in the zoom region
var_z = var[z0:z1]
var_diffs = np.diff(var_z)
print(f"\nVariability in 1800–2300:")
print(f"  mean={var_z.mean():.4f}, std={var_z.std():.4f}")
print(f"  mean |diff| (roughness): {np.abs(var_diffs).mean():.6f}")
print(f"  max |diff|: {np.abs(var_diffs).max():.6f}")

# ── Step 2: Test window_smooth_sizes ──────────────────────────────────────────
print("\n=== Step 2: Testing window_smooth_size values ===")

# Peak/baseline detection helpers
def detect_peaks_adaptive(series, coarse_bw=201, prom_thresh_mad_mult=1.5):
    """Return boolean mask of 'peak' frames."""
    cw = _make_odd(coarse_bw)
    baseline = median_filter(series, size=cw, mode='reflect')
    detrended = series - baseline
    mad = np.median(np.abs(detrended - np.median(detrended)))
    threshold = prom_thresh_mad_mult * mad
    return detrended > threshold, detrended, baseline, threshold

def compute_residuals(raw, smoothed, peak_mask):
    residuals = np.abs(raw - smoothed)
    peak_res = residuals[peak_mask].mean() if peak_mask.any() else np.nan
    base_res = residuals[~peak_mask].mean() if (~peak_mask).any() else np.nan
    return peak_res, base_res

smooth_sizes = [1, 5, 11, 21, 31]
results = {}

for wsz in smooth_sizes:
    sm_w, win_w, _, _ = adaptive_medfilt_3d(
        cube,
        time=time,
        metric='deviation',
        coarse_windows=(11, 21, 51, 101),
        low_pct=5,
        high_pct=80,
        local_norm_window=201,
        w_min=3,
        w_max=51,
        n_jobs=1,
        window_smooth_size=wsz,
    )
    sm_1d = sm_w[:, 0, 0]
    win_1d = win_w[:, 0, 0]

    peak_mask, detrended, baseline, threshold = detect_peaks_adaptive(raw_series)
    peak_res, base_res = compute_residuals(raw_series, sm_1d, peak_mask)

    n_switches = np.sum(np.diff(win_1d) != 0)
    win_std = win_1d.std()
    win_mean = win_1d.mean()
    fmin = np.mean(win_1d == _make_odd(3))
    fmax = np.mean(win_1d == _make_odd(51))

    results[wsz] = {
        'smoothed': sm_1d,
        'windows': win_1d,
        'peak_res': peak_res,
        'base_res': base_res,
        'win_mean': win_mean,
        'win_std': win_std,
        'n_switches': n_switches,
        'fmin': fmin,
        'fmax': fmax,
        'peak_mask': peak_mask,
        'detrended': detrended,
        'threshold': threshold,
    }

    print(f"\nwindow_smooth_size={wsz}:")
    print(f"  win mean={win_mean:.2f}, std={win_std:.2f}, n_switches={n_switches}")
    print(f"  frac@w_min={fmin:.3f}, frac@w_max={fmax:.3f}")
    print(f"  peak_residual={peak_res:.5f}, base_residual={base_res:.5f}")
    ratio = peak_res / base_res if base_res > 0 else np.nan
    print(f"  peak/base ratio={ratio:.3f}")

# Determine best: want low base residual, retain peak residual, fewer switches
# Score: lower is better — base_res (main), penalise very high n_switches
# We want ratio reasonably high (peaks preserved) and base_res low
print("\n=== Score summary ===")
for wsz in smooth_sizes:
    r = results[wsz]
    print(f"  wsz={wsz:2d}: base_res={r['base_res']:.5f}, peak_res={r['peak_res']:.5f}, "
          f"ratio={r['peak_res']/r['base_res']:.3f}, n_switches={r['n_switches']}")

# Choose best: wsz that gives fewest switches with stable base/peak residuals
# Use elbow: pick smallest wsz where n_switches drops by >50% from wsz=1
base_switches = results[1]['n_switches']
best_wsz = 1
for wsz in smooth_sizes[1:]:
    if results[wsz]['n_switches'] < 0.4 * base_switches:
        best_wsz = wsz
        break

print(f"\nSelected best window_smooth_size = {best_wsz}")

# ── Step 3: Final 4-panel diagnostic ──────────────────────────────────────────
print(f"\n=== Step 3: Saving adaptive_smooth_final.png with wsz={best_wsz} ===")

r = results[best_wsz]
r_raw = results[1]
sm_best = r['smoothed']
win_best = r['windows']
win_raw = r_raw['windows']
peak_mask = r['peak_mask']
detrended = r['detrended']

frames = np.arange(T)

fig, axes = plt.subplots(4, 1, figsize=(16, 16))
fig.suptitle(f'Pixel (100,100) — adaptive_medfilt_3d  |  window_smooth_size={best_wsz}', fontsize=12)

# Panel 1: raw + smoothed + peak shading
ax = axes[0]
ax.plot(frames, raw_series, lw=0.6, color='steelblue', alpha=0.7, label='Raw')
ax.plot(frames, sm_best, lw=1.2, color='firebrick', label=f'Smoothed (wsz={best_wsz})')
# Shade peak regions
in_peak = False
for i in range(T):
    if peak_mask[i] and not in_peak:
        start = i
        in_peak = True
    elif not peak_mask[i] and in_peak:
        ax.axvspan(start, i, alpha=0.15, color='gold')
        in_peak = False
if in_peak:
    ax.axvspan(start, T, alpha=0.15, color='gold')
ax.set_ylabel('Flux')
ax.set_title('Raw signal + smoothed (peak regions shaded)')
ax.legend(fontsize=8)

# Panel 2: original vs smoothed window sizes
ax = axes[1]
ax.plot(frames, win_raw, lw=0.5, color='gray', alpha=0.7, label='Original windows (wsz=1)')
ax.plot(frames, win_best, lw=1.0, color='darkorange', label=f'Smoothed windows (wsz={best_wsz})')
ax.set_ylabel('Window size')
ax.set_title('Window size: original vs smoothed')
ax.legend(fontsize=8)

# Panel 3: zoomed 1800–2300
ax = axes[2]
z0, z1 = 1800, min(2300, T)
fz = np.arange(z0, z1)
ax.plot(fz, raw_series[z0:z1], lw=0.8, color='steelblue', alpha=0.8, label='Raw')
ax.plot(fz, sm_best[z0:z1], lw=1.3, color='firebrick', label=f'Smoothed (wsz={best_wsz})')
ax_r = ax.twinx()
ax_r.plot(fz, win_best[z0:z1], color='darkorange', lw=0.8, alpha=0.6, label='Window')
ax_r.set_ylabel('Window size', color='darkorange')
ax_r.tick_params(axis='y', labelcolor='darkorange')
ax.set_ylabel('Flux')
ax.set_title('Zoom: frames 1800–2300')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Panel 4: detrended signal with peaks marked
ax = axes[3]
ax.plot(frames, detrended, lw=0.7, color='darkgreen', label='Detrended (raw − coarse baseline)')
ax.axhline(r['threshold'], color='red', ls='--', lw=0.8, label=f'Threshold (1.5×MAD={r["threshold"]:.4f})')
ax.scatter(frames[peak_mask], detrended[peak_mask], s=4, color='red', alpha=0.5, label='Peaks')
ax.set_xlabel('Frame')
ax.set_ylabel('Residual flux')
ax.set_title('Detrended signal with detected peaks')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/rri38/Documents/work/code/tess/tessreduce/development/adaptive_smooth_final.png', dpi=120)
plt.close()
print("Saved adaptive_smooth_final.png")
print("\nDone.")
