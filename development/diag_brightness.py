import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('/Users/rri38/Documents/work/code/tess/tessreduce/development/test_bkg.npy')
time = np.load('/Users/rri38/Documents/work/code/tess/tessreduce/development/test_bkg_time.npy')

# Use central pixel for the time series panels (same as run_diag)
px, py = 100, 100
series = data[:, px, py]
T = len(series)

flat = data.ravel()
flat = flat[np.isfinite(flat)]

# --- Percentile thresholds at several candidate values ---
pcts = [50, 60, 70, 80, 90]
pct_thresholds = {p: np.nanpercentile(flat, p) for p in pcts}

# --- Sigma-clipped stats ---
clipped = flat.copy()
for _ in range(5):
    med = np.median(clipped)
    std = np.std(clipped)
    if std == 0:
        break
    clipped = clipped[np.abs(clipped - med) < 2.0 * std]
clip_mean = float(np.mean(clipped))
clip_std = float(np.std(clipped))
sigma_threshold = clip_mean + 2.0 * clip_std

print(f"Flat pixel count: {len(flat):,}")
print(f"Sigma-clip: mean={clip_mean:.4f}, std={clip_std:.4f}, threshold={sigma_threshold:.4f}")
for p, t in pct_thresholds.items():
    frac_above = np.mean(flat > t)
    print(f"  pct={p}: threshold={t:.4f}, frac_above={frac_above:.3f}")
frac_above_sigma = np.mean(flat > sigma_threshold)
print(f"  sigma-clip (mean+2σ): threshold={sigma_threshold:.4f}, frac_above={frac_above_sigma:.3f}")

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Brightness threshold diagnostics — pixel (100,100)', fontsize=12)

# Panel 1: histogram of all pixel values + threshold lines
ax = axes[0]
counts, bin_edges = np.histogram(flat, bins=300)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.semilogy(bin_centres, counts, color='steelblue', lw=0.8, label='All pixels')
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pcts)))
for (p, t), c in zip(pct_thresholds.items(), colors):
    ax.axvline(t, color=c, lw=1.0, ls='--', label=f'pct={p} ({t:.3f})')
ax.axvline(sigma_threshold, color='black', lw=1.5, ls='-', label=f'mean+2σ clipped ({sigma_threshold:.3f})')
ax.set_xlabel('Pixel value')
ax.set_ylabel('Count (log)')
ax.set_title('Global pixel value distribution with candidate thresholds')
ax.legend(fontsize=8, ncol=2)

# Panel 2: per-frame mean brightness + threshold lines
frame_mean = data.mean(axis=(1, 2))
ax = axes[1]
ax.plot(time, frame_mean, lw=0.7, color='steelblue', label='Per-frame mean brightness')
for (p, t), c in zip(pct_thresholds.items(), colors):
    ax.axhline(t, color=c, lw=0.8, ls='--')
ax.axhline(sigma_threshold, color='black', lw=1.5, ls='-', label='mean+2σ clipped')
ax.set_xlabel('Time (BTJD)')
ax.set_ylabel('Mean flux')
ax.set_title('Per-frame mean brightness vs time')
ax.legend(fontsize=8)

# Panel 3: bright mask fraction over time for each threshold
ax = axes[2]
for (p, t), c in zip(pct_thresholds.items(), colors):
    mask_frac = (data > t).mean(axis=(1, 2))
    ax.plot(time, mask_frac, color=c, lw=0.8, label=f'pct={p}')
mask_frac_sigma = (data > sigma_threshold).mean(axis=(1, 2))
ax.plot(time, mask_frac_sigma, color='black', lw=1.5, label='mean+2σ clipped')
ax.set_xlabel('Time (BTJD)')
ax.set_ylabel('Fraction of pixels above threshold')
ax.set_title('Bright-mask active fraction over time (per frame)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/rri38/Documents/work/code/tess/tessreduce/development/diag_brightness.png', dpi=120)
plt.close()
print("Saved diag_brightness.png")
