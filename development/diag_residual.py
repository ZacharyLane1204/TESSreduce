import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, '/Users/rri38/Documents/work/code/tess/tessreduce/development')
from run_diag import adaptive_medfilt_3d, _get_segments, _make_odd

_base = "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors/FFI_Cadence/"
_dev = '/Users/rri38/Documents/work/code/tess/tessreduce/development/'

datasets = [
    dict(ds='test_bkg',  px=100, py=100, sector=38, cam=4, label='pixel (100,100)'),
    dict(ds='test_bkg2', px=20,  py=20,  sector=34, cam=1, label='pixel (20,20)'),
]


def with_breaks(t, y, segments):
    t_out, y_out = [], []
    for s, e in segments:
        t_out.extend(t[s:e].tolist())
        y_out.extend(y[s:e].tolist())
        t_out.append(np.nan)
        y_out.append(np.nan)
    return np.array(t_out), np.array(y_out)


for cfg in datasets:
    ds = cfg['ds']
    px, py = cfg['px'], cfg['py']
    label = cfg['label']

    data = np.load(f"{_dev}{ds}.npy")
    time = np.load(f"{_dev}{ds}_time.npy")
    cube = data[:, px:px+1, py:py+1]
    raw = cube[:, 0, 0]

    segs = _get_segments(time, gap_thresh=3.0)

    _df = pd.read_csv(f"{_base}TessVectors_S0{cfg['sector']:02d}_C{cfg['cam']}_FFI.csv", comment='#')
    earth_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Earth_Camera_Angle'].values)
    moon_angle = np.interp(time - 57000, _df['MidTime'].values, _df['Moon_Camera_Angle'].values)

    smoothed, windows, variability, _ = adaptive_medfilt_3d(
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
        earth_angle=earth_angle,
        moon_angle=moon_angle,
        scatter_angle_thresh=50.0,
    )

    sm = smoothed[:, 0, 0]
    win = windows[:, 0, 0]
    residual = raw - sm

    t_r, res_r = with_breaks(time, residual, segs)
    t_s, sm_r = with_breaks(time, sm, segs)
    t_w, win_r = with_breaks(time, win.astype(float), segs)
    t_raw, raw_r = with_breaks(time, raw, segs)

    mean_res = np.nanmean(np.abs(residual))
    print(f"{ds}: residual mean={mean_res:.5f}, std={np.nanstd(residual):.5f}, "
          f"max abs={np.nanmax(np.abs(residual)):.5f}, win_mean={win.mean():.1f}, "
          f"frac_wmax={np.mean(win==101):.3f}")

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Residual diagnostics — {label}  [w_max=51, scatter_angle_thresh=50]', fontsize=12)

    ax = axes[0]
    ax.plot(t_raw, raw_r, lw=0.6, color='steelblue', alpha=0.7, label='Raw')
    ax.plot(t_s, sm_r, lw=1.2, color='firebrick', label='Smoothed')
    ax.set_ylabel('Flux')
    ax.set_title('Raw + smoothed')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_r, res_r, lw=0.7, color='darkgreen', label='Residual (raw − smooth)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Residual')
    ax.set_title('Residual')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_r, np.abs(res_r), lw=0.7, color='purple', label='|Residual|')
    ax.set_ylabel('|Residual|')
    ax.set_title('Absolute residual')
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(t_w, win_r, lw=0.8, color='darkorange', label='Window size')
    ax.set_ylabel('Window size')
    ax.set_xlabel('Time (BTJD)')
    ax.set_title('Adaptive window size')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outfile = f"{_dev}residual_plot_{ds}.png"
    plt.savefig(outfile, dpi=120)
    plt.close()
    print(f"Saved {outfile}")
