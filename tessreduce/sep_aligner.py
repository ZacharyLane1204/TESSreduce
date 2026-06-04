"""
sep_aligner.py
==============
A tessreduce-compatible image alignment class using SEP source extraction
and 3×3 PSF-core stamp minimisation.

Usage inside tessreduce
-----------------------
Drop this file next to tessreduce.py (or anywhere on sys.path) and call:

    from sep_aligner import SepAligner

    # Standalone:
    aligner = SepAligner(tr.ref, tr.flux, n_jobs=-1)
    aligner.run()
    tr.shift = aligner.shift          # hand shifts back to tessreduce
    tr.shift_images()                 # tessreduce applies them

    # Or plug into the shift_method dispatch in tessreduce.reduce():
    #   elif self._shift_method == 'sep_core':
    #       SepAligner.from_tessreduce(self).run()

Interface contract with tessreduce
-----------------------------------
tessreduce.shift_images() expects self.shift to be (T, 2) float32 where
    shift[t] = [row_shift, col_shift]  ≡  [dy, dx]
and applies   scipy.ndimage.shift(frame, [dy, dx])
i.e. a POSITIVE dy moves the image content DOWN (row += dy).

SepAligner measures dx, dy as (science − reference) pixel displacement and
stores them with the same sign convention so tessreduce.shift_images() will
call shift(frame, [-dy, -dx]) ... actually tessreduce calls
shift(frame, [+dy, +dx]) so we store the NEGATIVE of our measured offset.
See _build_shift_array() for details.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import sep
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift as nd_shift
from scipy.optimize import minimize
from scipy.spatial import cKDTree


fig_width = 240.0 / 72.27  # matches tessreduce figure sizing convention


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (module-level so joblib can pickle them)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_params(shape: tuple[int, int]) -> dict:
    """Scale stamp / extraction params to image size."""
    H, W = shape
    minsize = min(H, W)
    stamp_half = max(3, min(7, minsize // 6))
    core_half = 2 if stamp_half >= 3 else 0
    minarea = 3 if minsize < 80 else 5
    match_radius = max(2.0, min(3.0, minsize * 0.05))
    coarse_range = max(0.3, min(1.0, minsize * 0.03))
    return dict(stamp_half=stamp_half, core_half=core_half,
                minarea=minarea, match_radius=match_radius,
                coarse_range=coarse_range)


def _extract(arr: np.ndarray, thresh: float = 3.0, minarea: int = 5):
    """Background-subtract and extract sources. Returns (sub, src)."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    bkg = sep.Background(a)
    sub = a - bkg
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        src = sep.extract(sub, thresh, err=bkg.globalrms, minarea=minarea)
    return sub, src


def _select_stars(src_ref, src_img, sub_ref, sub_img,
                  sat_frac: float = 0.7,
                  ell_max: float = 0.5,
                  var_thresh: float = 0.05,
                  match_radius: float = 3.0,
                  flag_max: int = 0,
                  edge_margin: int = 3,
                  pixel_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ri, ii) index arrays into src_ref / src_img for matched pairs that
    are round, unflagged, unsaturated, flux-stable, and not within
    edge_margin pixels of any border.  Sources whose centre pixel has
    pixel_mask True are also excluded.
    """
    empty = np.array([], dtype=int)
    if len(src_ref) == 0 or len(src_img) == 0:
        return empty, empty

    H, W = sub_ref.shape
    sat_level = sat_frac * max(float(sub_ref.max()), float(sub_img.max()))
    m = edge_margin

    ref_ok = ((src_ref['x'] >= m) & (src_ref['x'] <= W - 1 - m) &
              (src_ref['y'] >= m) & (src_ref['y'] <= H - 1 - m))
    img_ok = ((src_img['x'] >= m) & (src_img['x'] <= W - 1 - m) &
              (src_img['y'] >= m) & (src_img['y'] <= H - 1 - m))

    if pixel_mask is not None:
        rx = np.clip(np.round(src_ref['x']).astype(int), 0, W - 1)
        ry = np.clip(np.round(src_ref['y']).astype(int), 0, H - 1)
        ix = np.clip(np.round(src_img['x']).astype(int), 0, W - 1)
        iy = np.clip(np.round(src_img['y']).astype(int), 0, H - 1)
        ref_ok &= ~pixel_mask[ry, rx]
        img_ok &= ~pixel_mask[iy, ix]

    src_r = src_ref[ref_ok];  ri_orig = np.where(ref_ok)[0]
    src_i = src_img[img_ok];  ii_orig = np.where(img_ok)[0]

    if len(src_r) == 0 or len(src_i) == 0:
        return empty, empty

    tree = cKDTree(np.c_[src_i['x'], src_i['y']])
    dist, idx = tree.query(np.c_[src_r['x'], src_r['y']],
                           k=1, distance_upper_bound=match_radius)
    valid = dist < match_radius
    ri_in = np.where(valid)[0]
    ii_in = idx[valid]

    if len(ri_in) == 0:
        return empty, empty

    flux_ratio = (src_i['flux'][ii_in] /
                  np.clip(src_r['flux'][ri_in], 1.0, None))
    med_ratio = float(np.median(flux_ratio))

    good = (
        (1.0 - src_r['b'][ri_in] / src_r['a'][ri_in] < ell_max) &
        (src_r['flag'][ri_in] == 0) &
        (src_r['peak'][ri_in] < sat_level) &
        (src_i['peak'][ii_in] < sat_level) &
        (np.abs(flux_ratio - med_ratio) < var_thresh * med_ratio)
    )
    return ri_orig[ri_in[good]], ii_orig[ii_in[good]]


def _build_ref_cores(sub_ref: np.ndarray, src_ref, ri: np.ndarray,
                     stamp_half: int, core_half: int, bkg_rms: float):
    """
    Pre-extract the (2*core_half+1)² PSF core of each selected reference star.
    Returns (cores, positions, snr_weights) or (None, None, None) on failure.
    positions: list of (x0_stamp, y0_stamp, actual_stamp_half).
    """
    H, W = sub_ref.shape
    cores, positions, weights = [], [], []

    for r in ri:
        xc = int(round(float(src_ref['x'][r])))
        yc = int(round(float(src_ref['y'][r])))
        sh = stamp_half
        while sh >= core_half + 1:
            x0 = xc - sh; x1 = xc + sh
            y0 = yc - sh; y1 = yc + sh
            if x0 >= 0 and y0 >= 0 and x1 < W and y1 < H:
                break
            sh -= 1
        else:
            continue
        c0 = sh - core_half;  c1 = sh + core_half + 1
        core = sub_ref[y0 + c0: y0 + c1, x0 + c0: x0 + c1].copy()
        if core.shape != (c1 - c0, c1 - c0):
            continue
        snr = float(src_ref['flux'][r]) / (float(src_ref['npix'][r]) *
                                           max(bkg_rms, 1e-6))
        cores.append(core)
        positions.append((x0, y0, sh))
        weights.append(max(snr, 0.0) ** 2)

    if not cores:
        return None, None, None
    return np.array(cores), positions, np.array(weights)


def _loss_spline(spline, rows, cols, cores_ref, positions, weights,
                 dx, dy, core_half) -> float:
    shifted = spline(rows - dy, cols - dx)
    total = w_sum = 0.0
    for k, (x0, y0, sh) in enumerate(positions):
        c0 = sh - core_half;  c1 = sh + core_half + 1
        core = shifted[y0 + c0: y0 + c1, x0 + c0: x0 + c1]
        if core.shape != cores_ref[k].shape:
            continue
        diff = core - cores_ref[k]
        if not np.all(np.isfinite(diff)):
            continue
        w = weights[k]
        total += w * float(np.mean(diff ** 2))
        w_sum += w
    return total / max(w_sum, 1e-30)


def _loss_spline_grid(spline, cores_ref, positions, weights,
                      dy_grid, dx_grid, core_half) -> np.ndarray:
    """
    Evaluate the stamp-MSE loss over a full (dy, dx) grid in a single
    spline.ev() call instead of one full-image spline evaluation per point.

    Returns a (G, G) float64 array of loss values indexed [i_dy, j_dx].
    """
    G = len(dy_grid)

    # Collect stamp pixel coords and flattened ref values for every star
    stamp_rows, stamp_cols, ref_list = [], [], []
    for k, (x0, y0, sh) in enumerate(positions):
        c0 = sh - core_half;  c1 = sh + core_half + 1
        rs = np.arange(y0 + c0, y0 + c1)
        cs = np.arange(x0 + c0, x0 + c1)
        rr, cc = np.meshgrid(rs, cs, indexing='ij')
        stamp_rows.append(rr.ravel())
        stamp_cols.append(cc.ravel())
        ref_list.append(cores_ref[k].ravel())

    all_rows = np.concatenate(stamp_rows)    # (P,)
    all_cols = np.concatenate(stamp_cols)    # (P,)
    ref_flat = np.concatenate(ref_list)      # (P,)
    star_sizes = [len(r) for r in stamp_rows]

    # Build shifted pixel coords for every grid point — shape (G*G, P)
    DY, DX = np.meshgrid(dy_grid, dx_grid, indexing='ij')
    dy_flat = DY.ravel()[:, None]   # (G*G, 1)
    dx_flat = DX.ravel()[:, None]

    rows_eval = (all_rows[None, :] - dy_flat).ravel()   # (G*G*P,)
    cols_eval = (all_cols[None, :] - dx_flat).ravel()

    # Single scattered-point spline evaluation for the entire grid
    sci = spline.ev(rows_eval, cols_eval).reshape(G * G, -1)  # (G*G, P)

    diff2 = (sci - ref_flat[None, :]) ** 2   # (G*G, P)

    # Accumulate weighted MSE per grid point, star by star
    total = np.zeros(G * G)
    w_sum = np.zeros(G * G)
    offset = 0
    for k, sz in enumerate(star_sizes):
        sl = slice(offset, offset + sz)
        d2 = diff2[:, sl]                              # (G*G, sz)
        fin = np.all(np.isfinite(d2), axis=1)           # (G*G,)
        total += np.where(fin, weights[k] * d2.mean(axis=1), 0.0)
        w_sum += np.where(fin, weights[k], 0.0)
        offset += sz

    loss = np.where(w_sum > 0, total / np.maximum(w_sum, 1e-30), np.nan)
    return loss.reshape(G, G)


def _build_composite(img: np.ndarray,
                     cores_ref: np.ndarray,
                     positions: list,
                     weights: np.ndarray,
                     core_half: int):
    """
    Extract PSF-core stamps from *img* and lay them side by side into a single
    composite image of shape (c, N*c) where c = 2*core_half+1.

    The matching reference composite is built the same way from *cores_ref*.

    Returns
    -------
    sci_comp : (c, N*c) float64  — stamps from the science frame
    ref_comp : (c, N*c) float64  — stamps from the reference cores
    w_cols   : (N*c,)   float64  — per-column SNR² weights (same value per stamp)
    """
    c = 2 * core_half + 1
    N = len(positions)
    sci_comp = np.empty((c, N * c), dtype=np.float64)
    ref_comp = np.empty((c, N * c), dtype=np.float64)
    w_cols = np.empty(N * c, dtype=np.float64)

    for k, (x0, y0, sh) in enumerate(positions):
        c0 = sh - core_half
        c1 = sh + core_half + 1
        sci_stamp = img[y0 + c0: y0 + c1, x0 + c0: x0 + c1]
        col_start = k * c
        col_end = col_start + c
        sci_comp[:, col_start:col_end] = sci_stamp
        ref_comp[:, col_start:col_end] = cores_ref[k]
        w_cols[col_start:col_end] = weights[k]

    return sci_comp, ref_comp, w_cols


def _loss_composite(sci_comp: np.ndarray,
                    ref_comp: np.ndarray,
                    w_cols: np.ndarray,
                    dx: float, dy: float) -> float:
    """
    Weighted MSE loss for a (dx, dy) shift evaluated on the composite image,
    considering only the central 3×3 pixels of each source stamp.

    *sci_comp* is shifted by (-dy, -dx) — the same rigid shift applied to all
    stamps simultaneously — then compared to *ref_comp* at the inner pixels.

    Parameters
    ----------
    sci_comp   : (c, N*c)  science composite from _build_composite
    ref_comp   : (c, N*c)  reference composite from _build_composite
    w_cols     : (N*c,)    per-column SNR² weights
    valid_mask : (N*c,)    columns that were finite at zero shift
    inner_mask : (c, N*c)  central 3×3 pixel mask from _build_composite
    dx, dy     : float     sub-pixel shift to test
    """
    shifted = nd_shift(sci_comp, (-dy, -dx), order=5, mode='nearest')
    diff2 = (shifted - ref_comp) ** 2
    c = sci_comp.shape[0]
    N = w_cols.shape[0] // c
    ch = (c - 1) // 2  # core_half
    r0, r1 = ch - 1, ch + 2   # inner row slice [1:4]
    total = w_sum = 0.0
    for k in range(N):
        cs = k * c + ch - 1    # inner col start
        block = diff2[r0:r1, cs:cs + 3]   # (3, 3)
        if not np.all(np.isfinite(block)):
            continue
        w = w_cols[k * c]
        total += w * block.sum()
        w_sum += w
    return total / max(w_sum, 1e-30)


def _loss_ndshift(sub_img, cores_ref, positions, weights,
                  dx, dy, core_half) -> float:
    shifted = nd_shift(sub_img, (-dy, -dx), order=5, mode='nearest')
    total = w_sum = 0.0
    for k, (x0, y0, sh) in enumerate(positions):
        c0 = sh - core_half;  c1 = sh + core_half + 1
        core = shifted[y0 + c0: y0 + c1, x0 + c0: x0 + c1]
        if core.shape != cores_ref[k].shape:
            continue
        diff = core - cores_ref[k]
        if not np.all(np.isfinite(diff)):
            continue
        w = weights[k]
        total += w * float(np.mean(diff ** 2))
        w_sum += w
    return total / max(w_sum, 1e-30)


def _per_stamp_uncertainty(sub_img, sub_ref, src_ref, ri,
                           dx_opt, dy_opt, stamp_half, core_half):
    H, W = sub_ref.shape
    shifted = nd_shift(sub_img, (-dy_opt, -dx_opt), order=5, mode='nearest')
    per_dx, per_dy = [], []
    for r in ri:
        xc = int(round(float(src_ref['x'][r])))
        yc = int(round(float(src_ref['y'][r])))
        sh = stamp_half
        while sh >= core_half + 1:
            x0 = xc - sh; x1 = xc + sh; y0 = yc - sh; y1 = yc + sh
            if x0 >= 0 and y0 >= 0 and x1 < W and y1 < H:
                break
            sh -= 1
        else:
            continue
        c0 = sh - core_half;  c1 = sh + core_half + 1
        cr = sub_ref [y0 + c0: y0 + c1, x0 + c0: x0 + c1]
        ci = shifted  [y0 + c0: y0 + c1, x0 + c0: x0 + c1]
        diff = ci - cr
        if not np.all(np.isfinite(diff)):
            continue
        gx = np.gradient(cr, axis=1);  gy = np.gradient(cr, axis=0)
        dx2 = float(np.sum(gx ** 2));  dy2 = float(np.sum(gy ** 2))
        if dx2 > 0: per_dx.append(-float(np.sum(diff * gx)) / dx2)
        if dy2 > 0: per_dy.append(-float(np.sum(diff * gy)) / dy2)
    n = min(len(per_dx), len(per_dy))
    if n > 2:
        mad_dx = float(np.median(np.abs(np.array(per_dx) -
                                        np.median(per_dx)))) * 1.4826
        mad_dy = float(np.median(np.abs(np.array(per_dy) -
                                        np.median(per_dy)))) * 1.4826
        return mad_dx / np.sqrt(n), mad_dy / np.sqrt(n)
    return np.nan, np.nan


def _build_sci_composite(img: np.ndarray, positions: list, core_half: int):
    """
    Extract PSF-core stamps from *img* at pre-computed reference positions
    and lay them side by side into a (c, N*c) array.
    """
    c = 2 * core_half + 1
    N = len(positions)
    sci_comp = np.empty((c, N * c), dtype=np.float64)
    for k, (x0, y0, sh) in enumerate(positions):
        c0 = sh - core_half
        c1 = sh + core_half + 1
        sci_comp[:, k * c:(k + 1) * c] = img[y0 + c0: y0 + c1, x0 + c0: x0 + c1]
    return sci_comp


def _align_fallback_source_pixels(t: int, img: np.ndarray,
                                   ref: np.ndarray,
                                   source_mask: np.ndarray) -> dict:
    """
    Fallback alignment matching fit_shift / image_sub behaviour:
    shifts the raw science frame (order-5, mode='nearest') to minimise
    nansum of squared differences against the source-masked reference,
    with a 5-pixel border crop, using Powell with bounds ±3 px.
    """
    fail = dict(t=t, dx=0.0, dy=0.0, err_dx=np.nan, err_dy=np.nan,
                n_stars=0, converged=False)
    try:
        if np.nansum(np.abs(img)) == 0:
            return fail

        # Build masked template — zeros outside source regions become nan
        template = ref.astype(np.float64) * source_mask
        template[template == 0] = np.nan

        H = img.shape[0]
        crop = 5 if H <= 50 else 10

        def loss(theta):
            row_shift, col_shift = theta
            s = nd_shift(img.astype(np.float64), [row_shift, col_shift],
                         order=5, mode='nearest')
            diff = (template - s) ** 2
            return float(np.nansum(diff[crop:-crop, crop:-crop]))

        res = minimize(loss, x0=[0.0, 0.0], method='Powell',
                       bounds=[(-3.0, 3.0), (-3.0, 3.0)],
                       options={'xtol': 1e-7, 'ftol': 1e-10, 'maxiter': 5000})

        row_shift, col_shift = float(res.x[0]), float(res.x[1])

        # Guard against Powell hitting a bound (indicates failure)
        if np.allclose(res.x, [3.0, 3.0]) or np.allclose(res.x, [-3.0, -3.0]):
            return fail

        # Convert to sep_aligner dx/dy convention:
        # shift(img, [row_shift, col_shift]) aligns science to ref
        # _build_shift_array stores arr[t,0]=-dy, arr[t,1]=-dx
        # so dy = -row_shift, dx = -col_shift
        return dict(t=t, dx=-col_shift, dy=-row_shift,
                    err_dx=np.nan, err_dy=np.nan,
                    n_stars=0, converged=True)
    except Exception:
        return fail


def _align_one_frame(t: int, img: np.ndarray,
                     ref_comp: np.ndarray,
                     w_cols: np.ndarray,
                     positions: list,
                     n_stars: int,
                     core_half: int,
                     source_mask: Optional[np.ndarray] = None,
                     ref: Optional[np.ndarray] = None) -> dict:
    """Align one frame. Called by joblib — must be a top-level function.

    Source positions and reference composites are pre-computed once from the
    reference frame. Per-frame work is only background subtraction, stamp
    extraction at known positions, and Nelder-Mead minimisation.
    """
    fail = dict(t=t, dx=0.0, dy=0.0, err_dx=np.nan, err_dy=np.nan,
                n_stars=n_stars, converged=False)
    try:
        img64 = np.ascontiguousarray(img, dtype=np.float64)
        bkg = sep.Background(img64)
        sub_img = img64 - bkg

        sci_comp = _build_sci_composite(sub_img, positions, core_half)

        fn = lambda p: _loss_composite(sci_comp, ref_comp, w_cols, p[0], p[1])
        res = minimize(fn, x0=[0.0, 0.0], method='Powell',
                       bounds=[(-3.0, 3.0), (-3.0, 3.0)],
                       options={'xtol': 1e-7, 'ftol': 1e-10, 'maxiter': 5000})
        dx_opt = float(res.x[0])
        dy_opt = float(res.x[1])

        return dict(t=t, dx=dx_opt, dy=dy_opt,
                    err_dx=np.nan, err_dy=np.nan,
                    n_stars=n_stars, converged=res.success)
    except Exception:
        if source_mask is not None and ref is not None:
            return _align_fallback_source_pixels(t, img, ref, source_mask)
        return fail


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class SepAligner:
    """
    SEP-based sub-pixel image aligner for TESS datacubes.

    Measures the (dx, dy) offset of each frame relative to a reference image
    using 3×3 PSF-core stamp minimisation over SEP-selected point sources,
    then stores the result in a tessreduce-compatible ``shift`` array.

    Parameters
    ----------
    ref : (H, W) ndarray
        Reference image (background-subtracted or raw — background is
        re-estimated internally per frame).
    flux : (T, H, W) ndarray
        Datacube of science frames.  First axis is time.
    n_jobs : int, optional
        Number of parallel workers (joblib).  -1 = all CPUs.  Default -1.
    thresh : float, optional
        SEP source extraction threshold in units of background σ.  Default 3.
    sat_frac : float, optional
        Sources with peak > sat_frac × global_max are considered saturated.
        Default 0.7.
    ell_max : float, optional
        Maximum ellipticity (1 − b/a) for a source to be used.  Default 0.3.
    var_thresh : float, optional
        Maximum fractional flux change between ref and science frame for a
        source to be considered non-variable.  Default 0.05.
    edge_margin : int, optional
        Exclude sources within this many pixels of any image border.
        Default 3.  Never relaxed even in fallback mode.
    coarse_steps : int, optional
        Grid points per axis in the coarse offset search.  Default 21.
    clip_sigma : float, optional
        σ threshold for flagging outlier frames in the output DataFrame.
        Default 3.

    Attributes
    ----------
    shift : (T, 2) ndarray, float32
        Shifts in tessreduce convention: ``shift[t] = [dy, dx]`` such that
        ``scipy.ndimage.shift(frame, shift[t])`` moves the frame content to
        align with the reference.  Zero for failed frames.
    offsets : pd.DataFrame
        Per-frame measurement table with columns
        ``t, dx, dy, err_dx, err_dy, offset, n_stars, converged, sigma_clipped``.

    Examples
    --------
    Standalone usage::

        from sep_aligner import SepAligner
        aligner = SepAligner(tr.ref, tr.flux, n_jobs=-1)
        aligner.run()
        tr.shift = aligner.shift
        tr.shift_images()

    Inside tessreduce (add to tessreduce.py reduce() dispatch)::

        elif self._shift_method == 'sep_core':
            aligner = SepAligner.from_tessreduce(self)
            aligner.run()
    """

    def __init__(self,
                 ref:          np.ndarray,
                 flux:         np.ndarray,
                 n_jobs:       int = -1,
                 thresh:       float = 3.0,
                 sat_frac:     float = 0.7,
                 ell_max:      float = 0.5,
                 var_thresh:   float = 0.05,
                 edge_margin:  int = 3,
                 coarse_steps: int = 21,
                 clip_sigma:   float = 3.0,
                 pixel_mask:     Optional[np.ndarray] = None,
                 use_pixel_mask: bool = True,
                 source_mask:    Optional[np.ndarray] = None):

        if flux.ndim != 3:
            raise ValueError(
                f"flux must be 3-D (T, H, W), got shape {flux.shape}")
        if ref.shape != flux.shape[1:]:
            raise ValueError(
                f"ref shape {ref.shape} must match flux frame shape "
                f"{flux.shape[1:]}")

        self.ref = ref
        self.flux = flux
        self.n_jobs = n_jobs
        self.thresh = thresh
        self.sat_frac = sat_frac
        self.ell_max = ell_max
        self.var_thresh = var_thresh
        self.edge_margin = edge_margin
        self.coarse_steps = coarse_steps
        self.clip_sigma = clip_sigma

        # Build a 2-D boolean mask excluding strap (bit 4) and blended source
        # (bit 2) pixels from star selection
        if use_pixel_mask and pixel_mask is not None:
            m = np.asarray(pixel_mask)
            if m.ndim == 3:
                m = np.any((m & 6) > 0, axis=0)
            else:
                m = (m & 6) > 0
            self._pixel_mask: Optional[np.ndarray] = m
        else:
            self._pixel_mask = None

        # 2-D bool source mask (tessreduce bit 1) for fallback alignment
        if source_mask is not None:
            sm = np.asarray(source_mask)
            self._source_mask: Optional[np.ndarray] = (
                sm.any(axis=0) if sm.ndim == 3 else sm.astype(bool))
        else:
            self._source_mask = None

        # Outputs — populated by run()
        self.shift:   Optional[np.ndarray] = None
        self.offsets: Optional[pd.DataFrame] = None

        # Pre-compute reference products once
        params = _adaptive_params(ref.shape)
        self._params = params
        self._sub_ref, self._src_ref = _extract(
            ref, thresh, params['minarea'])

    # ── Class-method constructor ──────────────────────────────────────────────

    @classmethod
    def from_tessreduce(cls, tr, **kwargs) -> 'SepAligner':
        """
        Construct from a live tessreduce instance.

        Reads ``tr.ref`` and ``tr.flux``, respects ``tr.parallel`` and
        ``tr.num_cores``, and after ``run()`` writes ``tr.shift`` directly.

        Parameters
        ----------
        tr : tessreduce instance
        **kwargs : forwarded to SepAligner.__init__ (override any default).
        """
        n_jobs = tr.num_cores if getattr(tr, 'parallel', True) else 1
        tr_mask = getattr(tr, 'mask', None)
        if tr_mask is not None:
            _sm = ((tr_mask & 1) == 1).astype(float) - (tr_mask & 2).astype(float)
            _sm[_sm <= 0] = 0
            tr_source_mask = _sm.astype(bool)
        else:
            tr_source_mask = None
        inst = cls(ref=np.asarray(tr.ref),
                   flux=np.asarray(tr.flux),
                   n_jobs=kwargs.pop('n_jobs', n_jobs),
                   pixel_mask=kwargs.pop('pixel_mask', tr_mask),
                   source_mask=kwargs.pop('source_mask', tr_source_mask),
                   **kwargs)
        inst._tr = tr   # keep reference so run() can write back
        return inst

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, time: Optional[np.ndarray] = None,
            savgol_window: int = 25,
            verbose: int = 0) -> 'SepAligner':
        """
        Measure and store offsets for every frame, then smooth with a
        Savitzky-Golay filter (order 3) if *time* is provided.

        Parameters
        ----------
        time : (T,) array, optional
            Observation times in days.  If provided, ``savgol_smooth()`` is
            called automatically after alignment.
        savgol_window : int
            Window width passed to ``savgol_smooth()``.  Default 25.
        verbose : int
            joblib verbosity.  0 = silent (default).

        Returns
        -------
        self  (for method chaining)
        """
        T = self.flux.shape[0]
        p = self._params

        # Select sources once from the reference frame
        ri, _ = _select_stars(self._src_ref, self._src_ref,
                               self._sub_ref, self._sub_ref,
                               sat_frac=self.sat_frac, ell_max=self.ell_max,
                               var_thresh=self.var_thresh,
                               match_radius=p['match_radius'],
                               edge_margin=self.edge_margin,
                               pixel_mask=self._pixel_mask)

        bkg_rms = float(np.std(
            self._sub_ref[self._sub_ref < np.percentile(self._sub_ref, 30)]))
        bkg_rms = max(bkg_rms, 1e-6)

        cores, positions, weights = _build_ref_cores(
            self._sub_ref, self._src_ref, ri,
            p['stamp_half'], p['core_half'], bkg_rms)

        if cores is None or len(cores) < 2:
            raise RuntimeError('[SepAligner] fewer than 2 usable sources in reference frame')

        print(f'[SepAligner] {len(cores)} sources selected for alignment')

        _, ref_comp, w_cols = _build_composite(
            self._sub_ref, cores, positions, np.asarray(weights), p['core_half'])

        results = Parallel(n_jobs=self.n_jobs, verbose=verbose,
                           backend="multiprocessing", verbose=1)(
            delayed(_align_one_frame)(
                t, self.flux[t],
                ref_comp, w_cols,
                positions, len(cores), p['core_half'],
                self._source_mask, self.ref)
            for t in range(T))

        results.sort(key=lambda r: r['t'])

        n_fallback = sum(1 for r in results if r['converged'] and r['n_stars'] == 0)
        if n_fallback > 0:
            print(f"!!!WARNING!!! {n_fallback} frame(s) had fewer than 10 sources — "
                  f"source-pixel alignment used as fallback.")

        self.offsets = self._build_offsets_df(results)
        self.shift = self._build_shift_array(results, T)

        if time is not None:
            self.smooth_shift(time, method='savgol', savgol_window=savgol_window,
                              update_shift=True)

        # Write back to tessreduce instance if constructed via from_tessreduce
        if hasattr(self, '_tr'):
            self._tr.shift = self.shift
            if getattr(self._tr, 'diagnostic_plot', False):
                savename = getattr(self._tr, 'savename', None)
                self.plot_source_selection(savename=savename)
                self.plot_source_quality(savename=savename)

        return self

    # ── Output builders ───────────────────────────────────────────────────────

    def _build_offsets_df(self, results: list) -> pd.DataFrame:
        rows = [dict(
            t = r['t'],
            dx = r['dx'],
            dy = r['dy'],
            err_dx = r['err_dx'],
            err_dy = r['err_dy'],
            offset = float(np.hypot(r['dx'], r['dy']))
                        if r['converged'] else np.nan,
            n_stars = r['n_stars'],
            converged = r['converged'],
        ) for r in results]
        df = pd.DataFrame(rows)

        # Flag outlier frames across the time axis
        ok = df['converged']
        df['sigma_clipped'] = False
        if ok.sum() > 3:
            for col in ('dx', 'dy'):
                vals = df.loc[ok, col].values
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                df.loc[ok, 'sigma_clipped'] |= (
                    np.abs(vals - med) > self.clip_sigma * mad)
        return df

    def _build_shift_array(self, results: list, T: int) -> np.ndarray:
        """
        Build (T, 2) float32 shift array in tessreduce convention.

        tessreduce.shift_images() does:
            shifted[i] = ndimage.shift(frame, [shift[i,0], shift[i,1]])
        where positive shift[i,0] moves content DOWN (row increases).

        Our dx, dy are defined as (x_science − x_ref), (y_science − y_ref).
        To bring science into alignment with ref we need to apply (−dx, −dy).
        In scipy row-major convention that is shift=(−dy, −dx).
        So:   shift[t, 0] = −dy   (row shift)
              shift[t, 1] = −dx   (col shift)
        """
        arr = np.zeros((T, 2), dtype=np.float32)
        for r in results:
            t = r['t']
            if r['converged']:
                arr[t, 0] = -r['dy']   # row shift = −dy
                arr[t, 1] = -r['dx']   # col shift = −dx
        return arr

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def dx(self) -> Optional[np.ndarray]:
        """Measured x-offsets (science − ref) in pixels, shape (T,)."""
        return (None if self.offsets is None
                else self.offsets['dx'].to_numpy())

    @property
    def dy(self) -> Optional[np.ndarray]:
        """Measured y-offsets (science − ref) in pixels, shape (T,)."""
        return (None if self.offsets is None
                else self.offsets['dy'].to_numpy())

    @property
    def n_stars(self) -> Optional[np.ndarray]:
        """Number of stars used per frame, shape (T,)."""
        return (None if self.offsets is None
                else self.offsets['n_stars'].to_numpy())

    # ── Time-smoothed shifts ──────────────────────────────────────────────────

    def savgol_smooth(self,
                      time: np.ndarray,
                      window: int = 25,
                      gap_thresh: float = 0.5,
                      update_shift: bool = True,
                      plot: Optional[bool] = None,
                      savename: Optional[str] = None,
                      ) -> np.ndarray:
        """
        Smooth measured shifts with a 3rd-order Savitzky-Golay filter,
        applied independently within each observing segment.

        Parameters
        ----------
        time : (T,) array
            Observation times in days.
        window : int
            Filter window width in frames.  Must be odd and > 3; even values
            are incremented by 1.  Default 25.
        gap_thresh : float
            Cadence gap in days that marks a segment break.  Default 0.5.
        update_shift : bool
            If True (default), overwrite ``self.shift`` with the smoothed
            values.
        plot : bool, optional
            Show diagnostic plot.  If None, reads ``self._tr.diagnostic_plot``
            when constructed via ``from_tessreduce()``.
        savename : str, optional
            If provided, save the plot as ``<savename>_disp_corr.pdf``.

        Returns
        -------
        smoothed : (T, 2) float32 ndarray
        """
        from scipy.signal import savgol_filter as _savgol

        if self.shift is None:
            raise RuntimeError("Call run() before savgol_smooth().")

        T = len(self.shift)
        t_arr = np.asarray(time, dtype=np.float64)

        win = int(window)
        if win % 2 == 0:
            win += 1

        diffs = np.diff(t_arr)
        gap_idx = np.where(diffs > gap_thresh)[0]
        seg_starts = np.concatenate([[0], gap_idx + 1])
        seg_ends = np.concatenate([gap_idx, [T - 1]])
        segments = [np.arange(s, e + 1)
                    for s, e in zip(seg_starts, seg_ends)]

        raw = self.shift.astype(np.float64).copy()
        smoothed = raw.copy()

        for seg_idx in segments:
            n = len(seg_idx)
            if n < 4:
                continue
            w = min(win, n if n % 2 == 1 else n - 1)
            for axis in range(2):
                smoothed[seg_idx, axis] = _savgol(
                    raw[seg_idx, axis], window_length=w, polyorder=3)

        smoothed = smoothed.astype(np.float32)
        if update_shift:
            self.shift = smoothed
            if hasattr(self, '_tr'):
                self._tr.shift = smoothed

        if plot is None:
            plot = getattr(getattr(self, '_tr', None), 'diagnostic_plot', False)
        if plot:
            self._plot_shifts(t_arr, raw, smoothed, gap_thresh, savename)

        return smoothed

    def smooth_shift(self,
                     time: np.ndarray,
                     method: str = 'savgol',
                     gap_thresh: float = 0.5,
                     length_scale: Optional[float] = None,
                     sigma_clip: float = 4.0,
                     sigma_clip_shifts: bool = False,
                     adaptive: bool = True,
                     adaptive_range: float = 3.0,
                     median_filter_width: Optional[int] = 'auto',
                     savgol_window: int = 25,
                     update_shift: bool = True,
                     plot: Optional[bool] = None,
                     savename: Optional[str] = None,
                     ) -> np.ndarray:
        """
        Return a time-smoothed version of the measured shifts.

        Parameters
        ----------
        method : {'savgol', 'gp'}
            Smoothing method.  ``'savgol'`` (default) applies a 3rd-order
            Savitzky-Golay filter with window ``savgol_window``.  ``'gp'``
            uses the error-weighted Gaussian-process smoother.

        All other parameters apply only when ``method='gp'``; ``savgol_window``
        applies only when ``method='savgol'``.

        GP smoother: return a time-smoothed version of the measured shifts using a
        Gaussian-process-inspired weighted smoother, with automatic gap
        detection and error-weighted fitting.

        Data is assumed to be **continuously sampled** within each
        observing segment.  A segment boundary is declared wherever
        consecutive ``time`` values differ by more than ``gap_thresh``
        days (default 0.5 d).  Each segment is smoothed independently
        so the filter never interpolates across a gap.

        Masked / non-converged frames are **excluded from the fit** but
        their smoothed values are filled from the GP prediction evaluated
        at their time position, so the returned array is always complete.

        Smoothing kernel
        ----------------
        For each output frame *i*, the estimate is an error-weighted
        Gaussian kernel average over all *usable* frames *j* in the
        same segment::

            ŝ(tᵢ) = Σⱼ wⱼ · G(tⱼ - tᵢ, ℓ) · sⱼ
                    ─────────────────────────────
                     Σⱼ wⱼ · G(tⱼ - tᵢ, ℓ)

        where wⱼ = 1/σⱼ² (from per-stamp measurement errors) and G is
        a unit Gaussian with width ℓ (``length_scale``).

        Parameters
        ----------
        time : (T,) array of float
            Observation times in days (MJD, BJD, …).  Must have the
            same length as the number of frames T.
        gap_thresh : float
            Cadence gap in days that marks a segment break.  Default 0.5.
        length_scale : float, optional
            Gaussian smoothing width in days.  If None, defaults to
            5 × the median cadence within the first segment (i.e. roughly
            5 frames' worth of smoothing), which is conservative and
            appropriate for TESS 2-min or 30-min cadence data.
        adaptive : bool
            If True (default), dynamically widen the GP kernel where shifts
            are stable and contract it where they change rapidly.
        adaptive_range : float
            Maximum factor by which the length scale can expand or contract
            from the base value.  ``adaptive_range=3`` means the kernel can
            be up to 3× wider or 3× narrower than ``length_scale``.
            Default 3.0.
        median_filter_width : int or 'auto' or None
            Maximum window width (in frames) for the adaptive median post-filter
            applied after the GP smooth.  ``'auto'`` (default) sets the width
            to match the GP length scale converted to frames.  ``None`` disables
            the filter entirely.  Must be odd when specified as int; even values
            are incremented by 1.
        sigma_clip_shifts : bool
            If True, enable outlier rejection before the GP fit.  Default False.
        sigma_clip : float
            Threshold in units of MAD used when ``sigma_clip_shifts=True``.
            Frames deviating more than ``sigma_clip`` × MAD from the segment
            weighted median are excluded from the fit (but still filled from
            the GP).  Default 4.0.
        update_shift : bool
            If True (default), overwrite ``self.shift`` with the smoothed
            values so a subsequent ``apply()`` or
            ``tessreduce.shift_images()`` uses them.
        plot : bool, optional
            Whether to show the diagnostic plot.  If None (default), reads
            ``self._tr.diagnostic_plot`` when constructed via
            ``from_tessreduce()``, otherwise defaults to False.
        savename : str, optional
            If provided, save the plot as ``<savename>_disp_corr.pdf``
            matching tessreduce's naming convention.

        Returns
        -------
        smoothed : (T, 2) float32 ndarray
            Smoothed shifts in tessreduce convention ``[dy, dx]``.
            Every frame is populated — masked frames receive the GP
            interpolated value.
        """
        if method == 'savgol':
            return self.savgol_smooth(
                time, window=savgol_window, gap_thresh=gap_thresh,
                update_shift=update_shift, plot=plot, savename=savename)

        if self.offsets is None:
            raise RuntimeError("Call run() before smooth_shift().")

        T = len(self.offsets)
        t_arr = np.asarray(time, dtype=np.float64)
        if len(t_arr) != T:
            raise ValueError(
                f"time length {len(t_arr)} must match number of frames {T}")

        # ── Segment detection ─────────────────────────────────────────────
        diffs = np.diff(t_arr)
        gap_idx = np.where(diffs > gap_thresh)[0]   # indices just BEFORE each gap
        seg_starts = np.concatenate([[0],      gap_idx + 1])
        seg_ends = np.concatenate([gap_idx,  [T - 1]])
        segments = [np.arange(s, e + 1)
                      for s, e in zip(seg_starts, seg_ends)]

        # ── Default length scale ──────────────────────────────────────────
        first = segments[0]
        if len(first) > 1:
            med_cad = float(np.median(np.diff(t_arr[first])))
        else:
            med_cad = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        if length_scale is None:
            length_scale = 5.0 * med_cad

        # ── Resolve auto median filter width ─────────────────────────────
        if median_filter_width == 'auto':
            _mfw = max(3, int(round(length_scale / med_cad)))
            if _mfw % 2 == 0:
                _mfw += 1
            median_filter_width = _mfw

        # ── Per-frame measurement weights  (1/σ²) ────────────────────────
        # shift[:,0] = -dy,  shift[:,1] = -dx  (tessreduce convention)
        converged = self.offsets['converged'].to_numpy().astype(bool)
        err_dy = self.offsets['err_dy'].to_numpy()
        err_dx = self.offsets['err_dx'].to_numpy()

        star_based = converged & np.isfinite(err_dy) & (err_dy > 0)
        w_dy = np.where(star_based, 1.0 / err_dy**2, 0.0)
        w_dx = np.where(star_based, 1.0 / err_dx**2, 0.0)

        # Fallback frames (converged via source-pixel method) have no error
        # estimate — assign them a weight derived from star-based frames so
        # they participate in the GP smooth with lower confidence.
        fallback = converged & ~star_based
        if fallback.any():
            if star_based.any():
                # Use 10% of median star-based weight
                med_w_dy = float(np.median(w_dy[star_based]))
                med_w_dx = float(np.median(w_dx[star_based]))
            else:
                # No star-based frames at all — assign uniform unit weight
                med_w_dy = 1.0
                med_w_dx = 1.0
            w_dy = np.where(fallback, 0.1 * med_w_dy, w_dy)
            w_dx = np.where(fallback, 0.1 * med_w_dx, w_dx)

        # raw shift columns in tessreduce convention
        raw = self.shift.astype(np.float64).copy()   # (T, 2): [−dy, −dx]

        smoothed = np.full((T, 2), np.nan)

        for seg_idx in segments:
            n = len(seg_idx)
            if n == 0:
                continue
            if n == 1:
                smoothed[seg_idx] = raw[seg_idx]
                continue

            t_seg = t_arr[seg_idx]

            for axis, w_full in enumerate([w_dy, w_dx]):
                vals = raw[seg_idx, axis].copy()     # e.g. −dy for axis 0
                w_seg = w_full[seg_idx].copy()

                # ── sigma-clip by value within segment ────────────────
                usable = w_seg > 0
                if sigma_clip_shifts and usable.sum() >= 3:
                    wmed = self._weighted_median(vals[usable], w_seg[usable])
                    mad = float(np.median(
                        np.abs(vals[usable] - wmed))) * 1.4826
                    if mad > 0:
                        outlier = np.abs(vals - wmed) > sigma_clip * mad
                        w_seg[outlier] = 0.0   # exclude from GP fit
                        usable = w_seg > 0

                # ── GP (error-weighted Gaussian kernel) ───────────────
                if adaptive and usable.sum() >= 3:
                    l_arr = self._adaptive_length_scale(
                        t_seg, vals, length_scale, adaptive_range)
                    sm = self._gp_smooth_adaptive(
                        t_seg, vals, w_seg, usable, l_arr)
                else:
                    sm = self._gp_smooth(t_seg, vals, w_seg, usable,
                                         length_scale)

                # ── adaptive median post-filter ────────────────────────
                if median_filter_width is not None and n >= 3:
                    mfw_max = int(median_filter_width)
                    mfw_min = 3

                    grad = np.abs(np.gradient(sm, t_seg))
                    grad_sm = np.convolve(grad, np.ones(3) / 3.0, mode='same')
                    med_g = float(np.median(grad_sm))
                    if med_g > 1e-30:
                        norm = np.clip(grad_sm / (med_g * adaptive_range), 0.0, 1.0)
                    else:
                        norm = np.zeros(n)
                    raw_w = mfw_max - norm * (mfw_max - mfw_min)
                    win_arr = np.round(raw_w).astype(int)
                    win_arr += 1 - win_arr % 2

                    filtered = sm.copy()
                    for i in range(n):
                        hw = win_arr[i] // 2
                        lo = max(0, i - hw)
                        hi = min(n, i + hw + 1)
                        filtered[i] = np.median(sm[lo:hi])
                    sm = filtered

                smoothed[seg_idx, axis] = sm

        # ── Fallback: any still-NaN positions get nearest-segment-edge ──
        for axis in range(2):
            col = smoothed[:, axis]
            nan_mask = ~np.isfinite(col)
            if nan_mask.any():
                good_idx = np.where(~nan_mask)[0]
                if len(good_idx):
                    col[nan_mask] = col[good_idx[
                        np.searchsorted(good_idx,
                                        np.where(nan_mask)[0]).clip(
                                        0, len(good_idx)-1)]]
            smoothed[:, axis] = col

        smoothed = smoothed.astype(np.float32)
        if update_shift:
            self.shift = smoothed
            if hasattr(self, '_tr'):
                self._tr.shift = smoothed

        # ── Diagnostic plot ───────────────────────────────────────────────
        if plot is None:
            plot = getattr(getattr(self, '_tr', None), 'diagnostic_plot', False)
        if plot:
            self._plot_shifts(t_arr, raw, smoothed, gap_thresh, savename)

        return smoothed

    # ── Diagnostic plot ─────────────────────────────────────────────────

    def _plot_shifts(self, t_arr, raw, smoothed, gap_thresh, savename):
        import matplotlib.pyplot as plt
        raw_dy = -raw[:, 0].copy()
        raw_dx = -raw[:, 1].copy()
        sm_dy = -smoothed[:, 0].copy()
        sm_dx = -smoothed[:, 1].copy()
        gap_idx = np.where(np.diff(t_arr) > gap_thresh)[0]
        sm_dy[gap_idx] = np.nan
        sm_dx[gap_idx] = np.nan
        plt.figure(figsize=(1.5 * fig_width, 1 * fig_width))
        plt.plot(t_arr, raw_dy, '.', label='Row shift', alpha=0.5)
        plt.plot(t_arr, raw_dx, '.', label='Col shift', alpha=0.5)
        plt.plot(t_arr, sm_dy, '-', label='Smoothed row shift')
        plt.plot(t_arr, sm_dx, '-', label='Smoothed col shift')
        plt.ylabel('Shift (pixels)', fontsize=15)
        plt.xlabel('Time (MJD)', fontsize=15)
        plt.legend(fontsize=9)
        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename + '_disp_corr.pdf', bbox_inches='tight')
        plt.show()

    def plot_source_selection(self, savename: Optional[str] = None) -> None:
        """
        Diagnostic plot showing the reference image with accepted and rejected
        SEP sources overlaid, plus stamp/core footprints for accepted sources.

        Green circles  — accepted sources used for alignment.
        Red   circles  — detected but rejected sources.
        Green squares  — stamp region (stamp_half) for each accepted source.
        Cyan  squares  — core region (core_half) used in the MSE loss.

        Parameters
        ----------
        savename : str, optional
            If provided, save as ``<savename>_source_selection.pdf``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle

        sub_ref = self._sub_ref
        src_ref = self._src_ref
        p = self._params
        stamp_half = p['stamp_half']
        core_half = p['core_half']

        # Re-run star selection on the reference against itself to get
        # accepted / rejected index sets (img == ref, so it's self-consistent)
        ri_acc, _ = _select_stars(
            src_ref, src_ref, sub_ref, sub_ref,
            sat_frac=self.sat_frac, ell_max=self.ell_max,
            var_thresh=self.var_thresh,
            match_radius=self._params['match_radius'],
            flag_max=0,
            edge_margin=self.edge_margin,
            pixel_mask=self._pixel_mask)

        acc_set = set(ri_acc.tolist())
        all_idx = np.arange(len(src_ref))
        rej_idx = np.array([i for i in all_idx if i not in acc_set])

        fig, ax = plt.subplots(figsize=(1.5 * fig_width, 1.5 * fig_width))
        vmin, vmax = np.nanpercentile(sub_ref, [1, 99])
        ax.imshow(sub_ref, origin='lower', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='nearest')

        # Rejected sources
        if len(rej_idx):
            ax.scatter(src_ref['x'][rej_idx], src_ref['y'][rej_idx],
                       s=40, facecolors='none', edgecolors='red',
                       linewidths=0.8, label='Rejected')

        # Accepted sources + stamp/core boxes
        for r in ri_acc:
            xc = float(src_ref['x'][r])
            yc = float(src_ref['y'][r])
            ax.scatter(xc, yc, s=40, facecolors='none',
                       edgecolors='limegreen', linewidths=0.8)

            # Stamp box
            sh = stamp_half
            ax.add_patch(Rectangle(
                (xc - sh - 0.5, yc - sh - 0.5),
                2 * sh + 1, 2 * sh + 1,
                linewidth=0.6, edgecolor='limegreen',
                facecolor='none', linestyle='--'))

            # Core box
            ch = core_half
            ax.add_patch(Rectangle(
                (xc - ch - 0.5, yc - ch - 0.5),
                2 * ch + 1, 2 * ch + 1,
                linewidth=0.6, edgecolor='cyan',
                facecolor='none', linestyle='-'))

        # Legend proxies
        legend_handles = [
            mpatches.Patch(facecolor='none', edgecolor='limegreen',
                           label=f'Accepted ({len(ri_acc)})'),
            mpatches.Patch(facecolor='none', edgecolor='red',
                           label=f'Rejected ({len(rej_idx)})'),
            mpatches.Patch(facecolor='none', edgecolor='limegreen',
                           linestyle='--', label='Stamp region'),
            mpatches.Patch(facecolor='none', edgecolor='cyan',
                           label='Core region'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.7)
        H, W = sub_ref.shape
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_title('Source selection — reference image', fontsize=11)
        ax.set_xlabel('Column (px)', fontsize=15)
        ax.set_ylabel('Row (px)', fontsize=15)
        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename + '_source_selection.pdf', bbox_inches='tight')
        plt.show()

    def plot_source_quality(self, savename: Optional[str] = None, print_table: bool = False) -> None:
        """
        Diagnostic overview of which quality conditions each detected reference
        source passes or fails.

        Conditions evaluated (same as _select_stars, self-matched):
          1. In bounds     — centre not within edge_margin of any border
          2. Round         — ellipticity (1 − b/a) < ell_max
          3. Unflagged     — SEP flag == 0
          4. Unsaturated   — peak < sat_frac × global_max
          5. Not masked    — centre pixel not flagged by pixel_mask (bit 4)

        Top panel  : horizontal bar chart — number of sources passing each
                     condition (green) and failing (red).
        Bottom-left: ellipticity vs normalised peak flux, coloured by
                     overall pass (green) / fail (red).
        Bottom-right: SEP flag value histogram.

        Parameters
        ----------
        savename : str, optional
            If provided, save as ``<savename>_source_quality.pdf``.
        """
        import matplotlib.pyplot as plt

        src = self._src_ref
        sub = self._sub_ref
        H, W = sub.shape
        p = self._params
        m = self.edge_margin
        N = len(src)

        if N == 0:
            print('SepAligner: no sources detected in reference — skipping quality plot.')
            return

        sat_level = self.sat_frac * float(sub.max())

        # ── Evaluate each condition per source ────────────────────────────
        in_bounds = (
            (src['x'] >= m) & (src['x'] <= W - 1 - m) &
            (src['y'] >= m) & (src['y'] <= H - 1 - m))

        ell = 1.0 - src['b'] / np.clip(src['a'], 1e-9, None)
        is_round = ell < self.ell_max

        is_unflagged = src['flag'] == 0
        is_unsaturated = src['peak'] < sat_level

        if self._pixel_mask is not None:
            px = np.clip(np.round(src['x']).astype(int), 0, W - 1)
            py = np.clip(np.round(src['y']).astype(int), 0, H - 1)
            not_masked = ~self._pixel_mask[py, px]
        else:
            not_masked = np.ones(N, dtype=bool)

        conditions = {
            'In bounds':    in_bounds,
            'Round':        is_round,
            'Unflagged':    is_unflagged,
            'Unsaturated':  is_unsaturated,
            'Not masked':   not_masked,
        }

        overall_pass = np.ones(N, dtype=bool)
        for v in conditions.values():
            overall_pass &= v

        # ── Plot ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(1.5 * fig_width, 2 * fig_width))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
        ax_bar = fig.add_subplot(gs[0, :])
        ax_scat = fig.add_subplot(gs[1, 0])
        ax_flag = fig.add_subplot(gs[1, 1])

        # Bar chart
        labels = list(conditions.keys())
        n_pass = [int(v.sum()) for v in conditions.values()]
        n_fail = [N - p for p in n_pass]
        y = np.arange(len(labels))
        ax_bar.barh(y, n_pass, color='limegreen', label='Pass')
        ax_bar.barh(y, n_fail, left=n_pass, color='tomato', label='Fail')
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels, fontsize=9)
        ax_bar.set_xlabel('Number of sources', fontsize=15)
        ax_bar.set_title(
            f'Source quality — {N} detected  |  {int(overall_pass.sum())} pass all',
            fontsize=11)
        ax_bar.axvline(N, color='k', linewidth=0.5, linestyle='--')
        for i, (np_, nf) in enumerate(zip(n_pass, n_fail)):
            ax_bar.text(np_ / 2, i, str(np_), va='center', ha='center',
                        fontsize=7, color='white')
            if nf > 0:
                ax_bar.text(np_ + nf / 2, i, str(nf), va='center',
                            ha='center', fontsize=7, color='white')
        ax_bar.legend(fontsize=9, loc='lower right')

        # Ellipticity vs peak scatter
        peak_norm = src['peak'] / max(float(src['peak'].max()), 1e-9)
        colors = np.where(overall_pass, 'limegreen', 'tomato')
        ax_scat.scatter(peak_norm, ell, c=colors, s=15, alpha=0.7,
                        linewidths=0)
        ax_scat.axhline(self.ell_max, color='k', linewidth=0.8,
                        linestyle='--', label=f'ell_max={self.ell_max}')
        ax_scat.axvline(self.sat_frac, color='orange', linewidth=0.8,
                        linestyle='--', label=f'sat_frac={self.sat_frac}')
        ax_scat.set_xlabel('Normalised peak flux', fontsize=15)
        ax_scat.set_ylabel('Ellipticity (1 − b/a)', fontsize=15)
        ax_scat.set_title('Ellipticity vs peak', fontsize=11)
        ax_scat.legend(fontsize=9)

        # SEP flag histogram
        flag_vals = src['flag'].astype(int)
        unique_flags = np.unique(flag_vals)
        ax_flag.bar(unique_flags,
                    [int((flag_vals == f).sum()) for f in unique_flags],
                    color='steelblue', width=0.6)
        ax_flag.set_xlabel('SEP flag value', fontsize=15)
        ax_flag.set_ylabel('Count', fontsize=15)
        ax_flag.set_title('SEP extraction flags', fontsize=11)
        ax_flag.set_xticks(unique_flags)

        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename + '_source_quality.pdf', bbox_inches='tight')
        plt.show()

        # ── Printed table ─────────────────────────────────────────────────
        rows = []
        for i in range(N):
            rows.append({
                'ID':          i,
                'x (px)':      f"{src['x'][i]:.2f}",
                'y (px)':      f"{src['y'][i]:.2f}",
                'peak':        f"{src['peak'][i]:.1f}",
                'flux':        f"{src['flux'][i]:.1f}",
                'In bounds':   'PASS' if bool(in_bounds[i])     else 'FAIL',
                'Round':       'PASS' if bool(is_round[i])      else 'FAIL',
                'Unflagged':   'PASS' if bool(is_unflagged[i])  else 'FAIL',
                'Unsaturated': 'PASS' if bool(is_unsaturated[i])else 'FAIL',
                'Not masked':  'PASS' if bool(not_masked[i])    else 'FAIL',
                'Overall':     'PASS' if bool(overall_pass[i])  else 'FAIL',
            })
        table = pd.DataFrame(rows).set_index('ID')
        if print_table:
            print('\n--- Source quality table ---')
            print(table.to_string())
            print(f"\n{int(overall_pass.sum())}/{N} sources pass all conditions\n")

    # ── Smoothing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _adaptive_length_scale(t_seg: np.ndarray, vals: np.ndarray,
                               base_l: float,
                               adaptive_range: float) -> np.ndarray:
        """
        Compute a per-point length scale that widens where shifts are stable
        and contracts where they change rapidly.

        The local rate of change is estimated as the magnitude of the gradient
        of ``vals`` w.r.t. ``t_seg``, smoothed with a 3-point moving average
        to reduce noise sensitivity.  The length scale at point i is::

            ℓ(i) = base_l / max(rate_norm(i), 1 / adaptive_range)

        clipped to [base_l / adaptive_range, base_l * adaptive_range].
        """
        grad = np.abs(np.gradient(vals, t_seg))
        # 3-point moving average to smooth the rate estimate
        kernel = np.ones(3) / 3.0
        grad_smooth = np.convolve(grad, kernel, mode='same')
        med = float(np.median(grad_smooth))
        if med < 1e-30:
            return np.full(len(t_seg), base_l)
        rate_norm = grad_smooth / med
        l_arr = base_l / np.maximum(rate_norm, 1.0 / adaptive_range)
        return np.clip(l_arr, base_l / adaptive_range, base_l * adaptive_range)

    @staticmethod
    def _gp_smooth_adaptive(t_seg: np.ndarray,
                            vals:   np.ndarray,
                            w_seg:  np.ndarray,
                            usable: np.ndarray,
                            l_arr:  np.ndarray) -> np.ndarray:
        """
        Non-stationary GP smoother with a per-query-point length scale.

        Each query point i uses its own kernel width ℓ(i) from ``l_arr``::

            ŝ(i) = Σ_{j∈usable} w_j · exp(-0.5·(tⱼ-tᵢ)²/ℓ(i)²) · vⱼ
                   ──────────────────────────────────────────────────────
                   Σ_{j∈usable} w_j · exp(-0.5·(tⱼ-tᵢ)²/ℓ(i)²)
        """
        n = len(t_seg)
        sm = vals.copy()

        if not usable.any():
            return sm

        t_use = t_seg[usable]
        v_use = vals[usable]
        w_use = w_seg[usable]

        dt = t_seg[:, None] - t_use[None, :]          # (n, n_usable)
        G = np.exp(-0.5 * (dt / l_arr[:, None]) ** 2)
        WG = w_use[None, :] * G                        # (n, n_usable)
        denom = WG.sum(axis=1)

        valid = denom > 0
        sm[valid] = (WG[valid] @ v_use) / denom[valid]

        if not valid.all():
            t_q = t_seg[~valid]
            near = np.argmin(np.abs(t_q[:, None] - t_use[None, :]), axis=1)
            sm[~valid] = v_use[near]

        return sm

    @staticmethod
    def _weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
        """Weighted median via sorted cumulative weight."""
        idx = np.argsort(vals)
        sv = vals[idx];  sw = weights[idx]
        cw = np.cumsum(sw) / sw.sum()
        return float(np.interp(0.5, cw, sv))

    @staticmethod
    def _gp_smooth(t_seg: np.ndarray,
                   vals:   np.ndarray,
                   w_seg:  np.ndarray,
                   usable: np.ndarray,
                   length_scale: float) -> np.ndarray:
        """
        Error-weighted Gaussian kernel smoother (vectorised).

        For each query point i, computes::

            ŝ(i) = Σ_{j∈usable} w_j · G(tⱼ-tᵢ, ℓ) · vⱼ
                   ─────────────────────────────────────
                   Σ_{j∈usable} w_j · G(tⱼ-tᵢ, ℓ)

        All query points (including masked frames) are evaluated, so
        the output is always fully populated within the segment.
        Falls back to the nearest usable value if no usable points
        exist within ~3ℓ of the query point.
        """
        n = len(t_seg)
        sm = np.full(n, np.nan)

        if not usable.any():
            # No usable points — return raw values as-is
            return vals.copy()

        t_use = t_seg[usable]
        v_use = vals[usable]
        w_use = w_seg[usable]

        # Vectorised: (n_query × n_usable) kernel matrix
        dt = t_seg[:, None] - t_use[None, :]   # (n, n_usable)
        G = np.exp(-0.5 * (dt / length_scale) ** 2)
        WG = w_use[None, :] * G                 # (n, n_usable)
        denom = WG.sum(axis=1)                     # (n,)

        valid_query = denom > 0
        sm[valid_query] = (WG[valid_query] @ v_use) / denom[valid_query]

        # Fill any query points with zero kernel weight (very isolated)
        # using the nearest usable value
        if not valid_query.all():
            t_q = t_seg[~valid_query]
            near = np.argmin(np.abs(t_q[:, None] - t_use[None, :]), axis=1)
            sm[~valid_query] = v_use[near]

        return sm

    # ── Apply alignment to an arbitrary cube ─────────────────────────────────

    def apply(self, cube: Optional[np.ndarray] = None,
              order: int = 3) -> np.ndarray:
        """
        Apply the measured shifts to a cube and return the aligned result.

        Parameters
        ----------
        cube : (T, H, W) ndarray, optional
            Cube to align.  Defaults to ``self.flux``.
        order : int
            Spline interpolation order (default 3).

        Returns
        -------
        aligned : (T, H, W) float32 ndarray
        """
        if self.shift is None:
            raise RuntimeError("Call run() before apply().")
        if cube is None:
            cube = self.flux
        T = cube.shape[0]
        out = np.full_like(cube, np.nan, dtype=np.float32)
        for t in range(T):
            out[t] = nd_shift(cube[t].astype(np.float64),
                              [self.shift[t, 0], self.shift[t, 1]],
                              order=order, mode='constant',
                              cval=np.nan).astype(np.float32)
        return out

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = 'not run' if self.shift is None else (
            f"{int((self.offsets['converged']).sum())}"
            f"/{len(self.offsets)} frames converged")
        return (f"SepAligner(shape={self.flux.shape}, "
                f"n_jobs={self.n_jobs}, status={status})")
