"""
TESSBackgroundSeparator
=======================
Standalone multi-method background separation for TESS flux cubes (T, X, Y).

Signal model
------------
  flux[t, x, y] = background[t, x, y] + astrophysical[t, x, y] + noise[t, x, y]

  background   – scattered light / instrumental (spatially non-coherent)
  astrophysical – localized to source pixels; can contain transients and trends
  noise         – per-pixel random noise

Mask convention (tessreduce bit values)
-----------------------------------------
  bit 1  – stellar source
  bit 2  – saturated source
  bit 4  – strap column (variable multiplicative QE)

Methods
-------
  vectors     physically-driven regression on Earth/Moon separation angles
  savgol      adaptive per-pixel Savitzky-Golay (baseline)
  gp          per-pixel Gaussian Process (RBF kernel, efficient batch solve)
  local_pca   pixel-level decorrelation using nearby background pixels (PLD-style)
  rpca        Robust PCA: low-rank background + sparse astrophysical signal
  nmf         Non-negative Matrix Factorization (requires scikit-learn)
  autoenc     1-D temporal convolutional autoencoder (requires PyTorch)

Flux correction rules
---------------------
  Non-strap pixels : only addition / subtraction  (flux − background_model)
  Strap columns    : division also permitted       (flux / QE)
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import savgol_filter
from scipy.linalg import solve as la_solve
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

try:
    from sklearn.decomposition import NMF as _NMF
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# figure sizing consistent with tessreduce.py
fig_width_pt = 240.0
_ipt = 1.0 / 72.27
fig_width = fig_width_pt * _ipt


# ── module-level helpers ──────────────────────────────────────────────────────

def _nan_interp_axis0(cube: np.ndarray) -> np.ndarray:
    """Fill NaNs along axis=0 via linear interpolation per pixel."""
    out = cube.copy()
    T = cube.shape[0]
    x_all = np.arange(T, dtype=float)
    for idx in np.ndindex(cube.shape[1:]):
        ts = cube[(slice(None),) + idx]
        good = np.isfinite(ts)
        if good.sum() < 2:
            out[(slice(None),) + idx] = 0.0
            continue
        out[(slice(None),) + idx] = np.interp(x_all, x_all[good], ts[good])
    return out


def _make_odd(x: int) -> int:
    x = max(int(x), 3)
    return x if x % 2 == 1 else x + 1


# ── main class ────────────────────────────────────────────────────────────────

class TESSBackgroundSeparator:
    """
    Separate astrophysical signal from background in a TESS flux cube.

    Parameters
    ----------
    flux : ndarray (T, X, Y)
        Raw pixel flux cube in e-/s.
    time : ndarray (T,)
        Timestamps in MJD (or any monotonic unit in days).
    mask : ndarray int (X, Y)
        Pixel classification mask with tessreduce bit values:
          bit 1  – stellar source
          bit 2  – saturated source
          bit 4  – strap column
    expand_mask : bool
        If True (default), run an additional SEP-based pass to detect
        any astrophysical sources not covered by `mask` and add them to
        the internal source mask used for background fitting.
    expand_sigma : float
        Detection threshold (in units of background sigma) for unlabelled
        source detection. Default 3.5.
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        flux: np.ndarray,
        time: np.ndarray,
        mask: np.ndarray,
        expand_mask: bool = True,
        expand_sigma: float = 3.5,
    ):
        self.flux = np.asarray(flux, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.mask = np.asarray(mask, dtype=int)

        self._T, self._X, self._Y = self.flux.shape

        # Decode mask bits
        self._source_mask = ((self.mask & 1) | (self.mask & 2)) > 0   # stars + saturated
        self._strap_mask = (self.mask & 4) > 0                         # strap columns

        # Background pixels are non-source, non-strap
        self._bkg_mask = ~self._source_mask & ~self._strap_mask

        # Optionally expand source mask with SEP-based detection
        self._unlabelled_mask = np.zeros((self._X, self._Y), dtype=bool)
        if expand_mask:
            self._unlabelled_mask = self._detect_unlabelled_sources(expand_sigma)
            self._bkg_mask &= ~self._unlabelled_mask

        # Storage
        self._backgrounds: dict = {}
        self._astro: dict = {}
        self._qe: dict = {}   # strap QE arrays keyed by strap column index

    # ── strap correction ─────────────────────────────────────────────────────

    def correct_straps(
        self,
        smooth_window_frac: float = 0.1,
        min_smooth_window: int = 7,
        gap_thresh: float = 0.5,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Estimate and remove the variable multiplicative quantum efficiency (QE)
        of strap columns (mask bit 4).

        For each strap column *c*:

        1. Identify background rows in *c* (non-source, non-strap pixels).
        2. Estimate the expected background at each strap pixel by linear
           interpolation of neighboring non-strap background columns at the
           same row.
        3. Compute  QE(t, c) = median_row[ flux_strap(t, row, c)
                                           / expected(t, row, c) ]
           restricted to background rows.
        4. Smooth QE(t, c) per time segment with Savitzky-Golay.
        5. Divide:  flux_corrected(t, :, c) = flux(t, :, c) / QE(t, c)

        The corrected flux is stored back into ``self.flux`` in-place.
        The QE arrays are stored in ``self._qe`` (keyed by column index).

        Returns
        -------
        flux_corrected : ndarray (T, X, Y)
            The strap-corrected flux cube (same object as self.flux after
            in-place update).
        """
        strap_cols = np.unique(np.where(self._strap_mask)[1])
        if len(strap_cols) == 0:
            return self.flux

        non_strap_cols = np.array(
            [c for c in range(self._Y) if c not in set(strap_cols)]
        )
        if len(non_strap_cols) < 2:
            warnings.warn(
                "Too few non-strap columns to estimate strap QE. Skipping.",
                UserWarning,
            )
            return self.flux

        segs = self._segment_indices(gap_thresh)

        # Pre-compute spatial-mean background expected value per non-strap col
        # as a function of (T, row) via 1-D column interpolation per row.
        # We work row-by-row because the column interpolation is 1-D in col-space.

        for col in strap_cols:
            # Background rows in this column: not a source, not another strap type
            col_source = self._source_mask[:, col]    # (X,)  row-wise
            bkg_rows = np.where(~col_source)[0]

            if len(bkg_rows) < 3:
                # Not enough background rows — skip this column
                continue

            # Expected flux at strap column from interpolating non-strap columns
            # For each row r and time t: expected = interp(non_strap_cols, flux[t,r,:], col)
            # Vectorise over t by working on the non-strap portion of each row.
            expected = np.full((self._T, self._X), np.nan)
            for row in range(self._X):
                # Non-strap flux values along this row
                ref_flux = self.flux[:, row, non_strap_cols]   # (T, N_ns)
                # Linear interpolation at the strap column position
                expected[:, row] = np.array([
                    np.interp(float(col), non_strap_cols.astype(float), ref_flux[t])
                    for t in range(self._T)
                ])

            # QE per time step from background rows only
            strap_bkg_flux = self.flux[:, bkg_rows, col]   # (T, N_bkg_rows)
            exp_bkg_flux = expected[:, bkg_rows]            # (T, N_bkg_rows)

            # Ratio per time step; protect against near-zero expected values
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = strap_bkg_flux / exp_bkg_flux       # (T, N_bkg_rows)
            ratio = np.where(np.abs(exp_bkg_flux) < 1e-3, np.nan, ratio)

            qe = np.nanmedian(ratio, axis=1)                # (T,)

            # Replace any remaining NaN/zero QE with 1.0 (no correction)
            qe = np.where(np.isfinite(qe) & (np.abs(qe) > 1e-3), qe, 1.0)

            # Smooth QE per segment
            qe_smooth = qe.copy()
            for start, end in segs:
                n = end - start
                if n < 5:
                    continue
                w = _make_odd(max(int(n * smooth_window_frac), min_smooth_window))
                w = min(w, n if n % 2 == 1 else n - 1)
                if w < 3:
                    continue
                seg_qe = qe[start:end]
                # Interpolate any residual NaNs before savgol
                good = np.isfinite(seg_qe)
                if good.sum() < 3:
                    continue
                seg_qe_filled = np.interp(
                    np.arange(n), np.where(good)[0], seg_qe[good]
                )
                qe_smooth[start:end] = savgol_filter(seg_qe_filled, w, 1)

            # Divide strap column by smoothed QE
            self.flux[:, :, col] /= qe_smooth[:, np.newaxis]
            self._qe[int(col)] = qe_smooth

        if plot:
            self._plot_strap_correction(strap_cols)

        return self.flux

    def _plot_strap_correction(self, strap_cols):
        """Diagnostic plot for strap QE correction."""
        n_cols = min(len(strap_cols), 4)
        fig, axes = plt.subplots(n_cols, 1,
                                 figsize=(1.5 * fig_width, n_cols * fig_width),
                                 sharex=True, squeeze=False)
        t = self.time
        gap_idx = np.where(np.diff(t) > 0.5)[0]

        def _ng(arr):
            a = arr.copy().astype(float)
            if len(gap_idx):
                a[gap_idx] = np.nan
            return a

        for ax, col in zip(axes[:, 0], strap_cols[:n_cols]):
            if int(col) in self._qe:
                qe = self._qe[int(col)]
                ax.plot(t, _ng(qe), lw=1.2, label=f'QE col {col}')
                ax.axhline(1.0, color='k', lw=0.8, ls='--')
                ax.set_ylabel('QE', fontsize=9)
                ax.set_title(f'Strap column {col} — smoothed QE', fontsize=9)
                ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel('Time (MJD)', fontsize=9)
        plt.tight_layout()
        plt.show()

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, method: str = 'local_pca', **kwargs) -> np.ndarray:
        """
        Estimate background and return the astrophysical signal cube.

        Parameters
        ----------
        method : str
            One of 'vectors', 'savgol', 'gp', 'local_pca', 'rpca', 'nmf',
            'autoenc'.
        **kwargs
            Forwarded to the individual method.

        Returns
        -------
        astro : ndarray (T, X, Y)
            flux − estimated_background
        """
        _dispatch = {
            'vectors'  : self._fit_vectors,
            'savgol'   : self._fit_savgol,
            'gp'       : self._fit_gp,
            'local_pca': self._fit_local_pca,
            'rpca'     : self._fit_rpca,
            'nmf'      : self._fit_nmf,
            'autoenc'  : self._fit_autoenc,
        }
        if method not in _dispatch:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {list(_dispatch)}."
            )
        bkg = _dispatch[method](**kwargs)

        # ── enforce non-negative residual constraint via inpainting ───────
        # Per-frame noise estimated from background-pixel residuals.
        # Where bkg > flux + k*noise, the background was overestimated for
        # that pixel-time. Mask those locations and inpaint spatially from
        # neighbouring pixels that do satisfy the constraint (same approach
        # as tessreduce's inpaint_biharmonic usage).
        from skimage.restoration import inpaint as sk_inpaint

        k = 3.0
        resid_bkg = (self.flux - bkg)[:, self._bkg_mask]  # (T, N_bkg)
        frame_noise = 1.4826 * np.nanmedian(
            np.abs(resid_bkg - np.nanmedian(resid_bkg, axis=1, keepdims=True)),
            axis=1,
        )  # (T,)

        ceiling = self.flux + (k * frame_noise)[:, np.newaxis, np.newaxis]
        _yy, _xx = np.mgrid[:self._X, :self._Y]
        yx_all = np.column_stack([_yy.ravel(), _xx.ravel()])
        for t in range(self._T):
            bad = bkg[t] > ceiling[t]
            if not bad.any():
                continue
            good = ~bad
            good_yx = np.argwhere(good)
            good_vals = bkg[t][good]
            # Fast spatial interpolation first
            filled_flat = griddata(good_yx, good_vals, yx_all,
                                   method='linear', fill_value=np.nan)
            filled = filled_flat.reshape(self._X, self._Y)
            # Nearest-neighbour for edge NaNs
            still_nan = bad & ~np.isfinite(filled)
            if still_nan.any():
                filled_nn = griddata(good_yx, good_vals, yx_all[still_nan.ravel()],
                                     method='nearest')
                filled[still_nan] = filled_nn
            # Biharmonic only if spatial interpolation still violates ceiling
            still_bad = filled > ceiling[t]
            if still_bad.any():
                frame_bkg = bkg[t].copy()
                frame_bkg[still_bad] = np.nan
                filled_bh = sk_inpaint.inpaint_biharmonic(
                    frame_bkg, still_bad.astype(bool)
                )
                filled[still_bad] = filled_bh[still_bad]
            bkg[t][bad] = filled[bad]

        self._backgrounds[method] = bkg
        self._astro[method] = self.flux - bkg
        return self._astro[method]

    def background(self, method: str) -> np.ndarray:
        """Return the estimated background cube for a fitted method."""
        if method not in self._backgrounds:
            raise KeyError(f"Method '{method}' not fitted. Call fit() first.")
        return self._backgrounds[method]

    def scatter(self, method: str) -> float:
        """
        RMS scatter of astrophysical residuals measured on background pixels.
        Lower is better.
        """
        if method not in self._astro:
            raise KeyError(f"Method '{method}' not fitted. Call fit() first.")
        residuals = self._astro[method][:, self._bkg_mask]
        return float(np.sqrt(np.nanmean(residuals ** 2)))

    def compare(self, methods=None) -> dict:
        """Return {method: scatter_rms} for all fitted (or listed) methods."""
        if methods is None:
            methods = list(self._backgrounds)
        return {m: self.scatter(m) for m in methods if m in self._backgrounds}

    # ── diagnostic plots ──────────────────────────────────────────────────────

    def plot_background(self, method: str, pixels=None, n_sample: int = 5):
        """
        Three-panel diagnostic figure for a single method.

        Panel 1: raw flux + background estimate for sampled pixels.
        Panel 2: astrophysical residual (flux − background).
        Panel 3: blend-weight alpha (savgol) or mean absolute correction.
        """
        if method not in self._backgrounds:
            raise KeyError(f"Fit '{method}' first.")

        bkg = self._backgrounds[method]
        astro = self._astro[method]
        t = self.time

        # --- choose display pixels (mix source + background) ----------------
        if pixels is None:
            src_c = np.argwhere(self._source_mask & ~self._strap_mask)
            bkg_c = np.argwhere(self._bkg_mask)
            rng = np.random.default_rng(42)
            n_src = min(n_sample // 2 + 1, len(src_c))
            n_bk = min(n_sample - n_src, len(bkg_c))
            pix = []
            if n_src > 0:
                pix.append(src_c[rng.choice(len(src_c), n_src, replace=False)])
            if n_bk > 0:
                pix.append(bkg_c[rng.choice(len(bkg_c), n_bk, replace=False)])
            pixels = np.vstack(pix) if pix else np.array([[0, 0]])

        gap_idx = np.where(np.diff(t) > 0.5)[0]

        def _ng(arr):
            a = arr.copy().astype(float)
            if len(gap_idx):
                a[gap_idx] = np.nan
            return a

        fig, axes = plt.subplots(
            3, 1,
            figsize=(1.5 * fig_width, 3.5 * fig_width),
            sharex=True,
        )

        ax = axes[0]
        for k, (yi, xi) in enumerate(pixels):
            c = f'C{k}'
            ax.plot(t, _ng(self.flux[:, yi, xi]),
                    '.', color=c, ms=1.5, alpha=0.3)
            ax.plot(t, _ng(bkg[:, yi, xi]),
                    '-', color=c, lw=1.0, label=f'bkg ({yi},{xi})')
        ax.set_ylabel('Flux (e⁻/s)', fontsize=10)
        ax.set_title(f'{method} — raw (dots) and background (lines)', fontsize=10)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1]
        for k, (yi, xi) in enumerate(pixels):
            ax.plot(t, _ng(astro[:, yi, xi]),
                    '.', color=f'C{k}', ms=1.5, alpha=0.5,
                    label=f'astro ({yi},{xi})')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_ylabel('Residual (e⁻/s)', fontsize=10)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[2]
        if method == 'savgol':
            alpha_all = np.full(self._T, np.nan)
            for start, end in self._segment_indices():
                n = end - start
                if n < 5:
                    continue
                av = np.nanmean(self.flux[start:end][:, self._bkg_mask], axis=1)
                alpha_all[start:end] = self._gradient_blend_weights(av, n)
            ax.plot(t, _ng(alpha_all), lw=1, color='C2', label='α (1=wide, 0=narrow)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Blend weight α', fontsize=10)
        else:
            delta = np.nanmean(np.abs(bkg - self.flux), axis=(1, 2))
            ax.plot(t, _ng(delta), lw=1, color='C3',
                    label='Mean |background correction|')
            ax.set_ylabel('Mean |correction| (e⁻/s)', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('Time (MJD)', fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_comparison(self, example_pixel=None):
        """
        Two-panel comparison of all fitted methods.

        Panel 1: bar chart of RMS scatter per method.
        Panel 2: background estimates for an example source pixel.
        """
        if not self._backgrounds:
            raise RuntimeError("No methods fitted yet. Call fit() first.")

        sc = self.compare()
        meths = list(sc)

        if example_pixel is None:
            src_c = np.argwhere(self._source_mask & ~self._strap_mask)
            if len(src_c) == 0:
                src_c = np.argwhere(~self._strap_mask)
            yi, xi = src_c[np.random.default_rng(0).integers(len(src_c))]
        else:
            yi, xi = example_pixel

        t = self.time
        gap_idx = np.where(np.diff(t) > 0.5)[0]

        def _ng(arr):
            a = arr.copy().astype(float)
            if len(gap_idx):
                a[gap_idx] = np.nan
            return a

        fig, axes = plt.subplots(
            1, 2,
            figsize=(3.0 * fig_width, 1.5 * fig_width),
        )

        ax = axes[0]
        vals = [sc[m] if isinstance(sc[m], float) else 0 for m in meths]
        colors = [f'C{i}' for i in range(len(meths))]
        ax.bar(meths, vals, color=colors)
        ax.set_ylabel('RMS scatter (e⁻/s)', fontsize=10)
        ax.set_title('Background-pixel residual RMS\n(lower = better)', fontsize=10)
        ax.tick_params(axis='x', rotation=30)

        ax = axes[1]
        ax.plot(t, _ng(self.flux[:, yi, xi]),
                '.k', ms=1.5, alpha=0.3, label='Raw flux')
        for m, col in zip(meths, colors):
            ax.plot(t, _ng(self._backgrounds[m][:, yi, xi]),
                    lw=1.2, color=col, label=m)
        ax.set_xlabel('Time (MJD)', fontsize=10)
        ax.set_ylabel('Flux (e⁻/s)', fontsize=10)
        ax.set_title(f'Background estimates — pixel ({yi},{xi})', fontsize=10)
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.show()

    def plot_mask(self):
        """Show the mask with source, strap, unlabelled, and background pixels."""
        display = np.zeros((self._X, self._Y), dtype=int)
        display[self._bkg_mask] = 0                               # background
        display[self._strap_mask] = 1                             # strap
        display[self._source_mask] = 2                            # labelled source
        display[self._unlabelled_mask & ~self._source_mask] = 3   # unlabelled

        cmap = matplotlib.colors.ListedColormap(
            ['#e0e0e0', '#f9a825', '#1565c0', '#c62828']
        )
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        im = ax.imshow(display, origin='lower', cmap=cmap, norm=norm,
                       interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(
            ['Background', 'Strap', 'Source (mask)', 'Source (detected)'],
            fontsize=8,
        )
        ax.set_title('Pixel classification mask', fontsize=10)
        plt.tight_layout()
        plt.show()

    # ── unlabelled-source detection ───────────────────────────────────────────

    def _detect_unlabelled_sources(self, ell_max=0.5, thresh=3.0) -> np.ndarray:
        """
        Identify pixels that behave like astrophysical sources but are not
        flagged in the provided mask.

        Uses SEP (Source Extractor Python) to detect sources in the temporal
        median image. Only sources with ellipticity < ell_max are kept.
        Their pixel footprints are marked using a circular aperture of
        radius = 2.5 * half-light radius (semi-major axis from SEP).

        Parameters
        ----------
        ell_max : float
            Maximum ellipticity (1 - b/a) for a source to be flagged.
            Default 0.5.
        thresh : float
            SEP detection threshold in units of background sigma. Default 3.0.

        Returns
        -------
        unlabelled : ndarray bool (X, Y)
            True for newly detected source pixels.
        """
        import sep
        med_img = np.nanmedian(self.flux, axis=0).astype(np.float64)
        # SEP requires C-contiguous float64
        med_img = np.ascontiguousarray(med_img)
        # Estimate background from non-masked pixels
        try:
            bkg = sep.Background(med_img, mask=~self._bkg_mask.astype(np.uint8))
            data_sub = med_img - bkg.back()
            noise = bkg.globalrms
            objects = sep.extract(data_sub, thresh, err=noise)
        except Exception:
            return np.zeros((self._X, self._Y), dtype=bool)

        unlabelled = np.zeros((self._X, self._Y), dtype=bool)
        ys_g, xs_g = np.mgrid[0:self._X, 0:self._Y]
        for obj in objects:
            if obj['a'] < 1e-3:
                continue
            ell = 1.0 - obj['b'] / obj['a']
            if ell > ell_max:
                continue
            # circular footprint radius = 2.5 * half-light radius (a is semi-major)
            r = max(2.5 * obj['a'], 2.0)
            dist2 = (ys_g - obj['y'])**2 + (xs_g - obj['x'])**2
            pix = dist2 <= r**2
            # Only flag pixels not already labelled as sources
            unlabelled |= pix & ~self._source_mask & ~self._strap_mask
        return unlabelled

    # ── utilities ─────────────────────────────────────────────────────────────

    def _segment_indices(self, gap_thresh: float = 0.5):
        """List of (start, end) index pairs for contiguous time segments."""
        breaks = np.where(np.diff(self.time) > gap_thresh)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [self._T]])
        return list(zip(starts.tolist(), ends.tolist()))

    def _gradient_blend_weights(self, av: np.ndarray, n: int) -> np.ndarray:
        """
        Per-frame blend weight alpha ∈ [0, 1] based on local background gradient.
        alpha = 1  →  stable   →  use wide SavGol
        alpha = 0  →  variable →  use narrow SavGol
        """
        rw = _make_odd(max(min(n // 32, 51), 5))
        rw = min(rw, n if n % 2 == 1 else n - 1)
        av_rough = savgol_filter(av, rw, 1)
        grad = np.abs(np.gradient(av_rough))
        gw = max(min(n // 32, 25), 3)
        grad_smooth = np.convolve(grad, np.ones(gw) / gw, mode='same')
        med_grad = float(np.nanmedian(grad_smooth))
        if med_grad < 1e-10:
            return np.ones(n)
        rate_norm = grad_smooth / med_grad
        return np.clip(1.0 / np.maximum(rate_norm, 1.0), 0.0, 1.0)

    def _rpca_ialm(self, M, lam, max_iter=500, tol=1e-7):
        """
        Inexact Augmented Lagrange Multiplier for Robust PCA.
        Decomposes M = L + S  (low-rank L, sparse S).
        Reference: Lin, Chen, Ma (2010).
        """
        m, n = M.shape
        mu = m * n / (4.0 * (np.sum(np.abs(M)) + 1e-12))
        mu_bar = mu * 1e7
        rho = 1.5
        norm_M = np.linalg.norm(M, 'fro') + 1e-12

        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf) / lam)

        for _ in range(max_iter):
            # Singular value thresholding for L
            U, sv, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
            L = (U * np.maximum(sv - 1.0 / mu, 0.0)) @ Vt

            # Element-wise soft thresholding for S
            tmp = M - L + Y / mu
            S = np.sign(tmp) * np.maximum(np.abs(tmp) - lam / mu, 0.0)

            R = M - L - S
            Y += mu * R
            mu = min(mu * rho, mu_bar)

            if np.linalg.norm(R, 'fro') / norm_M < tol:
                break

        return L, S

    # ── negative flux floor ───────────────────────────────────────────────────

    def floor_negative_flux(self, noise_floor=0.5, max_iter=3):
        """
        Recompute physically impossible negative flux values via inpainting.

        For each frame, pixels where flux < -noise_floor * per-pixel_noise are
        treated as bad and filled using biharmonic inpainting from their
        neighbours (same approach as tessreduce). The process iterates up to
        `max_iter` times per frame until no pixels violate the condition, or
        until no further improvement is possible. Only valid for raw
        (non-differenced) images.

        Parameters
        ----------
        noise_floor : float
            Multiplier on the per-pixel noise estimate. Default 0.5.
        max_iter : int
            Maximum inpainting iterations per frame. Default 3.
        """

        from skimage.restoration import inpaint as sk_inpaint

        med = np.nanmedian(self.flux, axis=0)
        mad = np.nanmedian(np.abs(self.flux - med[np.newaxis, :, :]), axis=0)
        noise = 1.4826 * mad  # (X, Y)
        threshold = -noise_floor * noise  # (X, Y) — always <= 0

        _yy, _xx = np.mgrid[:self._X, :self._Y]
        yx_all = np.column_stack([_yy.ravel(), _xx.ravel()])

        for t in range(self._T):
            frame = self.flux[t].copy()
            for _ in range(max_iter):
                bad = frame < threshold
                if not bad.any():
                    break
                good = ~bad
                good_yx = np.argwhere(good)
                good_vals = frame[good]
                # Fast spatial interpolation from good pixels
                filled_flat = griddata(good_yx, good_vals, yx_all,
                                       method='linear', fill_value=np.nan)
                filled = filled_flat.reshape(self._X, self._Y)
                # Nearest-neighbour fallback for any remaining NaNs
                still_nan = bad & ~np.isfinite(filled)
                if still_nan.any():
                    filled_nn = griddata(good_yx, good_vals, yx_all[still_nan.ravel()],
                                         method='nearest')
                    filled[still_nan] = filled_nn
                # For persistent edge cases use biharmonic inpainting
                still_bad = filled < threshold
                if still_bad.any():
                    frame_masked = frame.copy()
                    frame_masked[still_bad] = np.nan
                    filled_bh = sk_inpaint.inpaint_biharmonic(
                        frame_masked, still_bad.astype(bool)
                    )
                    filled[still_bad] = filled_bh[still_bad]
                frame[bad] = filled[bad]
            self.flux[t] = frame

        return self.flux

    # ── Method 0: Physical vectors (Earth/Moon angles) ────────────────────────

    def _fit_vectors(self, sector, camera, ccd, gap_thresh=0.5):
        """
        Background model driven by physical spacecraft vectors (Earth/Moon angles).

        Uses the `tessvectors` package to retrieve Earth angle, Moon angle from
        boresight as a function of time. Fits a linear harmonic model per pixel
        using background pixels, then predicts background for all pixels.

        Parameters
        ----------
        sector : int
        camera : int
        ccd    : int
        """
        try:
            import tessvectors
        except ImportError:
            raise ImportError(
                "tessvectors is required for the 'vectors' method. "
                "See https://github.com/tessgi/tessvectors"
            )

        # Get vectors for this sector/camera/ccd
        # tessvectors returns a table with columns including 'time', 'EarthAngle', 'MoonAngle'
        vec_table = tessvectors.get_vectors(sector, camera, ccd)
        vec_time = np.array(vec_table['time'])   # MJD

        # Interpolate vectors to our time grid
        from scipy.interpolate import interp1d

        def _interp_vec(col_name):
            v = np.array(vec_table[col_name], dtype=float)
            f = interp1d(vec_time, v, bounds_error=False, fill_value='extrapolate')
            return f(self.time)

        # Try common column names used by tessvectors
        earth_angle = _interp_vec('EarthAngle') if 'EarthAngle' in vec_table.colnames else None
        moon_angle = _interp_vec('MoonAngle') if 'MoonAngle' in vec_table.colnames else None

        # Build design matrix: [1, sin(e), cos(e), sin(m), cos(m)]
        cols = [np.ones(self._T)]
        if earth_angle is not None:
            cols += [np.sin(np.radians(earth_angle)), np.cos(np.radians(earth_angle))]
        if moon_angle is not None:
            cols += [np.sin(np.radians(moon_angle)), np.cos(np.radians(moon_angle))]
        A = np.column_stack(cols)  # (T, n_features)

        bkg = deepcopy(self.flux)
        bkg_yx = np.argwhere(self._bkg_mask)

        if len(bkg_yx) < 3:
            return bkg

        # Fit all pixels simultaneously: solve A @ C = F  (T, n_feat) @ (n_feat, X*Y) = (T, X*Y)
        flux_2d = self.flux.reshape(self._T, -1)  # (T, X*Y)
        nan_cols = ~np.all(np.isfinite(flux_2d), axis=0)
        flux_clean = flux_2d.copy()
        flux_clean[:, nan_cols] = 0.0

        try:
            C, _, _, _ = np.linalg.lstsq(A, flux_clean, rcond=None)  # (n_feat, X*Y)
            bkg_2d = A @ C  # (T, X*Y) — purely additive prediction
        except np.linalg.LinAlgError:
            return bkg

        bkg_2d[:, nan_cols] = np.nan
        bkg = bkg_2d.reshape(self._T, self._X, self._Y)
        return bkg

    # ── Method 1: Adaptive Savitzky-Golay ────────────────────────────────────

    def _fit_savgol(self, gap_thresh: float = 0.5) -> np.ndarray:
        """
        Per-pixel adaptive Savitzky-Golay background.

        Background pixels are smoothed temporally. Source and strap pixels
        receive a background estimate by spatial linear interpolation from
        the surrounding background pixel values at each frame. This prevents
        source flux from contaminating the background model.

        For each time segment:
        - Compute the gradient of the spatial-mean background (background pixels only).
        - Derive a blend weight alpha per frame.
        - Apply:  bkg = alpha * savgol(wide) + (1-alpha) * savgol(narrow)
          on background pixels only, then interpolate spatially to all pixels.
        """


        bkg = np.full_like(self.flux, np.nan)
        segs = self._segment_indices(gap_thresh)

        bkg_yx = np.argwhere(self._bkg_mask)  # (N_bkg, 2)
        all_yx = np.array([[y, x] for y in range(self._X) for x in range(self._Y)])

        for start, end in segs:
            n = end - start
            if n < 5:
                continue

            seg = self.flux[start:end]                        # (n, X, Y)
            av = np.nanmean(seg[:, self._bkg_mask], axis=1)  # (n,)
            alpha = self._gradient_blend_weights(av, n)

            w_wide = _make_odd(max(n // 4, 5))
            w_narrow = _make_odd(max(n // 32, 5))
            w_wide = min(w_wide, n if n % 2 == 1 else n - 1)
            w_narrow = min(w_narrow, n if n % 2 == 1 else n - 1)

            # Smooth only background pixel time series
            bkg_ts = seg[:, self._bkg_mask]  # (n, N_bkg)
            bkg_ts_filled = _nan_interp_axis0(bkg_ts[:, :, np.newaxis])[:, :, 0]
            sv_wide = savgol_filter(bkg_ts_filled, w_wide, 1, axis=0)
            sv_narrow = savgol_filter(bkg_ts_filled, w_narrow, 1, axis=0)

            a = alpha[:, np.newaxis]
            bkg_ts_smooth = a * sv_wide + (1.0 - a) * sv_narrow  # (n, N_bkg)

            # Spatial interpolation per frame to fill source/strap pixels
            result = np.empty((n, self._X, self._Y), dtype=float)
            for ti in range(n):
                vals = bkg_ts_smooth[ti]  # (N_bkg,)
                frame_bkg = griddata(
                    bkg_yx, vals, all_yx, method='linear', fill_value=np.nan,
                )
                # Nearest-neighbour fill for any remaining NaNs at edges
                nan_mask = ~np.isfinite(frame_bkg)
                if nan_mask.any():
                    frame_bkg[nan_mask] = griddata(
                        bkg_yx, vals, all_yx[nan_mask], method='nearest',
                    )
                result[ti] = frame_bkg.reshape(self._X, self._Y)

            result[~np.isfinite(seg)] = np.nan
            bkg[start:end] = result

        return bkg

    # ── Method 2: Gaussian Process ────────────────────────────────────────────

    def _fit_gp(
        self,
        gap_thresh: float = 0.5,
        l_fraction: float = 0.15,
    ) -> np.ndarray:
        """
        Per-pixel Gaussian Process (RBF kernel), batch-solved over all pixels.

        The covariance matrix K depends only on the time grid (shared by all
        pixels), so K^{-1} is computed once per segment and applied to all
        X*Y pixels simultaneously via a single linear solve.

        l = l_fraction * n  (in frame units).
        noise_var estimated from high-frequency residuals of the spatial mean.
        """
        bkg = deepcopy(self.flux)
        segs = self._segment_indices(gap_thresh)
        frames = np.arange(self._T, dtype=float)

        for start, end in segs:
            n = end - start
            if n < 5:
                continue

            seg = self.flux[start:end]               # (n, X, Y)
            f_idx = frames[start:end]                # (n,) frame indices

            av = np.nanmean(seg[:, self._bkg_mask], axis=1)
            l = l_fraction * n

            # Noise from HF residuals
            rw = _make_odd(max(min(n // 16, 51), 5))
            rw = min(rw, n if n % 2 == 1 else n - 1)
            av_s = savgol_filter(av, rw, 1)
            noise_var = max(float(np.nanvar(av - av_s)), 1e-10)
            sig_var = max(float(np.nanvar(av)), noise_var)

            # RBF kernel
            di = f_idx[:, None] - f_idx[None, :]       # (n, n)
            K_sig = sig_var * np.exp(-0.5 * (di / l) ** 2)
            K = K_sig + noise_var * np.eye(n)

            # Solve only for background pixels, then interpolate spatially


            bkg_yx = np.argwhere(self._bkg_mask)
            all_yx = np.array([[y, x] for y in range(self._X) for x in range(self._Y)])

            bkg_ts = seg[:, self._bkg_mask]                  # (n, N_bkg)
            nan_cols = ~np.all(np.isfinite(bkg_ts), axis=0)
            bkg_clean = bkg_ts.copy()
            bkg_clean[:, nan_cols] = 0.0

            try:
                coeff = la_solve(K, bkg_clean)               # (n, N_bkg)
                bkg_smooth = K_sig @ coeff                   # (n, N_bkg)
            except np.linalg.LinAlgError:
                continue

            bkg_smooth[:, nan_cols] = np.nan

            # Spatial interpolation per frame
            result = np.empty((n, self._X, self._Y), dtype=float)
            for ti in range(n):
                vals = bkg_smooth[ti]
                ok = np.isfinite(vals)
                if ok.sum() < 3:
                    result[ti] = np.nan
                    continue
                frame_bkg = griddata(bkg_yx[ok], vals[ok], all_yx,
                                     method='linear', fill_value=np.nan)
                nan_mask = ~np.isfinite(frame_bkg)
                if nan_mask.any():
                    frame_bkg[nan_mask] = griddata(
                        bkg_yx[ok], vals[ok], all_yx[nan_mask], method='nearest',
                    )
                result[ti] = frame_bkg.reshape(self._X, self._Y)

            result[~np.isfinite(seg)] = np.nan
            bkg[start:end] = result

        return bkg

    # ── Method 3: Local PCA (PLD-style) ───────────────────────────────────────

    def _fit_local_pca(
        self,
        radius: float = 5.0,
        n_components: int = 3,
        gap_thresh: float = 0.5,
    ) -> np.ndarray:
        """
        Pixel-level decorrelation using nearby background pixels.

        Source pixels:
          1. Find background pixels within `radius` (expand if needed).
          2. Normalise their time series; compute SVD → k temporal basis vectors.
          3. Regress source pixel time series against basis + constant.
          4. Fitted systematic = background estimate.

        Background pixels:
          Per-pixel SavGol with window n//4.

        Robust to spatial non-coherence because only LOCAL neighbours are used.
        """
        bkg = deepcopy(self.flux)
        segs = self._segment_indices(gap_thresh)
        bkg_yx = np.argwhere(self._bkg_mask)      # (N_bkg, 2)
        src_yx = np.argwhere(self._source_mask | self._unlabelled_mask)  # (N_src, 2)

        # ── Background pixels: per-pixel SavGol ──────────────────────────────
        for (yi, xi) in bkg_yx:
            ts = self.flux[:, yi, xi].copy()
            for start, end in segs:
                n = end - start
                if n < 5:
                    continue
                w = _make_odd(max(n // 4, 5))
                w = min(w, n if n % 2 == 1 else n - 1)
                seg_ts = ts[start:end]
                good = np.isfinite(seg_ts)
                if good.sum() < 3:
                    continue
                filled = np.interp(np.arange(n), np.where(good)[0], seg_ts[good])
                ts[start:end] = savgol_filter(filled, w, 1)
            bkg[:, yi, xi] = ts

        # ── Source pixels: local PCA ──────────────────────────────────────────
        if len(bkg_yx) < 2 or len(src_yx) == 0:
            return bkg

        tree = cKDTree(bkg_yx)

        for (ys, xs) in src_yx:
            # Expand radius until enough neighbours
            r = radius
            while True:
                idx = tree.query_ball_point([ys, xs], r)
                if len(idx) >= n_components + 1:
                    break
                r *= 1.5
                if r > max(self._X, self._Y) * 2:
                    break
            if len(idx) < 2:
                continue

            nbrs = bkg_yx[idx]   # (M, 2)

            # Process each segment
            for start, end in segs:
                n = end - start
                if n < 5:
                    continue

                # Local background time series matrix  (M, n)
                B = self.flux[start:end, nbrs[:, 0], nbrs[:, 1]].T.astype(float)

                # Normalise each reference pixel
                mu_B = np.nanmean(B, axis=1, keepdims=True)
                std_B = np.nanstd(B, axis=1, keepdims=True)
                std_B[std_B < 1e-10] = 1.0
                B_norm = (B - mu_B) / std_B

                # Drop mostly-NaN rows
                valid = np.sum(np.isfinite(B_norm), axis=1) > n // 2
                if valid.sum() < 1:
                    continue
                B_norm = np.nan_to_num(B_norm[valid], nan=0.0)

                k_eff = min(n_components, B_norm.shape[0])
                try:
                    _, _, Vt = np.linalg.svd(B_norm, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                basis = Vt[:k_eff].T          # (n, k_eff)

                flux_pix = self.flux[start:end, ys, xs].copy()
                good = np.isfinite(flux_pix)
                if good.sum() < k_eff + 2:
                    continue

                A = np.column_stack([basis[good], np.ones(good.sum())])
                c, _, _, _ = np.linalg.lstsq(A, flux_pix[good], rcond=None)

                A_full = np.column_stack([basis, np.ones(n)])
                bkg[start:end, ys, xs] = A_full @ c

        return bkg

    # ── Method 4: Robust PCA ──────────────────────────────────────────────────

    def _fit_rpca(
        self,
        lam: float = None,
        max_iter: int = 500,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """
        Robust PCA via Inexact ALM (Principal Component Pursuit).

        Decomposes flux matrix M = L + S where:
          L  low-rank   → background / scattered light
          S  sparse     → astrophysical transients

        Does not assume spatial coherence.  L is low-rank globally, driven
        by the small number of physical processes causing the background.
        """
        T, X, Y = self._T, self._X, self._Y
        M_full = self.flux.reshape(T, X * Y).T      # (X*Y, T)

        # normalise only; rescaled additively after decomposition
        row_means = np.nanmean(M_full, axis=1, keepdims=True)
        row_means[row_means == 0] = 1.0
        M_norm = M_full / row_means
        nan_mask = ~np.isfinite(M_norm)
        M_clean = np.where(nan_mask, 0.0, M_norm)

        lam_val = lam if lam is not None else 1.0 / np.sqrt(max(X * Y, T))
        L_norm, _ = self._rpca_ialm(M_clean, lam_val, max_iter, tol)

        L = (L_norm * row_means).T.reshape(T, X, Y)
        L[~np.isfinite(self.flux)] = np.nan
        return L

    # ── Method 5: NMF ────────────────────────────────────────────────────────

    def _fit_nmf(
        self,
        n_components: int = 5,
        max_iter: int = 500,
    ) -> np.ndarray:
        """
        Non-negative Matrix Factorization background.

        NMF is physically motivated: flux and background are non-negative.
        Fit the temporal basis W only on background pixels, then apply
        to all pixels via non-negative least squares.

        Requires scikit-learn.
        """
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for the 'nmf' method. "
                "Install with:  pip install scikit-learn"
            )
        T, X, Y = self._T, self._X, self._Y
        flux_flat = self.flux.reshape(T, X * Y)

        offset = max(0.0, -float(np.nanmin(flux_flat))) + 1.0
        flux_pos = flux_flat + offset

        bkg_idx = np.where(self._bkg_mask.ravel())[0]
        flux_bkg = flux_pos[:, bkg_idx]

        # Handle NaN
        col_med = np.nanmedian(flux_bkg, axis=0)
        nan_loc = ~np.isfinite(flux_bkg)
        flux_bkg_clean = flux_bkg.copy()
        flux_bkg_clean[nan_loc] = np.take(col_med, np.where(nan_loc)[1])

        k = min(n_components, min(flux_bkg_clean.shape) - 1)
        nmf = _NMF(n_components=k, init='nndsvd', max_iter=max_iter, random_state=0)
        W = nmf.fit_transform(flux_bkg_clean)   # (T, k)

        bkg_flat = np.zeros((T, X * Y))
        for px in range(X * Y):
            ts = flux_pos[:, px]
            ok = np.isfinite(ts)
            if ok.sum() < k:
                bkg_flat[:, px] = np.nan
                continue
            c, _ = nnls(W[ok], ts[ok])
            bkg_flat[:, px] = W @ c

        bkg_flat -= offset
        bkg_flat[~np.isfinite(flux_flat)] = np.nan
        return bkg_flat.reshape(T, X, Y)

    # ── Method 6: Convolutional Autoencoder ───────────────────────────────────

    def _fit_autoenc(
        self,
        bottleneck: int = 32,
        n_epochs: int = 300,
        lr: float = 5e-4,
        batch_size: int = 256,
        noise_scale: float = 0.05,
        n_neighbors: int = 16,
    ) -> np.ndarray:
        """
        1-D temporal convolutional autoencoder trained on background pixels.

        Architecture: multi-scale Conv1d encoder → bottleneck → ConvTranspose1d decoder.

        Training data: background pixel time series (globally normalised, denoising).
        Inference:
          - Background pixels: use their own reconstruction.
          - Source/strap pixels: replace latent code with IDW-interpolated code from
            the n_neighbors nearest background pixels, then decode. This prevents
            source-flux contamination of the background estimate.

        Normalisation: global background statistics (median/MAD across all background
        pixels at each time step) so that per-pixel mean is not inflated by sources.

        Requires PyTorch.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for the 'autoenc' method. "
                "Install with:  pip install torch"
            )

        T, X, Y = self._T, self._X, self._Y
        bkg_flat_mask = self._bkg_mask.ravel()           # (X*Y,) bool

        flux_2d = self.flux.reshape(T, X * Y).T          # (X*Y, T)
        bkg_rows = flux_2d[bkg_flat_mask]                 # (N_bkg, T)

        # Global normalisation: per-time-step median/MAD over background pixels
        g_med = np.nanmedian(bkg_rows, axis=0, keepdims=True)   # (1, T)
        g_mad = np.nanmedian(np.abs(bkg_rows - g_med), axis=0, keepdims=True) * 1.4826 + 1e-8

        # Per-pixel residual normalisation: remove local DC offset
        def _normalise(rows):
            z = (rows - g_med) / g_mad        # global scale
            dc = np.nanmean(z, axis=1, keepdims=True)
            return np.nan_to_num(z - dc, nan=0.0)

        tr_norm = _normalise(bkg_rows)        # (N_bkg, T)
        dc_bkg = np.nanmean((bkg_rows - g_med) / g_mad, axis=1)   # (N_bkg,) per-pixel offsets

        class _CAE(nn.Module):
            def __init__(self, seq_len, bn):
                super().__init__()
                # multi-scale encoder: three parallel kernel sizes
                self.enc_a = nn.Sequential(nn.Conv1d(1, 16,  7, padding=3),  nn.GELU())
                self.enc_b = nn.Sequential(nn.Conv1d(1, 16, 15, padding=7),  nn.GELU())
                self.enc_c = nn.Sequential(nn.Conv1d(1, 16, 31, padding=15), nn.GELU())
                self.merge = nn.Sequential(
                    nn.Conv1d(48, 32, 5, padding=2), nn.GELU(),
                    nn.Conv1d(32, 16, 3, padding=1), nn.GELU(),
                    nn.AdaptiveAvgPool1d(bn),
                )
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(16, 32, 5, padding=2), nn.GELU(),
                    nn.ConvTranspose1d(32, 16, 7, padding=3), nn.GELU(),
                    nn.Conv1d(16, 1, 3, padding=1),
                    nn.Upsample(size=seq_len, mode='linear', align_corners=False),
                )
            def encode(self, x):
                return self.merge(torch.cat([self.enc_a(x), self.enc_b(x), self.enc_c(x)], dim=1))
            def decode(self, z):
                return self.dec(z)
            def forward(self, x):
                return self.decode(self.encode(x))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _CAE(T, bottleneck).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.1)
        crit = nn.HuberLoss(delta=1.0)

        X_tr = torch.tensor(tr_norm[:, None, :], dtype=torch.float32, device=device)
        N_bkg = X_tr.shape[0]
        model.train()
        rng_t = torch.Generator(device=device)
        for ep in range(n_epochs):
            idx = torch.randperm(N_bkg, generator=rng_t, device=device)
            ep_loss = 0.0
            for start in range(0, N_bkg, batch_size):
                batch = X_tr[idx[start:start + batch_size]]
                noisy = batch + noise_scale * torch.randn_like(batch)
                opt.zero_grad()
                loss = crit(model(noisy), batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sched.step()

        # ── Encode background pixels → get latent codes ────────────────────
        model.eval()
        with torch.no_grad():
            latents_bkg = model.encode(X_tr).cpu().numpy()   # (N_bkg, bn, 1or...)
            # AdaptiveAvgPool → shape (N_bkg, 16, bottleneck) after merge; flatten last
            latents_bkg = latents_bkg.reshape(N_bkg, -1)     # (N_bkg, 16*bottleneck)

        # ── Spatial coordinates of background pixels ───────────────────────
        bkg_yx = np.argwhere(bkg_flat_mask.reshape(X, Y))    # (N_bkg, 2)
        src_yx = np.argwhere(~bkg_flat_mask.reshape(X, Y))   # (N_src, 2)

        tree = cKDTree(bkg_yx)
        dist, idx_nn = tree.query(src_yx, k=min(n_neighbors, N_bkg))
        dist = np.maximum(dist, 1e-6)
        w = 1.0 / dist                                         # (N_src, k)
        w /= w.sum(axis=1, keepdims=True)
        latents_src = np.stack([(w[i] @ latents_bkg[idx_nn[i]]) for i in range(len(src_yx))])

        # DC offset for source pixels: IDW of neighbouring background pixel DCs
        dc_src = np.stack([(w[i] @ dc_bkg[idx_nn[i]]) for i in range(len(src_yx))])

        # ── Decode all pixels ──────────────────────────────────────────────
        all_latents = np.zeros((X * Y, latents_bkg.shape[1]), dtype=np.float32)
        all_latents[bkg_flat_mask] = latents_bkg
        all_latents[~bkg_flat_mask] = latents_src.astype(np.float32)

        all_dc = np.zeros(X * Y, dtype=np.float64)
        all_dc[bkg_flat_mask] = dc_bkg
        all_dc[~bkg_flat_mask] = dc_src

        bn_size = bottleneck
        ch = all_latents.shape[1] // bn_size
        with torch.no_grad():
            z_t = torch.tensor(
                all_latents.reshape(X * Y, ch, bn_size), dtype=torch.float32, device=device
            )
            recon_norm = model.decode(z_t).squeeze(1).cpu().numpy()  # (X*Y, T)

        # Add per-pixel DC back, then undo global normalisation
        recon_z = recon_norm + all_dc[:, None]
        bkg_flat = recon_z * g_mad + g_med                    # (X*Y, T)
        bkg_flat[~np.isfinite(flux_2d)] = np.nan
        return bkg_flat.T.reshape(T, X, Y)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        bkg_method='savgol',
        ref_frame=None,
        sector=None,
        camera=None,
        ccd=None,
        plot=False,
        **bkg_kwargs,
    ):
        """
        Full two-stage background separation and image differencing pipeline.

        Stage 1 — background on raw data
        ---------------------------------
        1. Floor negative flux values.
        2. Correct strap column QE (division only, straps only).
        3. Estimate background with `bkg_method`.
        4. Subtract background → stage-1 images.
        5. Calculate alignment shifts via SepAligner on stage-1 images.

        Stage 2 — differencing pipeline
        --------------------------------
        6. Start from QE-corrected original flux.
        7. Apply alignment shifts (ndimage shift, order=5).
        8. Subtract the chosen reference frame.
        9. Recompute background on difference images with a reduced mask:
           - Strap columns (bit 4) are no longer masked.
           - Source mask is morphologically eroded by ~50% in area.
        10. Subtract stage-2 background.

        Parameters
        ----------
        bkg_method : str
            Background method for both stages (default 'savgol').
        ref_frame : int or None
            Index of the reference frame. If None, use the frame with the
            lowest median background (good proxy for low scattered light).
        sector, camera, ccd : int or None
            Required only if bkg_method='vectors'.
        plot : bool
            If True, produce diagnostic plots at each stage.

        Returns
        -------
        diff_cube : ndarray (T, X, Y)
            Background-subtracted, aligned, differenced image cube.
        shifts : ndarray (T, 2)
            Alignment results from SepAligner in tessreduce convention
            (shift[t] = [dy_apply, dx_apply]).
        stage1_bkg : ndarray (T, X, Y)
            Stage-1 background estimate.
        stage2_bkg : ndarray (T, X, Y)
            Stage-2 background estimate on difference images.
        """
        from scipy.ndimage import shift as nd_shift
        from scipy.ndimage import binary_erosion
        from .sep_aligner import SepAligner

        # ── Stage 1 ────────────────────────────────────────────────────────────
        print('[TESSBackgroundSeparator] Stage 1: raw background estimation')

        # Floor unphysical negatives
        self.floor_negative_flux()

        # QE-correct strap columns
        flux_qe = self.correct_straps(plot=plot)  # modifies self.flux in-place

        # Estimate background
        extra = {}
        if bkg_method == 'vectors':
            if sector is None or camera is None or ccd is None:
                raise ValueError("sector, camera, ccd required for vectors method")
            extra = dict(sector=sector, camera=camera, ccd=ccd)
        extra.update(bkg_kwargs)

        self.fit(method=bkg_method, **extra)
        stage1_sub = flux_qe - self._backgrounds[bkg_method]

        # Choose reference frame: lowest median background if not specified
        if ref_frame is None:
            med_bkg = np.nanmedian(self._backgrounds[bkg_method], axis=(1, 2))
            ref_frame = int(np.argmin(med_bkg))
            print(f'[TESSBackgroundSeparator] Reference frame: {ref_frame} '
                  f'(lowest background)')

        # ── Alignment ─────────────────────────────────────────────────────────
        print('[TESSBackgroundSeparator] Calculating alignment shifts via SepAligner')

        # SepAligner requires a reference image and the full cube
        ref_image_for_align = stage1_sub[ref_frame]

        # Source mask for SepAligner: combine labelled + detected sources (2D bool)
        sep_source_mask = (self._source_mask | self._unlabelled_mask).astype(bool)

        aligner = SepAligner(
            ref=ref_image_for_align,
            flux=stage1_sub,
            source_mask=sep_source_mask,
            pixel_mask=self._strap_mask.astype(int),
        )
        aligner.run()

        # aligner.shift is (T, 2) where shift[t] = [dy_apply, dx_apply]
        # (tessreduce convention: apply as nd_shift(frame, [dy, dx]))

        # ── Stage 2: apply to QE-corrected original ────────────────────────────
        print('[TESSBackgroundSeparator] Stage 2: align, difference, background')

        # Apply shifts
        flux_aligned = np.empty_like(flux_qe)
        shifts_arr = aligner.shift  # (T, 2): [dy_apply, dx_apply]
        for i in range(self._T):
            if shifts_arr is not None and np.all(np.isfinite(shifts_arr[i])):
                dy_apply = float(shifts_arr[i, 0])
                dx_apply = float(shifts_arr[i, 1])
                flux_aligned[i] = nd_shift(
                    flux_qe[i].astype(np.float64),
                    [dy_apply, dx_apply],
                    order=5,
                    mode='nearest',
                )
            else:
                flux_aligned[i] = flux_qe[i]

        # Subtract reference frame (additive/subtractive only — no division)
        ref_image = flux_aligned[ref_frame].copy()
        diff_cube = flux_aligned - ref_image[np.newaxis, :, :]

        # Reduced mask for stage 2:
        # - Ignore strap columns (bit 4 not masked)
        # - Erode source mask to ~50% area (area ∝ r², so erode by 1 pixel ~reduces area)
        # Erosion factor: iterate until area ≤ 50% of original
        src_mask_orig = self._source_mask | self._unlabelled_mask
        n_src_orig = src_mask_orig.sum()
        eroded = src_mask_orig.copy()
        for _ in range(20):
            candidate = binary_erosion(eroded)
            if candidate.sum() <= n_src_orig * 0.5:
                eroded = candidate
                break
            eroded = candidate

        # Build stage-2 separator (no strap masking, reduced source mask)
        # Encode eroded source mask as bit 1 (no bit 4 since we don't mask straps now)
        stage2_mask = eroded.astype(int)  # bit 1 = source only, no strap bit

        sep2 = TESSBackgroundSeparator(
            diff_cube,
            self.time,
            stage2_mask,
            expand_mask=False,  # don't re-detect; differenced images are unusual
        )
        # In differenced images negative values are permitted in source cores
        # so we do NOT call floor_negative_flux here

        sep2.fit(method=bkg_method, **bkg_kwargs)
        diff_cube_bkg_sub = diff_cube - sep2._backgrounds[bkg_method]

        if plot:
            self._plot_pipeline_diagnostic(
                flux_qe, stage1_sub, diff_cube, diff_cube_bkg_sub,
                ref_frame, aligner,
            )

        return (diff_cube_bkg_sub, aligner.shift,
                self._backgrounds[bkg_method], sep2._backgrounds[bkg_method])

    def _plot_pipeline_diagnostic(self, flux_qe, stage1_sub, diff_raw, diff_sub,
                                   ref_frame, aligner):
        """Four-panel pipeline diagnostic."""
        t = self.time
        gap_idx = np.where(np.diff(t) > 0.5)[0]

        def _ng(arr):
            a = arr.copy().astype(float)
            if len(gap_idx):
                a[gap_idx] = np.nan
            return a

        fig, axes = plt.subplots(4, 1,
                                  figsize=(1.5 * fig_width, 4.5 * fig_width),
                                  sharex=True)

        # Panel 1: QE-corrected flux spatial mean
        axes[0].plot(t, _ng(np.nanmean(flux_qe, axis=(1, 2))),
                     '.k', ms=1.5, alpha=0.5, label='QE-corrected flux')
        axes[0].set_ylabel('Mean flux (e⁻/s)', fontsize=9)
        axes[0].set_title('Stage 1: QE-corrected input', fontsize=9)

        # Panel 2: Stage 1 background-subtracted
        axes[1].plot(t, _ng(np.nanmean(stage1_sub, axis=(1, 2))),
                     '.', color='C0', ms=1.5, alpha=0.5, label='Stage 1 sub')
        axes[1].axhline(0, color='k', lw=0.7, ls='--')
        axes[1].set_ylabel('Mean flux (e⁻/s)', fontsize=9)
        axes[1].set_title('Stage 1: after background subtraction', fontsize=9)

        # Panel 3: alignment shifts
        if aligner.shift is not None:
            # aligner.shift is (T, 2): col 0 = dy_apply, col 1 = dx_apply
            # measured offsets (science - ref): dx = -shift[:,1], dy = -shift[:,0]
            dxs = _ng(-aligner.shift[:, 1])
            dys = _ng(-aligner.shift[:, 0])
            axes[2].plot(t, dxs, lw=1, label='dx')
            axes[2].plot(t, dys, lw=1, label='dy')
            axes[2].set_ylabel('Shift (px)', fontsize=9)
            axes[2].set_title('Alignment shifts', fontsize=9)
            axes[2].legend(fontsize=8)

        # Panel 4: final differenced + stage-2 subtracted
        axes[3].plot(t, _ng(np.nanmean(diff_raw, axis=(1, 2))),
                     '.', color='C1', ms=1.5, alpha=0.4, label='Diff raw')
        axes[3].plot(t, _ng(np.nanmean(diff_sub, axis=(1, 2))),
                     lw=1, color='C2', label='Diff - bkg2')
        axes[3].axhline(0, color='k', lw=0.7, ls='--')
        axes[3].set_ylabel('Mean flux (e⁻/s)', fontsize=9)
        axes[3].set_title('Stage 2: differenced - background', fontsize=9)
        axes[3].legend(fontsize=8)
        axes[3].set_xlabel('Time (MJD)', fontsize=9)

        plt.tight_layout()
        plt.show()
