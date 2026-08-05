"""
Shared scene-modelling PSF photometry engine.

Catalog-agnostic, pure numpy/scipy: builds a linear design matrix (target PSF
column, optional PSF-derivative columns, optional neighbour PSF columns, and a
2D polynomial background surface) at a FIXED sub-pixel position, then solves
for flux either per-frame (calibration star fitting) or for an entire time
series in a single vectorized linear solve (forced/scene photometry).

Because position is fixed, every basis column here is frame-independent, so
the one-shot vectorized solve in `fit_scene_lightcurve` only has to invert one
small (n_columns x n_columns) matrix regardless of how many frames there are.
"""
import numpy as np
from scipy.optimize import minimize, lsq_linear

_PRF_CACHE = {}


def _prf_cache(camera, ccd, sector, col, row, prf_path, bucket=100):
    """Bucketed cache for TESS_PRF objects, keyed on a coarse pixel grid.

    Mirrors the sector-dependent `localdatadir` convention already used
    elsewhere in tessreduce (Sectors1_2_3 vs Sectors4+), and TESSELLATE's
    bucketed-cache approach: one PRF object is built per ~100x100 pixel
    region per (camera, ccd, sector, prf_path), not per star.
    """
    from PRF import TESS_PRF

    col = int(np.clip(col, 45, 2090))
    row = int(np.clip(row, 1, 2040))
    cb = int(np.clip((col // bucket) * bucket + bucket // 2, 45, 2090))
    rb = int(np.clip((row // bucket) * bucket + bucket // 2, 1, 2040))
    localdatadir = None
    if prf_path is not None:
        subdir = 'Sectors4+' if sector >= 4 else 'Sectors1_2_3'
        localdatadir = f'{prf_path}/{subdir}'
    key = (camera, ccd, sector, cb, rb, prf_path)
    prf = _PRF_CACHE.get(key)
    if prf is None:
        prf = TESS_PRF(camera, ccd, sector, cb, rb, localdatadir=localdatadir)
        _PRF_CACHE[key] = prf
    return prf


def polynomial_columns(stamp_size, order):
    """Linear 2D polynomial background basis, one flattened column per term.

    Uses the same triangular index scheme (i+j <= order) as the existing
    nonlinear `polynomial_surface` in psf_photom.py, so `poly_order` means the
    same thing in both APIs. Evaluated on a fixed, stamp-centered pixel grid
    so the columns never depend on frame. order=0 returns a single all-ones
    column (equivalent to a flat background).
    """
    cent = (stamp_size - 1) / 2.0
    yy, xx = np.mgrid[0:stamp_size, 0:stamp_size]
    xx = (xx - cent).astype(float)
    yy = (yy - cent).astype(float)

    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append(((xx ** i) * (yy ** j)).ravel())
    return cols


def prf_column(prf, cent, dx, dy, stamp_size):
    """Evaluate + normalize + flatten the PRF at a fixed sub-pixel offset."""
    npix = stamp_size * stamp_size
    p = prf.locate(cent + dx, cent + dy, (stamp_size, stamp_size))
    s = np.nansum(p)
    return (p / s).ravel() if (np.isfinite(s) and s > 0) else np.zeros(npix)


def prf_derivative_columns(prf, cent, dx, dy, stamp_size, eps=0.01):
    """Finite-difference spatial derivatives of the PRF at a fixed position.

    These absorb the sharp, PSF-shaped residual left by a subpixel-misaligned
    or imperfectly kernel-matched difference-image subtraction near a source
    (often a dipole pattern) -- something a smooth polynomial background
    cannot reproduce. Still frame-independent (position is fixed), so this
    doesn't break the vectorized one-shot solve.
    """
    dpdx = (prf_column(prf, cent, dx + eps, dy, stamp_size)
            - prf_column(prf, cent, dx - eps, dy, stamp_size)) / (2 * eps)
    dpdy = (prf_column(prf, cent, dx, dy + eps, stamp_size)
            - prf_column(prf, cent, dx, dy - eps, stamp_size)) / (2 * eps)
    return dpdx, dpdy


def build_design_matrix(prf, cent, target_dxdy, neighbour_dxdys, stamp_size,
                         poly_order=2, include_psf_derivatives=True,
                         derivatives_for='target'):
    """Stack [target PSF, target PSF-derivatives, neighbour PSFs,
    neighbour PSF-derivatives, polynomial background] into a design matrix.

    Returns (A, column_info) where column_info is a dict describing which
    column index corresponds to what, so callers can locate the target flux
    column, neighbour columns, etc.
    """
    cols = []
    info = {'target': 0, 'neighbours': [], 'derivatives': {}, 'background': []}

    cols.append(prf_column(prf, cent, target_dxdy[0], target_dxdy[1], stamp_size))

    if include_psf_derivatives:
        dpdx, dpdy = prf_derivative_columns(prf, cent, target_dxdy[0], target_dxdy[1], stamp_size)
        info['derivatives']['target'] = (len(cols), len(cols) + 1)
        cols.append(dpdx)
        cols.append(dpdy)

    neighbour_dxdys = neighbour_dxdys or []
    for dx, dy in neighbour_dxdys:
        info['neighbours'].append(len(cols))
        cols.append(prf_column(prf, cent, dx, dy, stamp_size))
        if include_psf_derivatives and derivatives_for == 'all':
            dpdx, dpdy = prf_derivative_columns(prf, cent, dx, dy, stamp_size)
            info['derivatives'][len(info['neighbours']) - 1] = (len(cols), len(cols) + 1)
            cols.append(dpdx)
            cols.append(dpdy)

    bg_cols = polynomial_columns(stamp_size, poly_order)
    info['background'] = list(range(len(cols), len(cols) + len(bg_cols)))
    cols.extend(bg_cols)

    A = np.column_stack(cols)
    return A, info


def fit_scene_position(build_A, data_pix, tol_x, tol_y, x0=(0.0, 0.0)):
    """2-D nonlinear position refinement.

    `build_A(dx, dy)` must return a design matrix for that trial position.
    The objective is the unconstrained linear least-squares residual at each
    trial position -- i.e. we search over position, but flux/background are
    always solved for in closed form at each trial (ported from TESSELLATE's
    `_scene_fit_worker` position-search step, generalized to the polynomial
    background design matrix used here).
    """
    good = np.isfinite(data_pix)

    def _chi2(p):
        dx, dy = p
        A = build_A(dx, dy)
        Ag = A[good]
        if Ag.shape[0] < Ag.shape[1] + 1:
            return np.inf
        sol, *_ = np.linalg.lstsq(Ag, data_pix[good], rcond=None)
        resid = data_pix[good] - Ag @ sol
        return float(np.nansum(resid ** 2))

    if tol_x <= 0 and tol_y <= 0:
        return 0.0, 0.0

    opt = minimize(_chi2, list(x0), method='L-BFGS-B',
                    bounds=[(-tol_x, tol_x), (-tol_y, tol_y)])
    return float(opt.x[0]), float(opt.x[1])


def fit_scene_frame(A, data_pix, flux_bounds=None):
    """Single-frame scene solve.

    Bounded (`scipy.optimize.lsq_linear`) when `flux_bounds` is given
    (calibration use -- constrain each source's flux to a catalog-predicted
    range), otherwise a plain unconstrained `np.linalg.lstsq`. Also returns
    the formal parameter covariance `s2 * inv(A.T@A)` for per-star error
    propagation (ported from TESSELLATE's `_scene_fit_worker`).

    Returns (coeffs, cov, dof).
    """
    good = np.isfinite(data_pix) & np.all(np.isfinite(A), axis=1)
    Ag = A[good]
    dg = data_pix[good]

    if flux_bounds is not None:
        res = lsq_linear(Ag, dg, bounds=flux_bounds, method='trf', max_iter=200)
        coeffs = res.x
    else:
        coeffs, *_ = np.linalg.lstsq(Ag, dg, rcond=None)

    resid = dg - Ag @ coeffs
    dof = max(Ag.shape[0] - Ag.shape[1], 1)
    s2 = float(np.nansum(resid ** 2) / dof)
    cov = s2 * np.linalg.inv(Ag.T @ Ag)
    return coeffs, cov, dof


def fit_scene_lightcurve(stamp_cube, A, prior_flux=None, prior_strength=None):
    """Vectorized one-shot scene solve across an entire time series.

    Because `A` is frame-independent (position fixed), the whole time series
    is solved in a single matrix product rather than one nonlinear fit per
    frame. Optionally accepts a Tikhonov/ridge prior (`prior_flux`,
    `prior_strength`, both length-M arrays with zeros on unregularized
    columns) which pulls specific columns (e.g. crowded neighbours) toward a
    catalog-predicted value without breaking the closed-form vectorized
    solve -- unlike a hard per-frame bound, which would require iterating.

    Returns (flux, e_flux, background, coeffs) where `flux`/`e_flux` are the
    target (column 0) flux and its formal per-frame error, and `background`
    is the fitted per-frame amplitude of the first background column (the
    constant term).
    """
    stamp_cube = np.asarray(stamp_cube, dtype=float)
    nfr = stamp_cube.shape[0]
    npix = stamp_cube.shape[1] * stamp_cube.shape[2]
    D = stamp_cube.reshape(nfr, npix)

    good = np.all(np.isfinite(D), axis=0) & np.all(np.isfinite(A), axis=1)
    if good.sum() < A.shape[1] + 1:
        raise RuntimeError('Too few finite pixels for scene photometry.')
    Ag = A[good]
    Dg = D[:, good]

    ATA = Ag.T @ Ag
    if prior_strength is not None:
        prior_strength = np.asarray(prior_strength, dtype=float)
        prior_flux = np.asarray(prior_flux, dtype=float)
        # a single non-finite entry here would silently NaN-poison every
        # column of the solve (one bad row spreads through the matrix
        # inverse) -- fail loudly instead. Callers should zero out
        # prior_strength for any column with an unknown/non-finite prior.
        if not (np.all(np.isfinite(prior_strength)) and
                np.all(np.isfinite(prior_flux[prior_strength != 0]))):
            raise ValueError('fit_scene_lightcurve: prior_flux/prior_strength must be finite '
                              'wherever prior_strength is nonzero.')
        ATA = ATA + np.diag(prior_strength)
        # prior_flux may be non-finite wherever prior_strength is exactly 0
        # (no prior for that column) -- np.where avoids 0*nan=nan poisoning
        # the RHS despite the zero weight.
        pull = np.where(prior_strength != 0, prior_strength * prior_flux, 0.0)
        rhs = Ag.T @ Dg.T + pull[:, None]
    else:
        rhs = Ag.T @ Dg.T

    ATA_inv = np.linalg.inv(ATA)
    coeffs = ATA_inv @ rhs          # (M, n_frames)

    flux = coeffs[0]
    resid = Dg.T - Ag @ coeffs
    med = np.median(resid, axis=0)
    sigma = 1.4826 * np.median(np.abs(resid - med), axis=0)
    e_flux = sigma * np.sqrt(ATA_inv[0, 0])

    return flux, e_flux, coeffs
