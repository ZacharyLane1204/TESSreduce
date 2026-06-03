import numpy as np
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from copy import deepcopy


def _parallel_correlation(pixel,bkg,arr,coord,smth_time):
    """
    Calculates the Pearson r correlation coefficent between the savgol filtered lightcurve and the upper 30% of the background, at the same indices.

    Parameters:
    ----------
    pixel: ArrayLike
        The flux lightcurve to be filtered and correlated.
    bkg: ArrayLike
        The background lightcurve.

    arr: Not Used But Positional

    coord: Not Used But Positional

    smth_time: int
        The window lenght of the savgol filter, must be <= size of pixel

    Returns:
    -------
    corr: float
        The absolute value of the Pearson r correlation coefficent between the filtered lightcurve and the upper 30% of the background, rounded to 2 decimal places.
    """
    nn = np.isfinite(pixel)
    ff = savgol_filter(pixel[nn],smth_time,2)
    b = bkg[nn]
    indo = (b > np.percentile(b,70)) #& (bb < np.percentile(bb,95))
    corr = pearsonr(ff[indo],b[indo])[0]
    return np.round(abs(corr),2)

def _find_bkg_cor(tess,cores):
    """
    Takes a TESSreduce object and calculates the flux-background Pearson r correlation coefficent in parallel.

    Parameters:
    ----------
    tess: TESSreduce Object
        The TESSreduce object that is needing the correlation coefficents calculated.
    cores: int
        The number of cores to be used for parallel processing.
    
    Returns:
    cors: ArrayLike 
        The array of Pearson r correlation coefficents     

    """
    y,x = np.where(np.isfinite(tess.ref))
    coord = np.c_[y,x]
    cors = np.zeros_like(tess.ref)

    cor = Parallel(n_jobs=cores, backend="multiprocessing")(delayed(_parallel_correlation)
                                           (tess.flux[:,coord[i,0],coord[i,1]],
                                            tess.bkg[:,coord[i,0],coord[i,1]],
                                            cors,coord[i],30) for i in range(len(coord)))
    cor = np.array(cor)
    cors[coord[:,0],coord[:,1]] = cor
    return cors

def _correct_pixel_correlation(flux, bkg, bright_pct=70, max_coeff=0.1):
    """Remove the linear flux-background correlation for a single pixel.

    Fits alpha = sum(flux * bkg_centered) / sum(bkg_centered²) on the
    brightest `bright_pct`% of background frames, then subtracts
    alpha * bkg_centered from the full time series.  The correction is only
    applied when it actually reduces the Pearson |r|.

    Parameters
    ----------
    flux : array (T,)
    bkg  : array (T,)
    bright_pct : float
        Percentile threshold for "scattered-light" frames used in fitting.
    max_coeff : float
        Hard cap on |alpha| to prevent overcorrection.

    Returns
    -------
    corrected flux : array (T,)  — same as input if correction not helpful.
    """
    nn = np.isfinite(flux) & np.isfinite(bkg)
    if nn.sum() < 10:
        return flux.copy()

    f = flux[nn]
    b = bkg[nn]

    # Centre bkg on the quiet-period level so the correction is zero when
    # there is no scattered light.
    bkg_quiet = np.nanmedian(b[b < np.percentile(b, 30)])
    b_centered = b - bkg_quiet

    bright = b > np.percentile(b, bright_pct)
    if bright.sum() < 5:
        return flux.copy()

    # OLS with no intercept: flux ≈ alpha * bkg_centered
    denom = np.nansum(b_centered[bright] ** 2)
    if denom == 0:
        return flux.copy()
    alpha = np.clip(np.nansum(f[bright] * b_centered[bright]) / denom,
                    -max_coeff, max_coeff)

    corrected = flux.copy()
    corrected[nn] = f - alpha * b_centered

    # Validate: only keep if |r| on bright frames actually decreases
    r_before = abs(pearsonr(f[bright], b[bright])[0])
    r_after = abs(pearsonr(corrected[nn][bright], b[bright])[0])
    if r_after >= r_before:
        return flux.copy()

    return corrected


def multi_correlation_cor(tess, limit=0.8, cores=7):
    """Remove residual flux-background correlation for highly correlated pixels.

    For each pixel where |Pearson r| > limit (computed on the upper 30% of
    background frames), fits a bounded linear coefficient and subtracts the
    correlated component from the full flux time series.  Only pixels flagged
    as sources or saturated stars in the catalogue mask are corrected —
    background pixels have intrinsic scattered-light correlation that should
    not be removed here.

    Parameters
    ----------
    tess  : TESSreduce object
    limit : float  (default 0.8)
    cores : int    (default 7)

    Returns
    -------
    flux : array (T, NY, NX)
    bkg  : array (T, NY, NX)
    """
    cors = _find_bkg_cor(tess, cores=cores)

    # Restrict to source / saturated pixels (mask bits 1 and 2).
    # Pure background pixels have high |r| because of scattered light and
    # do not need — or benefit from — this correction.
    mask2d = tess.mask[0] if tess.mask.ndim == 3 else tess.mask
    src_pix = (mask2d & 3) > 0

    y, x = np.where((cors > limit) & src_pix)

    flux = deepcopy(tess.flux)
    bkg = deepcopy(tess.bkg)

    if len(y) == 0:
        return flux, bkg

    results = Parallel(n_jobs=cores, backend="multiprocessing")(
        delayed(_correct_pixel_correlation)(
            tess.flux[:, y[i], x[i]],
            tess.bkg[:,  y[i], x[i]],
        )
        for i in range(len(y))
    )

    for i, corrected in enumerate(results):
        flux[:, y[i], x[i]] = corrected

    return flux, bkg
    

    
