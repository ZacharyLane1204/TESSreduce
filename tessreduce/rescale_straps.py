import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
import multiprocessing
from joblib import Parallel, delayed

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def grad_clip(data,box_size=100):
    """
    Perform a local sigma clip of points based on the gradient of the points. 
    Pixels with large gradients are contaminated by stars/galaxies.

    Inputs
    ------
        data : array
            1d array of the data to clip
        box_size : int 
            integer defining the box size to clip over 
    Output
    ------
        gradind : bool

    """
    gradind = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < box_size//2:
            d = data[:i+box_size//2]
        elif len(data) - i < box_size//2:
            d = data[i-box_size//2:]
        else:
            d = data[i-box_size//2:i+box_size//2]
        
        ind = np.isfinite(d)
        d = d[ind]
        if len(d) > 5:
            gind = ~sigma_clip(np.gradient(abs(d))+d,sigma=2).mask

            if i < box_size//2:
                gradind[:i+box_size//2][ind] = gind
            elif len(data) - i < box_size//2:
                gradind[i-box_size//2:][ind] = gind
            else:
                gradind[i-box_size//2:i+box_size//2][ind] = gind
    
    gradind = gradind > 0
    return gradind 

def fit_strap(data, percentile=20):
    """
    Interpolate over stellar-contaminated rows in strap (or neighbour) columns.

    Stellar flux is strictly additive, so the true background always sits at
    the *low* end of the pixel-value distribution in a strap column.  The
    original 50th-percentile threshold discards half the data unconditionally
    and fails completely when more than half of the rows in the column are
    contaminated by stars (the median is then elevated into the stellar
    signal).  Using a lower percentile (default 20th) means contamination must
    exceed 80 % of rows before the threshold is biased into stellar flux,
    making rejection far more robust in crowded fields.

    Parameters
    ----------
    data : array_like
        1-D array of pixel values along a single column.
    percentile : float, optional
        Percentile of the finite pixel values used as the upper clip limit.
        Values at or above this level are treated as stellar contamination and
        replaced with NaN before interpolation.  Default is 20 (was 50).
    """

    x = np.arange(0,len(data))
    y = data.copy()
    p =np.ones_like(x) * np.nan
    if len(y[np.isfinite(y)]) > 10:
        lim = np.percentile(y[np.isfinite(y)], percentile)
        y[y >= lim] = np.nan

        finite = np.isfinite(y)

        if len(y[finite]) > 5:
            finite = np.isfinite(y)
            p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='linear')
            p = p(x)
    return p

def calc_strap_factor(i,breaks,size,av_size,normals,data):
    qe = np.ones_like(data) * 1. * np.nan
    b = int(breaks[i])
    size = size.astype(int)
    nind = normals[b-av_size:b]
    eind = normals[b:b+av_size]
    nind = np.append(nind,eind) + 1
    nind = nind[nind<data.shape[1]-1]
    nind = nind[nind >= 0]
    norm = fit_strap(np.nanmedian(data[:,nind],axis=1))
    for j in range(size[i]): 
        ind =  normals[b]+1+j
        if (ind > 0) & (ind < data.shape[1]):
            s1 = fit_strap(data[:,ind])
            ratio = norm/s1
            m = ~sigma_clip(ratio,sigma=2).mask
            factor = np.nanmedian(ratio[m])
            qe[:,normals[b]+1+j] = factor
    return qe

def correct_straps(Image,mask,av_size=5,parallel=True):
    data = deepcopy(Image)
    mask = deepcopy(mask)
    av_size = int(av_size)
    sind = np.where(np.nansum((mask & 4),axis=0)>0)[0]
    normals = np.where(np.nansum((mask & 4),axis=0)==0)[0]
    normals = np.append(normals,data.shape[1])
    normals = np.insert(normals,0,-1)
    breaks = np.where(np.diff(normals,append=0)>1)[0]
    breaks[breaks==-1] = 0
    size = (np.diff(normals,append=0))[np.diff(normals,append=0)>1]
    if len(breaks) > 0:
        if parallel:
            num_cores = multiprocessing.cpu_count()
            x = np.arange(0,len(breaks),dtype=int)
            qe = np.array(Parallel(n_jobs=num_cores, backend="multiprocessing")(delayed(calc_strap_factor)(i,breaks,size,av_size,normals,data) for i in x))
            qe = np.nanmedian(qe,axis=0)
            qe[np.isnan(qe)] = 1   
        else:
            qe = []
            for i in range(len(breaks)):
                qe += [calc_strap_factor(i,breaks,size,av_size,normals,data)]
            qe = np.array(qe)
            qe = np.nanmedian(qe,axis=0)
            qe[np.isnan(qe)] = 1   
    else:
        qe = np.ones_like(Image)
    return qe