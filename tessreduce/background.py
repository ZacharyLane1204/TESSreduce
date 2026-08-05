import numpy as np
import multiprocessing
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from astropy.stats import sigma_clip, sigma_clipped_stats
from joblib import Parallel, delayed

from .helpers import strip_units, Smooth_bkg


class Background():
	def __init__(self, flux, mask, buffer=3, extrapolate=True, parallel=True, backend='loky', verbose=0):

		self.flux = flux
		self.mask = mask
		self.buffer = buffer
		self.extrapolate = extrapolate
		self.parallel = parallel
		self.backend = backend
		self.verbose = verbose
		self.size = self._check_size()

		self.smooth_bkg = np.zeros_like(flux)
		self.qe = np.zeros_like(flux)
		self.bkg = np.zeros_like(flux)

	def _check_size(self):
		size = self.flux.shape[1:]
		if (size < 30).any():
			return 'small'
		else:
			return 'normal'

	def _smooth_bkg(self, index):
		"""
		Interpolate over the masked objects to derive a background estimate.

		Parameters
		----------
		data : array
			A single image

		extrapolate : bool
			Switch for using extrapolation in the background.

		Returns
		-------
		estimate : array
			An estimate of the smooth background in the TESS image.
		"""
		data = self.flux * self.mask[index]

		if (~np.isnan(data)).any():
			x = np.arange(0, data.shape[1])
			y = np.arange(0, data.shape[0])
			arr = np.ma.masked_invalid(data)
			xx, yy = np.meshgrid(x, y)
			x1 = xx[~arr.mask]
			y1 = yy[~arr.mask]
			newarr = arr[~arr.mask]
			if (len(x1) > 10) & (len(y1) > 10):
				estimate = griddata((x1, y1), newarr.ravel(),
									(xx, yy), method='linear')
				nearest = griddata((x1, y1), newarr.ravel(),
								   (xx, yy), method='nearest')
				if self.extrapolate:
					estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
				estimate = gaussian_filter(estimate, 1.5)
			else:
				estimate = np.zeros_like(data) * np.nan
		else:
			estimate = np.zeros_like(data) * np.nan

		return estimate

	def _smooth_wrapper(self):
		m = (self.mask == 0) * 1.
		m[m == 0] = np.nan

		flux = strip_units(self.flux)

		if self.parallel:
			num_cores = multiprocessing.cpu_count()
			bkg_smth = Parallel(n_jobs=num_cores, backend=self.backend, verbose=self.verbose)(
				delayed(Smooth_bkg)(frame) for frame in flux * m)
		else:
			bkg_smth = np.zeros_like(flux) * np.nan
			for i in range(flux.shape[0]):
				bkg_smth[i] = Smooth_bkg(flux[i] * m)
		self.smooth_bkg = bkg_smth

	def _strap_wrapper(self):
		strap = (self.mask == 4) * 1.0
		strap[strap == 0] = np.nan
		if len(strap.shape) == 3:
			strap = strap[self.ref_ind]
		mask = ((self.mask & 1) == 0) * 1.0
		mask[mask == 0] = np.nan

		data = strip_units(self.flux) * mask
		bkg_smth = self.smooth_bkg  # was incorrectly a bare name reference
		qes = np.zeros_like(bkg_smth) * np.nan
		for i in range(data.shape[0]):
			s = (data[i] * strap) / bkg_smth[i]
			s[s > np.percentile(s[np.isfinite(s)], 50)] = np.nan
			q = np.zeros_like(s) * np.nan
			for j in range(s.shape[1]):
				ind = ~sigma_clip(s[:, j]).mask
				q[:, j] = np.nanmedian(abs(s[ind, j]))
			q[np.isnan(q)] = 1
			qes[i] = q
		self.qe = qes
		self.bkg = bkg_smth * qes

	def _small_background(self):
		bkg = np.zeros_like(self.flux)
		flux = strip_units(self.flux)
		lim = np.percentile(flux, 10, axis=(1, 2))
		ind = flux > lim[:, np.newaxis, np.newaxis]
		flux[ind] = np.nan
		val = np.nanmedian(flux, axis=(1, 2))
		bkg[:, :, :] = val[:, np.newaxis, np.newaxis]
		self.bkg = bkg

	def calculate(self):
		"""
		Calculate the background for all frames in the TPF.
		"""
		if self.size == 'normal':
			self._smooth_wrapper()
			self._strap_wrapper()
		else:
			self._small_background()
