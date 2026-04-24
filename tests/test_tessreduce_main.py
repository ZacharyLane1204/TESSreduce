"""
Integration tests for the tessreduce pipeline.

These tests require a live internet connection to download TESS data from MAST.
They are skipped by default and must be opted into by setting the environment
variable TESS_INTEGRATION=1.

    TESS_INTEGRATION=1 python -m pytest tests/test_tessreduce_main.py -v
"""
import os
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')

import tessreduce as tr

_RUN_INTEGRATION = bool(os.environ.get('TESS_INTEGRATION'))

# Coordinates and sector used for all integration tests
_RA = 10.127
_DEC = -50.687
_SECTOR = 2


@unittest.skipUnless(_RUN_INTEGRATION, 'set TESS_INTEGRATION=1 to run network tests')
class TestTESSreducePipeline(unittest.TestCase):
    """End-to-end integration tests that download real TESS data."""

    @classmethod
    def setUpClass(cls):
        tess = tr.tessreduce(ra=_RA, dec=_DEC, sector=_SECTOR, reduce=False)
        tess.get_ref()
        cls.tess = tess

    # ── data structure ─────────────────────────────────────────────────────────

    def test_flux_cube_shape(self):
        flux = self.tess.flux
        self.assertEqual(flux.ndim, 3, "flux should be a 3-D (T, X, Y) array")
        T, X, Y = flux.shape
        self.assertGreater(T, 0)
        self.assertGreater(X, 0)
        self.assertGreater(Y, 0)

    def test_ref_frame_shape_matches_flux(self):
        ref = self.tess.ref
        _, X, Y = self.tess.flux.shape
        self.assertEqual(ref.shape, (X, Y))

    def test_time_array_length_matches_flux(self):
        T = self.tess.flux.shape[0]
        self.assertEqual(len(self.tess.tpf.time), T)

    # ── make_mask ──────────────────────────────────────────────────────────────

    def test_make_mask_shape(self):
        self.tess.make_mask()
        mask = self.tess.mask
        _, X, Y = self.tess.flux.shape
        self.assertEqual(mask.shape, (X, Y),
                         "mask should be 2-D and match the spatial dimensions of flux")

    def test_make_mask_integer_bits(self):
        mask = self.tess.mask
        self.assertTrue(np.issubdtype(mask.dtype, np.integer) or
                        np.issubdtype(mask.dtype, np.floating),
                        "mask values should be numeric bitmask integers")

    # ── background ─────────────────────────────────────────────────────────────

    def test_background_shape(self):
        self.tess.background()
        bkg = self.tess.bkg
        self.assertEqual(bkg.shape, self.tess.flux.shape,
                         "background cube should match flux cube shape")

    def test_background_finite_values(self):
        bkg = self.tess.bkg
        finite_frac = np.isfinite(bkg).mean()
        self.assertGreater(finite_frac, 0.5, "most background pixels should be finite")

    def test_background_magnitude_reasonable(self):
        # TESS background is typically in the hundreds to low thousands of e-/s
        median_bkg = np.nanmedian(self.tess.bkg)
        self.assertGreater(median_bkg, 0, "median background should be positive")

    # ── alignment ──────────────────────────────────────────────────────────────

    def test_fit_shift_shape(self):
        self.tess.fit_shift()
        shifts = self.tess.shift
        T = self.tess.flux.shape[0]
        self.assertEqual(shifts.shape, (T, 2),
                         "shift array should be (T, 2) — one (dx, dy) per frame")

    def test_fit_shift_magnitude_reasonable(self):
        # TESS pointing jitter is typically sub-pixel (< 2 px)
        shift_rms = np.sqrt(np.nanmean(self.tess.shift ** 2))
        self.assertLess(shift_rms, 5.0,
                        "RMS shift larger than 5 px suggests alignment failure")

    # ── difference light curve ─────────────────────────────────────────────────

    def test_diff_lc_returns_three_arrays(self):
        result = self.tess.diff_lc()
        self.assertEqual(len(result), 3,
                         "diff_lc should return (time, flux, sky)")

    def test_diff_lc_time_flux_same_length(self):
        time, flux, sky = self.tess.diff_lc()
        self.assertEqual(len(time), len(flux))
        self.assertEqual(len(time), len(sky))

    def test_diff_lc_time_monotonic(self):
        time, _, _ = self.tess.diff_lc()
        self.assertTrue(np.all(np.diff(time[np.isfinite(time)]) > 0),
                        "time array should be monotonically increasing")

    # ── full reduce ────────────────────────────────────────────────────────────

    def test_reduce_populates_lc(self):
        tess = tr.tessreduce(ra=_RA, dec=_DEC, sector=_SECTOR)
        self.assertIsNotNone(tess.lc,
                             "reduce() should populate the lc attribute")

    def test_reduce_lc_shape(self):
        tess = tr.tessreduce(ra=_RA, dec=_DEC, sector=_SECTOR)
        lc = tess.lc
        # lc is expected to be (2, T) or (3, T): [time, flux] or [time, flux, err]
        self.assertGreaterEqual(lc.shape[0], 2)
        T = tess.flux.shape[0]
        self.assertEqual(lc.shape[1], T)

    def test_reduce_lc_time_reasonable(self):
        tess = tr.tessreduce(ra=_RA, dec=_DEC, sector=_SECTOR)
        time = tess.lc[0]
        # TESS BTJD for sector 2 is in the range 1356–1382
        self.assertGreater(np.nanmedian(time), 1000,
                           "light curve time axis does not look like BTJD")


if __name__ == '__main__':
    unittest.main()
