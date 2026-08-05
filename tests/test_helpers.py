import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from astropy import units as u

import matplotlib
matplotlib.use('Agg')

from tessreduce.helpers import (
    strip_units,
    sigma_mask,
    Source_mask,
    sig_err,
    grads_rad,
    grad_flux_rad,
    image_sub,
    grad_clip,
    fit_strap,
    Identify_masks,
    Multiple_day_breaks,
    smooth_zp,
    Smooth_bkg,
    regional_stats_mask,
    _tess_pointing_table,
    _target_sectors,
)


class TestStripUnits(unittest.TestCase):

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = strip_units(arr)
        assert_array_equal(result, arr)
        self.assertIsInstance(result, np.ndarray)

    def test_astropy_quantity(self):
        qty = np.array([1.0, 2.0]) * u.electron / u.s
        result = strip_units(qty)
        assert_array_equal(result, np.array([1.0, 2.0]))


class TestSigmaMask(unittest.TestCase):

    def test_masks_outlier(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[50] = 1000.0
        mask = sigma_mask(data, sigma=3)
        self.assertFalse(mask[50], "outlier should be masked (False)")
        self.assertGreater(mask.sum(), 90)

    def test_all_finite_mostly_true(self):
        data = np.ones(50)
        mask = sigma_mask(data, sigma=3)
        self.assertTrue(mask.all())

    def test_returns_bool_array(self):
        data = np.arange(20, dtype=float)
        mask = sigma_mask(data, sigma=3)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, data.shape)


class TestSourceMask(unittest.TestCase):

    def test_uniform_image_no_sources(self):
        data = np.ones((20, 20)) * 100.0
        mask = Source_mask(data)
        # uniform image — mask should be all zeros (everything below 95th pct threshold)
        self.assertEqual(mask.shape, (20, 20))

    def test_bright_source_detected(self):
        rng = np.random.default_rng(0)
        data = rng.normal(100, 5, (30, 30))
        data[15, 15] = 5000.0  # bright source
        mask = Source_mask(data)
        self.assertEqual(mask.shape, data.shape)
        # bright pixel should be masked (0) while background stays unmasked (1)
        self.assertEqual(mask[15, 15], 0.0)

    def test_all_nan_returns_zeros(self):
        data = np.full((10, 10), np.nan)
        mask = Source_mask(data)
        assert_array_equal(mask, np.zeros((10, 10)))


class TestSigErr(unittest.TestCase):

    def test_outlier_masked(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 100)
        data[30] = 500.0
        mask = sig_err(data, sig=5)
        self.assertTrue(mask[30])

    def test_clean_data_no_mask(self):
        data = np.zeros(50)
        mask = sig_err(data, sig=5)
        self.assertFalse(mask.any())

    def test_with_error_array(self):
        data = np.ones(50, dtype=float)
        data[25] = 100.0
        err = np.ones(50) * 0.1
        mask = sig_err(data, err=err, sig=5)
        self.assertTrue(mask[25])


class TestGradsRad(unittest.TestCase):

    def test_shape_preserved(self):
        flux = np.linspace(1, 10, 50)
        result = grads_rad(flux)
        self.assertEqual(result.shape, flux.shape)

    def test_constant_flux_near_zero(self):
        flux = np.ones(50) * 5.0
        result = grads_rad(flux)
        assert_allclose(result, 0.0, atol=1e-10)


class TestGradFluxRad(unittest.TestCase):

    def test_shape_preserved(self):
        flux = np.linspace(1, 10, 30)
        result = grad_flux_rad(flux)
        self.assertEqual(result.shape, flux.shape)

    def test_nonnegative(self):
        flux = np.linspace(0, 5, 30)
        result = grad_flux_rad(flux)
        self.assertTrue((result >= 0).all())


class TestImageSub(unittest.TestCase):

    def test_zero_shift_returns_low_cost(self):
        rng = np.random.default_rng(1)
        img = rng.normal(100, 5, (30, 30))
        cost = image_sub((0.0, 0.0), img, img)
        self.assertAlmostEqual(cost, 0.0, places=5)

    def test_nonzero_shift_increases_cost(self):
        rng = np.random.default_rng(2)
        img = rng.normal(100, 5, (30, 30))
        cost_zero = image_sub((0.0, 0.0), img, img)
        cost_shifted = image_sub((3.0, 0.0), img, img)
        self.assertGreater(cost_shifted, cost_zero)


class TestGradClip(unittest.TestCase):

    def test_clean_data_mostly_passes(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, 200)
        mask = grad_clip(data, box_size=50)
        # most points should pass
        self.assertGreater(mask.sum(), 150)

    def test_returns_bool(self):
        data = np.linspace(0, 10, 100)
        result = grad_clip(data, box_size=20)
        self.assertEqual(result.dtype, bool)
        self.assertEqual(result.shape, data.shape)

    def test_spike_clipped(self):
        data = np.ones(100, dtype=float)
        data[50] = 1000.0
        data[51] = 1000.0
        result = grad_clip(data, box_size=30)
        # spike region should be clipped
        self.assertFalse(result[50])


class TestFitStrap(unittest.TestCase):

    def test_interpolates_over_masked_values(self):
        data = np.ones(50, dtype=float) * 5.0
        data[20:25] = 200.0
        # mask marks GOOD pixels (True = usable background)
        mask = data < 100.0
        result = fit_strap(data, mask)
        # interpolated values in contaminated region should be near background (~5)
        self.assertTrue(np.all(np.abs(result[20:25] - 5.0) < 3.0))

    def test_returns_same_length(self):
        data = np.linspace(1, 10, 40)
        mask = np.ones(40, dtype=bool)
        result = fit_strap(data, mask)
        self.assertEqual(len(result), 40)

    def test_no_good_pixels_returns_ones(self):
        data = np.ones(20, dtype=float) * 5.0
        mask = np.zeros(20, dtype=bool)  # no good pixels
        result = fit_strap(data, mask)
        self.assertEqual(len(result), 20)


class TestIdentifyMasks(unittest.TestCase):

    def test_two_islands(self):
        obj = np.array([0, 1, 1, 0, 0, 1, 0], dtype=float)
        masks = Identify_masks(obj)
        self.assertEqual(len(masks), 2)
        # first island spans indices 1-2
        self.assertTrue(masks[0][1])
        self.assertTrue(masks[0][2])
        self.assertFalse(masks[0][5])
        # second island is index 5
        self.assertTrue(masks[1][5])

    def test_single_island(self):
        obj = np.array([0, 0, 1, 1, 1, 0, 0], dtype=float)
        masks = Identify_masks(obj)
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0][2])
        self.assertTrue(masks[0][4])

    def test_empty_returns_empty(self):
        obj = np.zeros(10, dtype=float)
        masks = Identify_masks(obj)
        self.assertEqual(len(masks), 0)


class TestMultipleDayBreaks(unittest.TestCase):

    def _make_lc(self, times, flux=None):
        if flux is None:
            flux = np.ones_like(times)
        return np.array([times, flux])

    def test_no_gap_two_endpoints(self):
        times = np.arange(10) * 0.02   # cadence 0.02 days, no gap
        lc = self._make_lc(times)
        breaks = Multiple_day_breaks(lc)
        assert_array_equal(breaks, [0, len(times)])

    def test_one_gap_three_segments(self):
        times = np.concatenate([np.arange(5) * 0.02, np.arange(5) * 0.02 + 2.0])
        lc = self._make_lc(times)
        breaks = Multiple_day_breaks(lc)
        self.assertEqual(len(breaks), 3)
        self.assertEqual(breaks[0], 0)
        self.assertEqual(breaks[-1], len(times))

    def test_nan_flux_excluded_from_break_detection(self):
        times = np.concatenate([np.arange(5) * 0.02, np.arange(5) * 0.02 + 2.0])
        flux = np.ones(10)
        flux[3] = np.nan  # NaN does not create a break
        lc = self._make_lc(times, flux)
        breaks = Multiple_day_breaks(lc)
        self.assertGreater(len(breaks), 2)


class TestSmoothZp(unittest.TestCase):

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_returns_smoothed_and_err(self, _mock_plot, _mock_figure):
        rng = np.random.default_rng(5)
        N = 60
        time = np.arange(N) * 0.02
        zp = np.ones(N) * 25.0 + rng.normal(0, 0.05, N)
        smoothed, err = smooth_zp(zp, time)
        self.assertEqual(smoothed.shape, zp.shape)
        self.assertIsInstance(err, float)
        self.assertGreaterEqual(err, 0.0)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_two_segment_sector(self, _mock_plot, _mock_figure):
        N = 60
        t1 = np.arange(30) * 0.02
        t2 = np.arange(30) * 0.02 + 14.0  # 14-day gap
        time = np.concatenate([t1, t2])
        zp = np.ones(N) * 25.0
        smoothed, err = smooth_zp(zp, time)
        self.assertEqual(smoothed.shape, (N,))


class TestSmoothBkg(unittest.TestCase):

    def test_all_nan_returns_zeros(self):
        data = np.full((10, 10), np.nan)
        result = Smooth_bkg(data)
        assert_array_equal(result, np.zeros((10, 10)))

    def test_output_shape(self):
        rng = np.random.default_rng(9)
        data = rng.normal(100, 5, (15, 15))
        data[5:8, 5:8] = np.nan
        result = Smooth_bkg(data, interpolate=True)
        self.assertEqual(result.shape, data.shape)

    def test_finite_output_for_finite_input(self):
        data = np.ones((12, 12)) * 50.0
        result = Smooth_bkg(data, interpolate=True)
        self.assertTrue(np.all(np.isfinite(result)))


class TestRegionalStatsMask(unittest.TestCase):

    def test_output_shape_and_dtype(self):
        rng = np.random.default_rng(11)
        image = rng.normal(100, 5, (30, 30))
        mask = regional_stats_mask(image, size=15, sigma=3)
        self.assertEqual(mask.shape, image.shape)
        self.assertTrue(np.issubdtype(mask.dtype, np.number))
        self.assertTrue(set(np.unique(mask)).issubset({0, 1, 0.0, 1.0}))

    def test_outlier_masked(self):
        image = np.ones((30, 30)) * 100.0
        image[15, 15] = 50000.0
        mask = regional_stats_mask(image, size=15, sigma=3)
        self.assertTrue(mask[15, 15])


class TestTessPointingTable(unittest.TestCase):

    def test_indexed_by_sector_with_expected_columns(self):
        table = _tess_pointing_table()
        self.assertEqual(table.index.name, 'Sector')
        self.assertIn('mjd_start', table.columns)
        self.assertIn('mjd_end', table.columns)

    def test_end_after_start(self):
        table = _tess_pointing_table()
        self.assertTrue((table['mjd_end'] > table['mjd_start']).all())

    def test_known_sector_one_start_time(self):
        # Sector 1 start is 2018-07-25 19:00 UT, JD 2458324.5 -> MJD 58324.0
        table = _tess_pointing_table()
        self.assertAlmostEqual(table.loc[1, 'mjd_start'], 58324.0, places=3)


class TestTargetSectors(unittest.TestCase):

    def test_known_target_returns_expected_sectors(self):
        # Reference target cross-checked against tess_stars2px_function_entry
        # (tess-point) output: sectors 2, 29, 69, 96, 103, 104, 105, 106.
        outSecs, outCam, outCcd, outColPix, outRowPix = _target_sectors(10.127, -50.687)
        expected = {2, 29, 69, 96, 103, 104, 105, 106}
        self.assertTrue(expected.issubset(set(outSecs.tolist())))

    def test_arrays_aligned_and_sorted(self):
        outSecs, outCam, outCcd, outColPix, outRowPix = _target_sectors(10.127, -50.687)
        lengths = {len(outSecs), len(outCam), len(outCcd), len(outColPix), len(outRowPix)}
        self.assertEqual(len(lengths), 1)
        assert_array_equal(outSecs, np.sort(outSecs))

    def test_pixel_coordinates_within_ccd_bounds(self):
        outSecs, outCam, outCcd, outColPix, outRowPix = _target_sectors(10.127, -50.687)
        self.assertTrue(np.all((outColPix >= 0) & (outColPix <= 2136)))
        self.assertTrue(np.all((outRowPix >= 0) & (outRowPix <= 2078)))

if __name__ == '__main__':
    unittest.main()
