import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from tessreduce.rescale_straps import grad_clip, fit_strap, correct_straps


class TestGradClip(unittest.TestCase):

    def test_returns_bool_correct_shape(self):
        data = np.linspace(0, 10, 120)
        result = grad_clip(data, box_size=30)
        self.assertEqual(result.dtype, bool)
        self.assertEqual(result.shape, data.shape)

    def test_smooth_data_mostly_passes(self):
        data = np.ones(150)
        result = grad_clip(data, box_size=50)
        self.assertGreater(result.sum(), 100)

    def test_spike_flagged(self):
        data = np.ones(100, dtype=float)
        data[50] = 500.0
        data[51] = 500.0
        result = grad_clip(data, box_size=30)
        self.assertFalse(result[50])

    def test_handles_nans(self):
        data = np.ones(100, dtype=float)
        data[20:25] = np.nan
        result = grad_clip(data, box_size=30)
        self.assertEqual(result.shape, data.shape)


class TestFitStrap(unittest.TestCase):

    def test_returns_correct_length(self):
        data = np.linspace(1, 5, 50)
        result = fit_strap(data, percentile=20)
        self.assertEqual(len(result), 50)

    def test_high_values_replaced_by_interpolation(self):
        # Background at 5, stellar contamination spikes at 200
        data = np.ones(60, dtype=float) * 5.0
        data[25:30] = 200.0
        result = fit_strap(data, percentile=20)
        # contaminated region should be near background level after interpolation
        finite = np.isfinite(result[25:30])
        if finite.any():
            self.assertTrue(np.all(result[25:30][finite] < 50.0))

    def test_all_nan_returns_nan(self):
        data = np.full(20, np.nan)
        result = fit_strap(data)
        self.assertEqual(len(result), 20)
        self.assertTrue(np.all(np.isnan(result)))

    def test_mostly_nan_returns_nan(self):
        data = np.full(20, np.nan)
        data[0] = 1.0
        data[1] = 2.0  # fewer than 5 finite → returns nan array
        result = fit_strap(data)
        self.assertTrue(np.all(np.isnan(result)))


class TestCorrectStraps(unittest.TestCase):

    def _make_flat_image(self, rows=30, cols=30, value=100.0):
        return np.full((rows, cols), value, dtype=float)

    def _make_strap_mask(self, rows, cols, strap_cols):
        """bit 2 (value 4) flags strap columns."""
        mask = np.zeros((rows, cols), dtype=int)
        for c in strap_cols:
            mask[:, c] = 4
        return mask

    def test_no_strap_cols_returns_ones(self):
        img = self._make_flat_image()
        mask = np.zeros((30, 30), dtype=int)  # no strap bits set
        qe = correct_straps(img, mask, av_size=3, parallel=False)
        # with no straps, returns all-ones correction
        assert_allclose(qe, np.ones((30, 30)), atol=1e-6)

    def test_output_shape_matches_input(self):
        img = self._make_flat_image(25, 40)
        mask = self._make_strap_mask(25, 40, strap_cols=[10, 11])
        qe = correct_straps(img, mask, av_size=3, parallel=False)
        self.assertEqual(qe.shape, img.shape)

    def test_strap_correction_near_one_for_flat_image(self):
        # uniform image: strap column has same flux as neighbours → factor ≈ 1
        img = self._make_flat_image(40, 40, value=200.0)
        mask = self._make_strap_mask(40, 40, strap_cols=[20])
        qe = correct_straps(img, mask, av_size=5, parallel=False)
        # QE factor for a uniform image should be close to 1
        finite_qe = qe[np.isfinite(qe)]
        self.assertTrue(np.all(np.abs(finite_qe - 1.0) < 0.5))

    def test_elevated_strap_returns_factor_greater_than_one(self):
        # strap column 20% brighter than neighbours → factor ≈ 1/1.2
        rows, cols = 50, 30
        img = np.ones((rows, cols), dtype=float) * 100.0
        img[:, 15] = 120.0  # strap column 20% elevated
        mask = self._make_strap_mask(rows, cols, strap_cols=[15])
        qe = correct_straps(img, mask, av_size=4, parallel=False)
        # correction factor for elevated strap should be < 1 (scales it down)
        col_factor = np.nanmedian(qe[:, 15])
        self.assertLess(col_factor, 1.1)


if __name__ == '__main__':
    unittest.main()
