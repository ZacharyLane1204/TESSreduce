import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from tessreduce.adaptive_background import (
    _make_odd,
    _get_segments,
    _block_reduce,
    _upsample_nearest,
    adaptive_medfilt_3d,
)


class TestMakeOdd(unittest.TestCase):

    def test_odd_input_unchanged(self):
        self.assertEqual(_make_odd(5), 5)
        self.assertEqual(_make_odd(11), 11)

    def test_even_input_incremented(self):
        self.assertEqual(_make_odd(4), 5)
        self.assertEqual(_make_odd(10), 11)

    def test_minimum_enforced(self):
        self.assertEqual(_make_odd(1), 3)
        self.assertEqual(_make_odd(0), 3)
        self.assertEqual(_make_odd(2), 3)

    def test_large_even(self):
        self.assertEqual(_make_odd(100), 101)


class TestGetSegments(unittest.TestCase):

    def test_no_gap_single_segment(self):
        time = np.arange(20) * 0.02  # uniform 0.02-day cadence
        segs = _get_segments(time, gap_thresh=3.0)
        self.assertEqual(segs, [(0, 20)])

    def test_one_gap_two_segments(self):
        t1 = np.arange(10) * 0.02
        t2 = np.arange(10) * 0.02 + 5.0  # 5-day gap
        time = np.concatenate([t1, t2])
        segs = _get_segments(time, gap_thresh=3.0)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (0, 10))
        self.assertEqual(segs[1], (10, 20))

    def test_two_gaps_three_segments(self):
        t1 = np.arange(8) * 0.02
        t2 = np.arange(8) * 0.02 + 5.0
        t3 = np.arange(8) * 0.02 + 10.0
        time = np.concatenate([t1, t2, t3])
        segs = _get_segments(time, gap_thresh=3.0)
        self.assertEqual(len(segs), 3)

    def test_segments_cover_full_range(self):
        time = np.concatenate([np.arange(15) * 0.02, np.arange(15) * 0.02 + 3.0])
        segs = _get_segments(time, gap_thresh=3.0)
        starts = [s for s, _ in segs]
        ends = [e for _, e in segs]
        self.assertEqual(starts[0], 0)
        self.assertEqual(ends[-1], len(time))


class TestBlockReduce(unittest.TestCase):

    def test_output_shape(self):
        arr = np.ones((10, 12, 12))
        result = _block_reduce(arr, bs=4)
        self.assertEqual(result.shape, (10, 3, 3))

    def test_uniform_array_value_preserved(self):
        arr = np.full((5, 8, 8), 7.0)
        result = _block_reduce(arr, bs=2)
        assert_allclose(result, 7.0)

    def test_dtype_preserved(self):
        arr = np.ones((4, 6, 6), dtype=np.float32)
        result = _block_reduce(arr, bs=3)
        self.assertEqual(result.dtype, np.float32)

    def test_edge_pixels_discarded(self):
        # 7 pixels with bs=3: only first 6 used → output size 2
        arr = np.ones((3, 7, 7))
        result = _block_reduce(arr, bs=3)
        self.assertEqual(result.shape, (3, 2, 2))


class TestUpsampleNearest(unittest.TestCase):

    def test_exact_multiple(self):
        arr = np.arange(12, dtype=float).reshape(3, 2, 2)
        result = _upsample_nearest(arr, X=4, Y=4, bs=2)
        self.assertEqual(result.shape, (3, 4, 4))

    def test_pads_to_target_size(self):
        arr = np.ones((2, 2, 2))
        result = _upsample_nearest(arr, X=5, Y=5, bs=2)
        self.assertEqual(result.shape, (2, 5, 5))

    def test_values_tiled_correctly(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = _upsample_nearest(arr, X=4, Y=4, bs=2)
        assert_allclose(result[0, :2, :2], 1.0)
        assert_allclose(result[0, :2, 2:4], 2.0)
        assert_allclose(result[0, 2:4, :2], 3.0)
        assert_allclose(result[0, 2:4, 2:4], 4.0)


class TestAdaptiveMedfilt3d(unittest.TestCase):

    def _make_cube(self, T=30, X=10, Y=10, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(100.0, 5.0, (T, X, Y)).astype(np.float32)

    def test_output_shapes(self):
        data = self._make_cube(T=25, X=8, Y=8)
        smoothed, windows, variability, windows_pre = adaptive_medfilt_3d(data, n_jobs=1)
        self.assertEqual(smoothed.shape, data.shape)
        self.assertEqual(windows.shape, data.shape)
        self.assertEqual(variability.shape, data.shape)
        self.assertEqual(windows_pre.shape, data.shape)

    def test_constant_cube_unchanged(self):
        data = np.full((20, 8, 8), 50.0, dtype=np.float32)
        smoothed, *_ = adaptive_medfilt_3d(data, n_jobs=1)
        assert_allclose(smoothed, 50.0, atol=1e-3)

    def test_windows_within_bounds(self):
        data = self._make_cube(T=20, X=6, Y=6)
        _, windows, _, _ = adaptive_medfilt_3d(data, w_min=3, w_max=11, n_jobs=1)
        self.assertGreaterEqual(int(windows.min()), 3)
        self.assertLessEqual(int(windows.max()), 11)

    def test_handles_nan_input(self):
        data = self._make_cube(T=20, X=6, Y=6)
        data[5, 3, 3] = np.nan
        smoothed, *_ = adaptive_medfilt_3d(data, n_jobs=1)
        self.assertEqual(smoothed.shape, data.shape)

    def test_time_array_accepted(self):
        data = self._make_cube(T=20, X=6, Y=6)
        time = np.arange(20) * 0.02
        smoothed, *_ = adaptive_medfilt_3d(data, time=time, n_jobs=1)
        self.assertEqual(smoothed.shape, data.shape)

    def test_gradient_metric(self):
        data = self._make_cube(T=20, X=6, Y=6)
        smoothed, *_ = adaptive_medfilt_3d(data, metric='gradient', n_jobs=1)
        self.assertEqual(smoothed.shape, data.shape)

    def test_with_gap_in_time(self):
        data = self._make_cube(T=30, X=6, Y=6)
        t1 = np.arange(15) * 0.02
        t2 = np.arange(15) * 0.02 + 10.0
        time = np.concatenate([t1, t2])
        smoothed, *_ = adaptive_medfilt_3d(data, time=time, n_jobs=1)
        self.assertEqual(smoothed.shape, data.shape)


if __name__ == '__main__':
    unittest.main()
