import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from tessreduce.cat_mask import (
    size_limit,
    circle_app,
    ps1_auto_mask,
    gaia_auto_mask,
    Big_sat,
    detect_straps_empirical,
    Strap_mask,
)


def _make_catalog(xs, ys, mags):
    """Build a minimal source catalog DataFrame."""
    return pd.DataFrame({'x': xs, 'y': ys, 'mag': mags})


class TestSizeLimit(unittest.TestCase):

    def test_inside_pixels_true(self):
        image = np.zeros((20, 20))
        x = np.array([5, 10, 15])
        y = np.array([5, 10, 15])
        ind = size_limit(x, y, image)
        self.assertTrue(ind.all())

    def test_boundary_pixels_false(self):
        image = np.zeros((20, 20))
        x = np.array([0, 19, 10])
        y = np.array([10, 10, 0])
        ind = size_limit(x, y, image)
        # x=0 and x=19 are at the edges, should be excluded
        self.assertFalse(ind[0])
        self.assertFalse(ind[1])

    def test_mixed(self):
        image = np.zeros((10, 10))
        x = np.array([5, 0, 9])
        y = np.array([5, 5, 5])
        ind = size_limit(x, y, image)
        self.assertTrue(ind[0])
        self.assertFalse(ind[1])
        self.assertFalse(ind[2])


class TestCircleApp(unittest.TestCase):

    def test_output_shape(self):
        mask = circle_app(5)
        # shape should be roughly (2*rad+1, 2*rad+1)
        self.assertEqual(mask.shape[0], mask.shape[1])
        self.assertGreater(mask.shape[0], 8)

    def test_center_is_set(self):
        rad = 4
        mask = circle_app(rad)
        cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
        self.assertEqual(mask[cy, cx], 1)

    def test_corners_zero(self):
        mask = circle_app(5)
        # corners of a circular aperture should be 0
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[-1, -1], 0)

    def test_binary_values(self):
        mask = circle_app(3)
        unique = np.unique(mask)
        self.assertTrue(set(unique).issubset({0, 1}))


class TestPs1AutoMask(unittest.TestCase):

    def _make_image(self, size=30):
        return np.zeros((size, size))

    def test_returns_dict_with_all_key(self):
        image = self._make_image()
        cat = _make_catalog(xs=[10, 15], ys=[10, 15], mags=[17.5, 16.5])
        masks = ps1_auto_mask(cat, image)
        self.assertIn('all', masks)

    def test_output_shape_matches_image(self):
        image = self._make_image(25)
        cat = _make_catalog(xs=[8, 12], ys=[8, 12], mags=[15.5, 14.0])
        masks = ps1_auto_mask(cat, image)
        self.assertEqual(masks['all'].shape, image.shape)

    def test_source_position_masked(self):
        image = self._make_image(40)
        cat = _make_catalog(xs=[20], ys=[20], mags=[16.5])
        masks = ps1_auto_mask(cat, image)
        # source at (20,20) should be masked in the 'all' layer
        self.assertGreater(masks['all'][20, 20], 0)

    def test_empty_catalog_all_zeros(self):
        image = self._make_image()
        cat = _make_catalog(xs=[], ys=[], mags=[])
        masks = ps1_auto_mask(cat, image)
        assert_array_equal(masks['all'], np.zeros_like(image))

    def test_out_of_bounds_sources_ignored(self):
        image = self._make_image(20)
        # sources outside image boundaries
        cat = _make_catalog(xs=[100, -5], ys=[100, -5], mags=[16.0, 17.0])
        masks = ps1_auto_mask(cat, image)
        assert_array_equal(masks['all'], np.zeros_like(image))


class TestGaiaAutoMask(unittest.TestCase):

    def _make_image(self, size=30):
        return np.zeros((size, size))

    def test_returns_dict_with_all_key(self):
        image = self._make_image()
        cat = _make_catalog(xs=[10], ys=[10], mags=[14.0])
        masks = gaia_auto_mask(cat, image)
        self.assertIn('all', masks)

    def test_output_shape_matches_image(self):
        image = self._make_image(35)
        cat = _make_catalog(xs=[15, 20], ys=[15, 20], mags=[13.0, 15.5])
        masks = gaia_auto_mask(cat, image)
        self.assertEqual(masks['all'].shape, image.shape)

    def test_bright_source_masked(self):
        image = self._make_image(50)
        cat = _make_catalog(xs=[25], ys=[25], mags=[9.0])
        masks = gaia_auto_mask(cat, image)
        self.assertGreater(masks['all'][25, 25], 0)

    def test_empty_catalog(self):
        image = self._make_image()
        cat = _make_catalog(xs=[], ys=[], mags=[])
        masks = gaia_auto_mask(cat, image)
        assert_array_equal(masks['all'], np.zeros_like(image))


class TestBigSat(unittest.TestCase):

    def test_no_saturated_stars_returns_empty(self):
        image = np.zeros((50, 50))
        # all stars fainter than magnitude 7 threshold
        cat = _make_catalog(xs=[20, 30], ys=[20, 30], mags=[10.0, 12.0])
        result = Big_sat(cat, image)
        self.assertEqual(len(result), 0)

    def test_saturated_star_produces_mask(self):
        image = np.zeros((80, 80))
        cat = _make_catalog(xs=[40], ys=[40], mags=[6.0])
        result = Big_sat(cat, image)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0].shape, image.shape)
        # center of the saturated star should be masked
        self.assertGreater(result[0][40, 40], 0)


class TestDetectStrapsEmpirical(unittest.TestCase):

    def _make_uniform_cube(self, T=50, X=20, Y=30, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(100.0, 2.0, (T, X, Y))

    def test_uniform_flux_no_straps(self):
        cube = self._make_uniform_cube()
        cols = detect_straps_empirical(cube, min_snr=5.0)
        # pure noise should not trigger detections at high SNR
        self.assertEqual(len(cols), 0)

    def test_elevated_column_detected(self):
        rng = np.random.default_rng(1)
        cube = rng.normal(100.0, 1.0, (60, 15, 25))
        # column 12 is 30% elevated in bright frames
        order = np.argsort(cube.mean(axis=(1, 2)))
        bright = order[-12:]
        cube[np.ix_(bright, np.arange(15), [12])] *= 1.3
        cols = detect_straps_empirical(cube, min_snr=3.0)
        self.assertIn(12, cols)

    def test_output_dtype_int(self):
        cube = self._make_uniform_cube()
        cols = detect_straps_empirical(cube)
        self.assertTrue(np.issubdtype(cols.dtype, np.integer))

    def test_detected_cols_within_bounds(self):
        cube = self._make_uniform_cube(Y=20)
        cols = detect_straps_empirical(cube, min_snr=1.0)
        self.assertTrue(np.all(cols >= 0))
        self.assertTrue(np.all(cols < 20))


class TestStrapMask(unittest.TestCase):

    def test_known_strap_column_masked(self):
        image = np.zeros((30, 40))
        col = [15]
        mask = Strap_mask(image, col, size=4)
        self.assertEqual(mask.shape, image.shape)
        # column 15 and neighbours should be in the strap mask (bit 2 = value 4)
        self.assertTrue(np.any(mask[:, 15] > 0))

    def test_empty_strap_list_no_mask(self):
        image = np.zeros((20, 20))
        mask = Strap_mask(image, col=[], size=4)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.sum(), 0)

    def test_output_shape_matches_image(self):
        image = np.zeros((25, 35))
        mask = Strap_mask(image, col=[10, 20], size=3)
        self.assertEqual(mask.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
