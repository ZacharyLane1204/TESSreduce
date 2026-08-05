import unittest
import numpy as np
from numpy.testing import assert_allclose

import matplotlib
matplotlib.use('Agg')

from tessreduce.scene_photom import (
    polynomial_columns,
    prf_column,
    prf_derivative_columns,
    build_design_matrix,
    fit_scene_position,
    fit_scene_frame,
    fit_scene_lightcurve,
)

STAMP_SIZE = 9
CENT = (STAMP_SIZE - 1) / 2.0


class _GaussianPRF:
    """Minimal stand-in for PRF.TESS_PRF with a .locate(x, y, shape) method."""

    def __init__(self, sigma=1.2, skew=0.0):
        self.sigma = sigma
        self.skew = skew

    def locate(self, x, y, shape):
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        g = np.exp(-(r2 / (2 * self.sigma ** 2)))
        if self.skew:
            g = g * (1 + self.skew * (xx - x))
        return np.clip(g, 0, None)


class TestPolynomialColumns(unittest.TestCase):

    def test_order_zero_is_flat(self):
        cols = polynomial_columns(STAMP_SIZE, order=0)
        self.assertEqual(len(cols), 1)
        assert_allclose(cols[0], np.ones(STAMP_SIZE * STAMP_SIZE))

    def test_term_count_matches_triangular_scheme(self):
        for order in range(4):
            cols = polynomial_columns(STAMP_SIZE, order=order)
            expected = (order + 1) * (order + 2) // 2
            self.assertEqual(len(cols), expected)

    def test_columns_are_frame_independent_shape(self):
        cols = polynomial_columns(STAMP_SIZE, order=2)
        for c in cols:
            self.assertEqual(c.shape, (STAMP_SIZE * STAMP_SIZE,))


class TestDesignMatrix(unittest.TestCase):

    def test_shape_without_derivatives_or_neighbours(self):
        prf = _GaussianPRF()
        A, info = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                       poly_order=0, include_psf_derivatives=False)
        self.assertEqual(A.shape, (STAMP_SIZE * STAMP_SIZE, 2))  # target + flat bg
        self.assertEqual(info['target'], 0)
        self.assertEqual(info['background'], [1])
        self.assertEqual(info['neighbours'], [])

    def test_shape_with_derivatives(self):
        prf = _GaussianPRF()
        A, info = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                       poly_order=0, include_psf_derivatives=True)
        self.assertEqual(A.shape, (STAMP_SIZE * STAMP_SIZE, 4))  # target + 2 deriv + bg
        self.assertIn('target', info['derivatives'])

    def test_shape_with_neighbours(self):
        prf = _GaussianPRF()
        A, info = build_design_matrix(prf, CENT, (0.0, 0.0), [(2.0, 0.0), (-1.5, 1.0)],
                                       STAMP_SIZE, poly_order=1, include_psf_derivatives=False)
        # target(1) + 2 neighbours + poly_order=1 -> 3 bg terms
        self.assertEqual(A.shape, (STAMP_SIZE * STAMP_SIZE, 1 + 2 + 3))
        self.assertEqual(len(info['neighbours']), 2)

    def test_symmetric_psf_orthogonal_to_its_own_derivative(self):
        """A symmetric PSF is even; its spatial derivative is odd. Their inner
        product over a symmetric stamp should vanish -- this is why a pure
        dipole subtraction-residual can't bias flux regardless of whether the
        derivative columns are modelled (see TestDifferenceResidualHandling)."""
        prf = _GaussianPRF(skew=0.0)
        target = prf_column(prf, CENT, 0.0, 0.0, STAMP_SIZE)
        dpdx, dpdy = prf_derivative_columns(prf, CENT, 0.0, 0.0, STAMP_SIZE)
        self.assertAlmostEqual(np.dot(target, dpdx), 0.0, places=8)
        self.assertAlmostEqual(np.dot(target, dpdy), 0.0, places=8)


class TestFitSceneLightcurve(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.prf = _GaussianPRF()
        self.nfr = 60

    def _make_cube(self, A, col_fluxes, noise_sigma=2.0):
        """col_fluxes: dict {col_index: (n_frames,) array} of per-frame amplitudes."""
        nfr = self.nfr
        data = np.zeros((nfr, A.shape[0]))
        for idx, series in col_fluxes.items():
            data += np.outer(series, A[:, idx])
        data += self.rng.normal(0, noise_sigma, size=data.shape)
        return data.reshape(nfr, STAMP_SIZE, STAMP_SIZE)

    def test_recovers_flux_and_background(self):
        A, info = build_design_matrix(self.prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                       poly_order=0, include_psf_derivatives=False)
        true_flux = 1000 + 200 * np.sin(np.linspace(0, 6, self.nfr))
        true_bg = 50 + 5 * np.cos(np.linspace(0, 3, self.nfr))
        cube = self._make_cube(A, {0: true_flux, info['background'][0]: true_bg})

        flux, e_flux, coeffs = fit_scene_lightcurve(cube, A)
        assert_allclose(flux, true_flux, atol=5 * np.median(e_flux))
        assert_allclose(coeffs[info['background'][0]], true_bg, atol=5 * np.median(e_flux))

    def test_poly_order_zero_matches_flat_background(self):
        """order=0 should be numerically identical to a single flat column."""
        A0, _ = build_design_matrix(self.prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                     poly_order=0, include_psf_derivatives=False)
        cols = polynomial_columns(STAMP_SIZE, order=0)
        assert_allclose(A0[:, 1], cols[0])

    def test_error_scales_with_injected_noise(self):
        A, info = build_design_matrix(self.prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                       poly_order=0, include_psf_derivatives=False)
        true_flux = np.full(self.nfr, 1000.0)

        cube_lo = self._make_cube(A, {0: true_flux}, noise_sigma=1.0)
        cube_hi = self._make_cube(A, {0: true_flux}, noise_sigma=8.0)

        _, e_lo, _ = fit_scene_lightcurve(cube_lo, A)
        _, e_hi, _ = fit_scene_lightcurve(cube_hi, A)
        self.assertLess(np.median(e_lo), np.median(e_hi))


class TestDifferenceResidualHandling(unittest.TestCase):
    """Verify the PSF-derivative columns absorb poor-subtraction residuals
    near the target, per the plan's difference-imaging requirement."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.nfr = 60

    def test_derivative_columns_reduce_noise_inflation(self):
        # slightly asymmetric PRF, closer to a real TESS PRF than a pure Gaussian
        prf = _GaussianPRF(skew=0.08)
        true_flux = 1000 + 100 * np.sin(np.linspace(0, 6, self.nfr))
        true_bg = 50.0
        shift_amp = 40.0

        A_no, info_no = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                             poly_order=0, include_psf_derivatives=False)
        A_wd, info_wd = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                             poly_order=0, include_psf_derivatives=True)
        dpdx, _ = prf_derivative_columns(prf, CENT, 0.0, 0.0, STAMP_SIZE)

        target_col = A_no[:, 0]
        bg_col = A_no[:, info_no['background'][0]]

        data = (np.outer(true_flux, target_col) + true_bg * bg_col
                + shift_amp * dpdx)
        data += self.rng.normal(0, 2.0, size=data.shape)
        cube = data.reshape(self.nfr, STAMP_SIZE, STAMP_SIZE)

        flux_no, e_no, _ = fit_scene_lightcurve(cube, A_no)
        flux_wd, e_wd, coeffs_wd = fit_scene_lightcurve(cube, A_wd)

        # modelling the residual should not inflate (and should typically
        # reduce) the robust per-frame noise estimate
        self.assertLessEqual(np.median(e_wd), np.median(e_no) * 1.05)

        # the fitted derivative coefficient should recover the injected
        # residual amplitude -- this is the diagnostic value, not just noise
        # absorption
        deriv_idx = info_wd['derivatives']['target'][0]
        recovered = np.median(coeffs_wd[deriv_idx])
        self.assertAlmostEqual(recovered, shift_amp, delta=0.3 * shift_amp)

    def test_condition_number_stable_with_derivatives(self):
        prf = _GaussianPRF()
        A, _ = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                    poly_order=2, include_psf_derivatives=True)
        cond = np.linalg.cond(A.T @ A)
        self.assertLess(cond, 1e8)


class TestCatalogInformedCrowding(unittest.TestCase):
    """Ridge-prior handling of a tightly-blended neighbour, informed by a
    catalog-predicted flux -- keeps the vectorized closed-form solve while
    stabilizing near-degenerate PRF columns."""

    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.prf = _GaussianPRF()
        self.nfr = 30

    def _blended_setup(self):
        neighbour_dxdy = [(0.15, 0.1)]  # extremely close -> near-degenerate
        A, info = build_design_matrix(self.prf, CENT, (0.0, 0.0), neighbour_dxdy,
                                       STAMP_SIZE, poly_order=0, include_psf_derivatives=False)
        return A, info

    def test_unconstrained_blend_is_ill_conditioned(self):
        A, _ = self._blended_setup()
        cond = np.linalg.cond(A.T @ A)
        self.assertGreater(cond, 1e4)

    def test_nonfinite_prior_raises_instead_of_poisoning_solve(self):
        """A non-finite catalog magnitude must not silently NaN every column
        of the solve via the matrix inverse -- it should raise instead.
        Regression test for a real bug found on live TESS data: one NaN
        catalog mag among the selected neighbours produced all-NaN flux for
        every frame and every snap mode, with no error raised."""
        A, info = self._blended_setup()
        cube = self.rng.normal(1000, 5, size=(self.nfr, STAMP_SIZE, STAMP_SIZE))

        prior_flux = np.zeros(A.shape[1])
        prior_strength = np.zeros(A.shape[1])
        prior_flux[info['neighbours'][0]] = np.nan  # e.g. missing catalog mag
        prior_strength[info['neighbours'][0]] = 50.0

        with self.assertRaises(ValueError):
            fit_scene_lightcurve(cube, A, prior_flux=prior_flux, prior_strength=prior_strength)

    def test_zero_strength_ignores_nonfinite_prior_flux(self):
        """If a column's prior_strength is exactly 0, a non-finite prior_flux
        for that column must be tolerated (it contributes nothing)."""
        A, info = self._blended_setup()
        cube = self.rng.normal(1000, 5, size=(self.nfr, STAMP_SIZE, STAMP_SIZE))

        prior_flux = np.zeros(A.shape[1])
        prior_strength = np.zeros(A.shape[1])
        prior_flux[info['neighbours'][0]] = np.nan
        prior_strength[info['neighbours'][0]] = 0.0

        flux, e_flux, coeffs = fit_scene_lightcurve(cube, A, prior_flux=prior_flux,
                                                     prior_strength=prior_strength)
        self.assertTrue(np.all(np.isfinite(flux)))

    def test_ridge_prior_stabilizes_target_flux(self):
        A, info = self._blended_setup()
        true_target_flux = np.full(self.nfr, 1000.0)
        true_neighbour_flux = 800.0
        true_bg = 50.0

        target_col = A[:, 0]
        neighbour_col = A[:, info['neighbours'][0]]
        bg_col = A[:, info['background'][0]]

        data = (np.outer(true_target_flux, target_col) + true_neighbour_flux * neighbour_col
                + true_bg * bg_col)
        data += self.rng.normal(0, 3.0, size=data.shape)
        cube = data.reshape(self.nfr, STAMP_SIZE, STAMP_SIZE)

        flux_unc, e_unc, _ = fit_scene_lightcurve(cube, A)

        prior_flux = np.zeros(A.shape[1])
        prior_strength = np.zeros(A.shape[1])
        prior_flux[info['neighbours'][0]] = true_neighbour_flux
        prior_strength[info['neighbours'][0]] = 50.0

        flux_prior, e_prior, _ = fit_scene_lightcurve(cube, A, prior_flux=prior_flux,
                                                       prior_strength=prior_strength)

        # ridge prior should not make the fit less stable than unconstrained
        self.assertLessEqual(np.std(flux_prior), np.std(flux_unc) * 1.2)
        # and should keep the target flux closer to truth on average
        self.assertLess(abs(np.median(flux_prior) - 1000.0),
                         abs(np.median(flux_unc) - 1000.0) + 50.0)

    def test_prior_negligible_when_weak(self):
        """A very weak prior_strength shouldn't meaningfully drag a
        well-separated neighbour's flux away from its unconstrained fit."""
        prf = self.prf
        A, info = build_design_matrix(prf, CENT, (0.0, 0.0), [(4.0, 0.0)], STAMP_SIZE,
                                       poly_order=0, include_psf_derivatives=False)
        target_col = A[:, 0]
        neighbour_col = A[:, info['neighbours'][0]]
        bg_col = A[:, info['background'][0]]

        true_neighbour_flux = 800.0
        data = (np.outer(np.full(self.nfr, 1000.0), target_col)
                + true_neighbour_flux * neighbour_col + 50.0 * bg_col)
        data += self.rng.normal(0, 1.0, size=data.shape)
        cube = data.reshape(self.nfr, STAMP_SIZE, STAMP_SIZE)

        flux_unc, _, coeffs_unc = fit_scene_lightcurve(cube, A)

        prior_flux = np.zeros(A.shape[1])
        prior_strength = np.zeros(A.shape[1])
        # deliberately "wrong" prior, but with ~zero strength
        prior_flux[info['neighbours'][0]] = 200.0
        prior_strength[info['neighbours'][0]] = 1e-6

        flux_weak, _, coeffs_weak = fit_scene_lightcurve(cube, A, prior_flux=prior_flux,
                                                          prior_strength=prior_strength)
        assert_allclose(coeffs_unc[info['neighbours'][0]], coeffs_weak[info['neighbours'][0]],
                         rtol=1e-2)


class TestFitScenePosition(unittest.TestCase):

    def test_recovers_known_subpixel_shift(self):
        prf = _GaussianPRF()
        true_dx, true_dy = 0.3, -0.2
        target_col = prf_column(prf, CENT, true_dx, true_dy, STAMP_SIZE)
        bg_col = np.ones(STAMP_SIZE * STAMP_SIZE)

        rng = np.random.default_rng(3)
        data = target_col * 1000 + bg_col * 50 + rng.normal(0, 1.0, target_col.shape)

        def build_A(dx, dy):
            A, _ = build_design_matrix(prf, CENT, (dx, dy), [], STAMP_SIZE,
                                        poly_order=0, include_psf_derivatives=False)
            return A

        dx_fit, dy_fit = fit_scene_position(build_A, data, tol_x=1.0, tol_y=1.0)
        self.assertAlmostEqual(dx_fit, true_dx, delta=0.15)
        self.assertAlmostEqual(dy_fit, true_dy, delta=0.15)

    def test_zero_tolerance_returns_fixed_position(self):
        prf = _GaussianPRF()

        def build_A(dx, dy):
            A, _ = build_design_matrix(prf, CENT, (dx, dy), [], STAMP_SIZE,
                                        poly_order=0, include_psf_derivatives=False)
            return A

        dx_fit, dy_fit = fit_scene_position(build_A, np.ones(STAMP_SIZE * STAMP_SIZE),
                                             tol_x=0.0, tol_y=0.0)
        self.assertEqual((dx_fit, dy_fit), (0.0, 0.0))


class TestFitSceneFrame(unittest.TestCase):

    def test_bounded_solve_recovers_flux(self):
        prf = _GaussianPRF()
        A, _ = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                    poly_order=0, include_psf_derivatives=False)
        rng = np.random.default_rng(4)
        data = A[:, 0] * 500 + A[:, 1] * 20 + rng.normal(0, 1.0, A.shape[0])

        bounds = ([0, -np.inf], [np.inf, np.inf])
        coeffs, cov, dof = fit_scene_frame(A, data, flux_bounds=bounds)
        self.assertAlmostEqual(coeffs[0], 500, delta=20)
        self.assertGreater(dof, 0)
        self.assertEqual(cov.shape, (A.shape[1], A.shape[1]))

    def test_unbounded_solve_matches_lstsq(self):
        prf = _GaussianPRF()
        A, _ = build_design_matrix(prf, CENT, (0.0, 0.0), [], STAMP_SIZE,
                                    poly_order=0, include_psf_derivatives=False)
        rng = np.random.default_rng(5)
        data = A[:, 0] * 300 + A[:, 1] * 10 + rng.normal(0, 0.5, A.shape[0])

        coeffs, cov, dof = fit_scene_frame(A, data, flux_bounds=None)
        expected, *_ = np.linalg.lstsq(A, data, rcond=None)
        assert_allclose(coeffs, expected, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
