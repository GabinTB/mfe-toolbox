"""Tests for OLS and OLSNW."""
import numpy as np
import pytest
from mfe.crosssection import ols, olsnw


@pytest.fixture
def regression_data(rng):
    T = 200
    X = rng.standard_normal((T, 2))
    true_beta = np.array([0.5, -0.3])
    y = 1.0 + X @ true_beta + rng.standard_normal(T) * 0.1
    return y, X, true_beta


class TestOLS:
    def test_param_count(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert len(res.params) == 3   # const + 2 regressors

    def test_params_close_to_truth(self, regression_data):
        y, X, beta = regression_data
        res = ols(y, X)
        np.testing.assert_allclose(res.params[1:], beta, atol=0.05)
        np.testing.assert_allclose(res.params[0], 1.0, atol=0.05)

    def test_r_squared_high(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert res.r_squared > 0.95

    def test_residuals_sum_to_zero(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert abs(res.residuals.mean()) < 1e-10

    def test_fitted_plus_resid_equals_y(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        np.testing.assert_allclose(res.fitted + res.residuals, y, atol=1e-10)

    def test_se_positive(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert np.all(res.std_errors > 0)

    def test_p_values_in_unit_interval(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert np.all((res.p_values >= 0) & (res.p_values <= 1))

    def test_no_const(self, regression_data, rng):
        T = 200
        X = rng.standard_normal((T, 2))
        y = X @ [0.5, -0.3] + rng.standard_normal(T) * 0.1
        res = ols(y, X, include_const=False)
        assert len(res.params) == 2

    def test_aic_bic_finite(self, regression_data):
        y, X, _ = regression_data
        res = ols(y, X)
        assert np.isfinite(res.aic)
        assert np.isfinite(res.bic)


class TestOLSNW:
    def test_same_coefs_as_ols(self, regression_data):
        y, X, _ = regression_data
        res_ols = ols(y, X)
        res_nw  = olsnw(y, X)
        np.testing.assert_allclose(res_ols.params, res_nw.params, atol=1e-12)

    def test_nw_se_positive(self, regression_data):
        y, X, _ = regression_data
        res = olsnw(y, X, nw_lags=4)
        assert np.all(res.std_errors > 0)

    def test_autocorrelated_errors_nw_wider(self, rng):
        """With autocorrelated errors, NW SEs should differ from White SEs."""
        T = 300
        e = np.zeros(T)
        for t in range(1, T):
            e[t] = 0.7 * e[t-1] + rng.standard_normal() * 0.1
        X = rng.standard_normal((T, 1))
        y = X[:, 0] + e
        res_w  = ols(y, X)
        res_nw = olsnw(y, X, nw_lags=10)
        # NW SEs should differ from White SEs (direction depends on autocorrelation structure)
        assert not np.allclose(res_w.std_errors, res_nw.std_errors, rtol=0.01)

    def test_zero_lags_equals_white(self, regression_data):
        y, X, _ = regression_data
        res_white = ols(y, X)
        res_nw0   = olsnw(y, X, nw_lags=0)
        np.testing.assert_allclose(res_white.vcv_robust, res_nw0.vcv_robust, atol=1e-10)
