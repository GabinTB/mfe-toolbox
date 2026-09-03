"""Tests for VAR estimation, Granger causality, and IRF."""
import numpy as np
import pytest
from mfe.timeseries import vectorar, grangercause, impulse_response


@pytest.fixture(scope="module")
def var_data(rng):
    T, K = 300, 3
    # Stable VAR(1): Phi with spectral radius < 1
    Phi = np.array([[0.5, 0.1, 0.0],
                    [0.0, 0.4, 0.1],
                    [0.1, 0.0, 0.5]])
    data = np.zeros((T, K))
    eps = rng.standard_normal((T, K)) * 0.01
    for t in range(1, T):
        data[t] = data[t-1] @ Phi.T + eps[t]
    return data


@pytest.fixture(scope="module")
def var_result(var_data):
    return vectorar(var_data, lags=1)


class TestVectorAR:
    def test_param_count(self, var_result):
        assert len(var_result.params) == 1
        assert var_result.params[0].shape == (3, 3)

    def test_const_shape(self, var_result):
        assert var_result.const.shape == (3,)

    def test_errors_shape(self, var_result, var_data):
        T, K = var_data.shape
        assert var_result.errors.shape == (T - 1, K)

    def test_sigma_psd(self, var_result):
        eigs = np.linalg.eigvalsh(var_result.sigma)
        assert np.all(eigs > -1e-10)

    def test_r_squared_bounded(self, var_result):
        assert np.all(var_result.r_squared >= 0)
        assert np.all(var_result.r_squared <= 1)

    def test_ll_finite(self, var_result):
        assert np.isfinite(var_result.log_likelihood)

    def test_aic_bic(self, var_result):
        assert np.isfinite(var_result.aic)
        assert np.isfinite(var_result.bic)
        assert var_result.bic > var_result.aic

    def test_irregular_lags(self, var_data):
        res = vectorar(var_data, lags=[1, 3])
        assert len(res.params) == 2

    def test_no_const(self, var_data):
        res = vectorar(var_data, lags=1, include_const=False)
        assert res.const is None

    def test_hom_uncorr_vcv(self, var_data):
        res = vectorar(var_data, lags=1, het=False, uncorr=True)
        assert np.isfinite(res.vcv).all()

    def test_het_corr_vcv(self, var_data):
        res = vectorar(var_data, lags=1, het=True, uncorr=False)
        assert res.vcv.shape[0] == res.vcv.shape[1]


class TestGrangerCause:
    def test_shape(self, var_data):
        gc = grangercause(var_data, lags=1)
        assert gc.statistics.shape == (3, 3)
        assert gc.p_values.shape == (3, 3)

    def test_diagonal_nan(self, var_data):
        gc = grangercause(var_data, lags=1)
        for i in range(3):
            assert np.isnan(gc.statistics[i, i])

    def test_p_values_in_unit_interval(self, var_data):
        gc = grangercause(var_data, lags=1)
        valid = ~np.isnan(gc.p_values)
        assert np.all(gc.p_values[valid] >= 0)
        assert np.all(gc.p_values[valid] <= 1)

    def test_wald_method(self, var_data):
        gc = grangercause(var_data, lags=1, method="wald")
        assert gc.method == "wald"
        assert np.all(np.isfinite(gc.statistics[~np.isnan(gc.statistics)]))

    def test_lm_method(self, var_data):
        gc = grangercause(var_data, lags=1, method="lm")
        assert gc.method == "lm"

    def test_true_causality_detected(self, rng):
        """y2 causes y1 by construction — should be detected with robust LR."""
        T = 500
        y2 = rng.standard_normal(T) * 0.01
        y1 = np.zeros(T)
        for t in range(1, T):
            y1[t] = 0.6 * y2[t-1] + rng.standard_normal() * 0.001
        data = np.column_stack([y1, y2])
        # Use non-robust LR (clean DGP, homoskedastic errors)
        gc = grangercause(data, lags=1, het=False)
        # y2 -> y1: gc.statistics[0,1] should be large
        assert gc.p_values[0, 1] < 0.05


class TestImpulseResponse:
    def test_shape(self, var_data):
        irf = impulse_response(var_data, lags=1, horizon=10)
        assert irf.responses.shape == (3, 3, 11)
        assert irf.std_errors.shape == (3, 3, 11)

    def test_identity_at_h0_unit(self, var_data):
        """Unit decomp: response at h=0 should be identity."""
        irf = impulse_response(var_data, lags=1, horizon=5, decomp="unit")
        np.testing.assert_allclose(irf.responses[:, :, 0], np.eye(3), atol=1e-10)

    def test_responses_finite(self, var_data):
        irf = impulse_response(var_data, lags=1, horizon=5)
        assert np.all(np.isfinite(irf.responses))

    def test_std_errors_nonnegative(self, var_data):
        irf = impulse_response(var_data, lags=1, horizon=5)
        assert np.all(irf.std_errors >= 0)

    def test_cholesky_decomp(self, var_data):
        irf = impulse_response(var_data, lags=1, horizon=5, decomp="cholesky")
        assert irf.decomp == "cholesky"

    def test_spectral_decomp(self, var_data):
        irf = impulse_response(var_data, lags=1, horizon=5, decomp="spectral")
        assert irf.decomp == "spectral"

    def test_irfs_decay(self, var_data):
        """For a stationary VAR, IRFs should decay toward zero."""
        irf = impulse_response(var_data, lags=1, horizon=20, decomp="unit")
        # max absolute response at h=20 should be < at h=1
        assert np.max(np.abs(irf.responses[:, :, 20])) < np.max(np.abs(irf.responses[:, :, 1]))
