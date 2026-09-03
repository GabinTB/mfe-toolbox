"""Tests for RCC (Rotated Conditional Correlation) model."""
import numpy as np
import pytest
from mfe.multivariate import RCC, RCCResult


@pytest.fixture(scope="module")
def return_data(rng):
    T, K = 600, 3
    corr = np.array([[1., .5, .2], [.5, 1., .3], [.2, .3, 1.]])
    L = np.linalg.cholesky(corr)
    # GARCH-style heteroskedastic correlated returns
    h = np.ones((T, K)) * 0.0001
    r = np.zeros((T, K))
    eps = rng.standard_normal((T, K))
    for t in range(1, T):
        h[t] = 1e-5 + 0.08 * r[t-1]**2 + 0.90 * h[t-1]
    for t in range(T):
        r[t] = eps[t] @ L.T * np.sqrt(h[t])
    return r


@pytest.fixture(scope="module")
def rcc_result(return_data):
    return RCC().fit(return_data)


class TestRCC:
    def test_sigma_t_shape(self, rcc_result, return_data):
        T, K = return_data.shape
        assert rcc_result.conditional_covariances.shape == (T, K, K)

    def test_G_t_shape(self, rcc_result, return_data):
        T, K = return_data.shape
        assert rcc_result.G_t.shape == (T, K, K)

    def test_sigma_t_psd(self, rcc_result):
        for t in [0, 100, 300, 599]:
            eigs = np.linalg.eigvalsh(rcc_result.conditional_covariances[t])
            assert np.all(eigs > -1e-10), f"Non-PSD Sigma_t at t={t}"

    def test_G_t_psd(self, rcc_result):
        for t in [0, 100, 300, 599]:
            eigs = np.linalg.eigvalsh(rcc_result.G_t[t])
            assert np.all(eigs > -1e-10), f"Non-PSD G_t at t={t}"

    def test_G_t_identity_unconditional(self, rcc_result):
        """Mean of G_t should be approximately I_K (covariance targeting)."""
        G_mean = np.mean(rcc_result.G_t, axis=0)
        K = G_mean.shape[0]
        np.testing.assert_allclose(np.diag(G_mean), np.ones(K), atol=0.15)

    def test_params_bounded(self, rcc_result):
        a, b = rcc_result.a, rcc_result.b
        assert 0 <= a <= 1
        assert 0 <= b <= 1
        assert a + b < 1.0

    def test_ll_finite(self, rcc_result):
        assert np.isfinite(rcc_result.log_likelihood)

    def test_aic_bic(self, rcc_result):
        assert rcc_result.aic < rcc_result.bic  # 2 params, T large

    def test_unconditional_corr_consistent(self, rcc_result, return_data):
        """Unconditional covariance P should be the sample covariance (no Bessel)."""
        T = len(return_data)
        P_sample = return_data.T @ return_data / T   # matches RCC convention
        np.testing.assert_allclose(rcc_result.P, P_sample, atol=1e-12)

    def test_conditional_correlations(self, rcc_result):
        """Diagonal of conditional correlations should be 1."""
        corr_t = rcc_result.conditional_correlations()
        K = rcc_result.n_vars
        for t in [0, 100, 299]:
            np.testing.assert_allclose(np.diag(corr_t[t]), np.ones(K), atol=1e-10)

    def test_sigma_t_symmetric(self, rcc_result):
        for t in [0, 200, 599]:
            S = rcc_result.conditional_covariances[t]
            np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_cholesky_rotation(self, return_data):
        rcc = RCC(rotation="cholesky").fit(return_data)
        assert rcc.converged
        # PSD check
        for t in [0, 300, 599]:
            eigs = np.linalg.eigvalsh(rcc.conditional_covariances[t])
            assert np.all(eigs > -1e-10)

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation"):
            RCC(rotation="invalid")

    def test_P_is_unconditional_cov(self, rcc_result, return_data):
        """P should equal sample covariance."""
        T = len(return_data)
        P_sample = return_data.T @ return_data / T
        np.testing.assert_allclose(rcc_result.P, P_sample, atol=1e-12)

    def test_u_t_approx_identity_cov(self, rcc_result):
        """Rotated residuals u_t should have approx identity unconditional cov."""
        u = rcc_result.u_t
        T = len(u)
        cov_u = u.T @ u / T
        K = cov_u.shape[0]
        np.testing.assert_allclose(cov_u, np.eye(K), atol=1e-10)

    def test_rcc_result_attrs(self, rcc_result):
        assert isinstance(rcc_result.diagnostics, dict)
        assert "a" in rcc_result.diagnostics
        assert "b" in rcc_result.diagnostics
