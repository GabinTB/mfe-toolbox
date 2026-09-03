"""Tests for multivariate normal log-likelihood and utilities."""
import numpy as np
import pytest
from mfe.distributions import mvnorm_loglik, mvnorm_loglik_t, mahalanobis, standardize_mvn


@pytest.fixture
def mv_data(rng):
    T, K = 100, 3
    Sigma = np.eye(K) * 1e-4
    data = rng.multivariate_normal(np.zeros(K), Sigma, size=T)
    return data, Sigma


class TestMvnormLoglik:
    def test_finite(self, mv_data):
        data, Sigma = mv_data
        ll = mvnorm_loglik(data, Sigma)
        assert np.isfinite(ll)

    def test_higher_variance_lower_ll(self, mv_data, rng):
        data, Sigma = mv_data
        Sigma_big = Sigma * 100
        ll_small = mvnorm_loglik(data, Sigma)
        ll_big   = mvnorm_loglik(data, Sigma_big)
        # Data comes from Sigma; ll_small should be higher
        assert ll_small > ll_big

    def test_constant_sigma(self, mv_data):
        data, Sigma = mv_data
        T, K = data.shape
        ll_const = mvnorm_loglik(data, Sigma)
        # Also works with (T, K, K) broadcast
        sigma_t = np.broadcast_to(Sigma[None], (T, K, K)).copy()
        ll_tv = mvnorm_loglik(data, sigma_t)
        np.testing.assert_allclose(ll_const, ll_tv, rtol=1e-10)

    def test_singular_sigma_returns_neg_inf(self, rng):
        T, K = 50, 3
        data = rng.standard_normal((T, K)) * 0.01
        Sigma = np.zeros((K, K))  # singular
        ll = mvnorm_loglik(data, Sigma)
        assert ll == -np.inf

    def test_single_obs(self, mv_data):
        data, Sigma = mv_data
        ll_t = mvnorm_loglik_t(data[0], Sigma)
        assert np.isfinite(ll_t)


class TestMahalanobis:
    def test_shape(self, mv_data):
        data, Sigma = mv_data
        d = mahalanobis(data, Sigma)
        assert d.shape == (len(data),)

    def test_mean_close_to_sqrt_K(self, rng):
        T, K = 10000, 3
        Sigma = np.eye(K) * 1e-4
        data = rng.multivariate_normal(np.zeros(K), Sigma, size=T)
        d = mahalanobis(data, Sigma)
        # E[||z||_2] where z ~ N(0,I_K) → sqrt(K) * Gamma((K+1)/2) / Gamma(K/2) ≈ sqrt(K)
        assert abs(d.mean() - np.sqrt(K)) < 0.5

    def test_nonnegative(self, mv_data):
        data, Sigma = mv_data
        d = mahalanobis(data, Sigma)
        assert np.all(d >= 0)


class TestStandardizeMvn:
    def test_shape(self, mv_data):
        data, Sigma = mv_data
        z = standardize_mvn(data, Sigma)
        assert z.shape == data.shape

    def test_standardized_approx_iid_normal(self, rng):
        T, K = 2000, 3
        Sigma = np.eye(K) * 1e-4
        data = rng.multivariate_normal(np.zeros(K), Sigma, size=T)
        z = standardize_mvn(data, Sigma)
        # Each column should be approx N(0,1)
        for k in range(K):
            assert abs(z[:, k].mean()) < 0.1
            assert abs(z[:, k].std() - 1.0) < 0.1
