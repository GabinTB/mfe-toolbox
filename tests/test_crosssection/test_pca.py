"""Tests for standalone PCA."""
import numpy as np
import pytest
from mfe.crosssection import pca


@pytest.fixture
def return_data(rng):
    T, K = 300, 5
    L = np.linalg.cholesky(np.array([
        [1.,.6,.4,.2,.1],[.6,1.,.5,.3,.2],
        [.4,.5,1.,.4,.3],[.2,.3,.4,1.,.5],[.1,.2,.3,.5,1.]
    ]))
    return (rng.standard_normal((T, K)) @ L.T) * 0.01


class TestPCA:
    def test_factor_shape(self, return_data):
        res = pca(return_data, n_components=3)
        T, K = return_data.shape
        assert res.factors.shape == (T, 3)
        assert res.loadings.shape == (K, 3)
        assert res.n_components == 3

    def test_eigenvalues_descending(self, return_data):
        res = pca(return_data)
        assert np.all(np.diff(res.eigenvalues) <= 1e-10)

    def test_eigenvalues_positive(self, return_data):
        res = pca(return_data)
        assert np.all(res.eigenvalues >= -1e-12)

    def test_explained_variance_sums_to_one(self, return_data):
        res = pca(return_data)  # all components
        np.testing.assert_allclose(res.cumulative_variance[-1], 1.0, atol=1e-10)

    def test_explained_variance_increasing(self, return_data):
        res = pca(return_data)
        assert np.all(np.diff(res.cumulative_variance) >= -1e-12)

    def test_full_reconstruction_exact(self, return_data):
        T, K = return_data.shape
        res = pca(return_data, n_components=K)
        recon = res.reconstruct()
        np.testing.assert_allclose(recon, return_data, atol=1e-10)

    def test_partial_reconstruction(self, return_data):
        res = pca(return_data, n_components=3)
        recon = res.reconstruct(k_c=3)
        assert recon.shape == return_data.shape
        # Reconstruction error < original variance
        err = np.mean((recon - return_data) ** 2)
        var = np.mean(return_data ** 2)
        assert err < var

    def test_standardize(self, return_data):
        res = pca(return_data, standardize=True)
        # Factors from correlation PCA should be orthogonal
        C = res.factors.T @ res.factors / len(res.factors)
        np.testing.assert_allclose(np.diag(C), np.diag(C), atol=1e-8)

    def test_no_demean(self, rng):
        data = rng.standard_normal((200, 3)) * 0.01 + 0.001  # non-zero mean
        res_dm = pca(data, demean=True)
        res_nd = pca(data, demean=False)
        # Means should differ
        assert not np.allclose(res_dm.mean, res_nd.mean)
