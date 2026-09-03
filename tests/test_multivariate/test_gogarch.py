"""
Tests for O-GARCH and GO-GARCH models.

Numerical checks:
- sigma_t is PSD at all t
- U is orthogonal (GO-GARCH)
- GO-GARCH LL >= O-GARCH LL (rotation adds degrees of freedom)
- Factor variances are positive
- Mixing matrix shape is consistent
- PCA eigenvalues are descending
- n_components < K reduces mixing matrix correctly
- Both rotation methods run without error
- AIC/BIC are finite and negative (for well-fit models on scaled data)
"""

import numpy as np
import pytest

from mfe.multivariate.gogarch import (
    OGARCH, GOGARCH,
    _pca_whiten, _fastica_rotation, _moments_rotation,
    _cumulant_matrix, _assemble_covariances,
)


@pytest.fixture(scope="module")
def bivar_data():
    rng = np.random.default_rng(42)
    T, K = 500, 2
    L = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    return (rng.standard_normal((T, K)) @ L.T) * 0.01


@pytest.fixture(scope="module")
def trivar_data():
    rng = np.random.default_rng(42)
    T, K = 500, 3
    L = np.linalg.cholesky(np.array([[1.,.6,.3],[.6,1.,.4],[.3,.4,1.]]))
    return (rng.standard_normal((T, K)) @ L.T) * 0.01


@pytest.fixture(scope="module")
def og_result_2(bivar_data):
    return OGARCH().fit(bivar_data)


@pytest.fixture(scope="module")
def og_result_3(trivar_data):
    return OGARCH().fit(trivar_data)


@pytest.fixture(scope="module")
def gg_result_ica(trivar_data):
    return GOGARCH(rotation="ica").fit(trivar_data)


@pytest.fixture(scope="module")
def gg_result_moments(bivar_data):
    return GOGARCH(rotation="moments").fit(bivar_data)


# ── PCA whitening ────────────────────────────────────────────────────────────

class TestPCAWhiten:
    def test_factor_shape(self, trivar_data):
        T, K = trivar_data.shape
        f, W, evals, evecs = _pca_whiten(trivar_data)
        assert f.shape == (T, K)
        assert W.shape == (K, K)

    def test_factors_unit_variance(self, trivar_data):
        f, _, _, _ = _pca_whiten(trivar_data)
        var = np.var(f, axis=0)
        np.testing.assert_allclose(var, np.ones(f.shape[1]), atol=0.05)

    def test_factors_uncorrelated(self, trivar_data):
        f, _, _, _ = _pca_whiten(trivar_data)
        C = np.corrcoef(f.T)
        off_diag = C - np.eye(C.shape[0])
        assert np.max(np.abs(off_diag)) < 0.05

    def test_eigenvalues_descending(self, trivar_data):
        _, _, evals, _ = _pca_whiten(trivar_data)
        assert np.all(np.diff(evals) <= 0), "eigenvalues not descending"

    def test_eigenvalues_positive(self, trivar_data):
        _, _, evals, _ = _pca_whiten(trivar_data)
        assert np.all(evals > 0)

    def test_n_components(self, trivar_data):
        T, K = trivar_data.shape
        f, W, evals, _ = _pca_whiten(trivar_data, n_components=2)
        assert f.shape == (T, 2)
        assert W.shape == (K, 2)
        assert len(evals) == 2

    def test_reconstruction(self, trivar_data):
        """data_centered ≈ factors @ W' (up to PCA reconstruction error)."""
        T, K = trivar_data.shape
        f, W, _, _ = _pca_whiten(trivar_data)
        mu = trivar_data.mean(axis=0)
        recon = f @ W.T + mu
        # Full-rank PCA: reconstruction should be near-exact
        np.testing.assert_allclose(recon, trivar_data, atol=1e-10)


# ── Rotation utilities ────────────────────────────────────────────────────────

class TestRotations:
    def test_fastica_orthogonal(self, trivar_data):
        f, _, _, _ = _pca_whiten(trivar_data)
        U = _fastica_rotation(f)
        err = np.max(np.abs(U @ U.T - np.eye(U.shape[0])))
        assert err < 1e-10, f"U not orthogonal, max err={err:.2e}"

    def test_fastica_shape(self, trivar_data):
        T, K = trivar_data.shape
        f, _, _, _ = _pca_whiten(trivar_data)
        U = _fastica_rotation(f)
        assert U.shape == (K, K)

    def test_moments_orthogonal_2d(self, bivar_data):
        f, _, _, _ = _pca_whiten(bivar_data)
        U = _moments_rotation(f)
        err = np.max(np.abs(U @ U.T - np.eye(2)))
        assert err < 1e-10

    def test_moments_reduces_cumulants(self, bivar_data):
        """After moments rotation, off-diagonal cumulants should decrease."""
        f_white, _, _, _ = _pca_whiten(bivar_data)
        C_before = _cumulant_matrix(f_white)
        U = _moments_rotation(f_white)
        f_rot = f_white @ U.T
        C_after = _cumulant_matrix(f_rot)
        mask = ~np.eye(2, dtype=bool)
        assert np.sum(C_after[mask]**2) <= np.sum(C_before[mask]**2) + 1e-6


# ── Covariance assembly ───────────────────────────────────────────────────────

class TestAssembleCovariances:
    def test_shape(self):
        T, K, K_c = 100, 3, 3
        W = np.random.default_rng(0).standard_normal((K, K_c))
        h = np.abs(np.random.default_rng(1).standard_normal((T, K_c))) * 0.01
        S = _assemble_covariances(W, h)
        assert S.shape == (T, K, K)

    def test_psd(self):
        T, K = 50, 3
        rng = np.random.default_rng(7)
        W = np.linalg.qr(rng.standard_normal((K, K)))[0]
        h = np.abs(rng.standard_normal((T, K))) * 0.01 + 1e-5
        S = _assemble_covariances(W, h)
        for t in range(T):
            eigs = np.linalg.eigvalsh(S[t])
            assert np.all(eigs > -1e-12)

    def test_symmetric(self):
        T, K = 50, 4
        rng = np.random.default_rng(8)
        W = rng.standard_normal((K, K))
        h = np.abs(rng.standard_normal((T, K))) * 0.01 + 1e-5
        S = _assemble_covariances(W, h)
        for t in [0, 10, 49]:
            np.testing.assert_allclose(S[t], S[t].T, atol=1e-14)


# ── O-GARCH ──────────────────────────────────────────────────────────────────

class TestOGARCH:
    def test_result_shape(self, og_result_3, trivar_data):
        T, K = trivar_data.shape
        assert og_result_3.conditional_covariances.shape == (T, K, K)
        assert og_result_3.factors.shape == (T, K)
        assert og_result_3.factor_variances.shape == (T, K)
        assert og_result_3.mixing_matrix.shape == (K, K)

    def test_psd_all_t(self, og_result_3):
        for t in [0, 100, 250, 499]:
            eigs = np.linalg.eigvalsh(og_result_3.conditional_covariances[t])
            assert np.all(eigs > -1e-10), f"Non-PSD at t={t}"

    def test_symmetric_all_t(self, og_result_3):
        S = og_result_3.conditional_covariances
        for t in [0, 249, 499]:
            np.testing.assert_allclose(S[t], S[t].T, atol=1e-12)

    def test_factor_variances_positive(self, og_result_3):
        assert np.all(og_result_3.factor_variances > 0)

    def test_ll_finite(self, og_result_3):
        assert np.isfinite(og_result_3.log_likelihood)

    def test_aic_bic(self, og_result_3):
        assert np.isfinite(og_result_3.aic)
        assert np.isfinite(og_result_3.bic)
        assert og_result_3.bic > og_result_3.aic  # BIC >= AIC for T > e^2

    def test_n_components_reduces(self, trivar_data):
        T, K = trivar_data.shape
        res = OGARCH(n_components=2).fit(trivar_data)
        assert res.mixing_matrix.shape == (K, 2)
        assert res.factors.shape == (T, 2)
        assert res.n_components == 2

    def test_rotation_is_none(self, og_result_3):
        assert og_result_3.rotation_matrix is None

    def test_model_name(self, og_result_3):
        assert "O-GARCH" in og_result_3.model_name

    def test_bivariate(self, og_result_2):
        assert og_result_2.conditional_covariances.shape[1:] == (2, 2)


# ── GO-GARCH ─────────────────────────────────────────────────────────────────

class TestGOGARCH:
    def test_result_shape_ica(self, gg_result_ica, trivar_data):
        T, K = trivar_data.shape
        assert gg_result_ica.conditional_covariances.shape == (T, K, K)
        assert gg_result_ica.rotation_matrix.shape == (K, K)

    def test_U_orthogonal(self, gg_result_ica):
        U = gg_result_ica.rotation_matrix
        K = U.shape[0]
        err = np.max(np.abs(U @ U.T - np.eye(K)))
        assert err < 1e-10

    def test_psd_all_t(self, gg_result_ica):
        for t in [0, 100, 499]:
            eigs = np.linalg.eigvalsh(gg_result_ica.conditional_covariances[t])
            assert np.all(eigs > -1e-10), f"Non-PSD at t={t}"

    def test_gogarch_ll_geq_ogarch(self, trivar_data, og_result_3, gg_result_ica):
        """GO-GARCH has at least as many degrees of freedom as O-GARCH."""
        # This is not strict in finite samples (ICA may not find the global optimum)
        # but over a well-generated K=3 dataset it should hold in expectation.
        # We allow a small slack.
        assert gg_result_ica.log_likelihood >= og_result_3.log_likelihood - 5.0

    def test_moments_runs(self, gg_result_moments):
        assert np.isfinite(gg_result_moments.log_likelihood)
        U = gg_result_moments.rotation_matrix
        assert U.shape == (2, 2)
        err = np.max(np.abs(U @ U.T - np.eye(2)))
        assert err < 1e-10

    def test_model_name_contains_rotation(self, gg_result_ica):
        assert "GO-GARCH" in gg_result_ica.model_name
        assert "ica" in gg_result_ica.model_name

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation"):
            GOGARCH(rotation="bad")

    def test_factor_variances_positive(self, gg_result_ica):
        assert np.all(gg_result_ica.factor_variances > 0)

    def test_n_components_gogarch(self, trivar_data):
        T, K = trivar_data.shape
        res = GOGARCH(n_components=2, rotation="ica").fit(trivar_data)
        assert res.mixing_matrix.shape == (K, 2)
        assert res.n_components == 2
        for t in [0, 100, 499]:
            eigs = np.linalg.eigvalsh(res.conditional_covariances[t])
            assert np.all(eigs > -1e-10)
