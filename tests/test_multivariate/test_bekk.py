"""Tests for BEKK-GARCH models."""

import numpy as np
import pytest

from mfe.multivariate.bekk import BEKK, BEKKVariant
from mfe.multivariate.base import ConvergenceWarning


def test_bekk_scalar_runs(bivariate_returns):
    model = BEKK(variant="scalar")
    res = model.fit(bivariate_returns)
    assert res.conditional_covariances.shape == (1000, 2, 2)
    assert np.isfinite(res.log_likelihood)


def test_bekk_scalar_covariances_psd(bivariate_returns):
    model = BEKK(variant="scalar")
    res = model.fit(bivariate_returns)
    for t in [0, 100, 500, 999]:
        eigvals = np.linalg.eigvalsh(res.conditional_covariances[t])
        assert np.all(eigvals > -1e-8), f"Non-PSD at t={t}"


def test_bekk_diagonal_runs(bivariate_returns):
    model = BEKK(variant="diagonal")
    res = model.fit(bivariate_returns)
    assert res.conditional_covariances.shape == (1000, 2, 2)
    assert np.isfinite(res.log_likelihood)


def test_bekk_scalar_params_stationarity(bivariate_returns):
    """a^2 + b^2 < 1 for estimated scalar BEKK."""
    model = BEKK(variant="scalar")
    res = model.fit(bivariate_returns)
    a = res.diagnostics["a"]
    b = res.diagnostics["b"]
    assert a ** 2 + b ** 2 < 1.0


def test_bekk_full_raises():
    with pytest.raises(NotImplementedError):
        BEKK(variant="full")
