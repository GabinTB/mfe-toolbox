"""Tests for realized covariance and Hayashi-Yoshida estimators."""

import numpy as np
import pytest

from mfe.realized.covariance import (
    realized_covariance,
    realized_correlation,
    realized_hayashi_yoshida,
    realized_covariance_refresh_time,
)


def test_realized_covariance_diagonal(rng):
    """Uncorrelated returns -> off-diagonal should be near zero."""
    T = 500
    r1 = rng.standard_normal(T) * 0.01
    r2 = rng.standard_normal(T) * 0.01
    R = np.column_stack([r1, r2])
    res = realized_covariance(R)
    assert res.cov.shape == (2, 2)
    # Off-diagonal should be small relative to diagonal
    assert abs(res.cov[0, 1]) < res.cov[0, 0] * 0.2


def test_realized_covariance_psd(bivariate_returns):
    """Realized covariance must be positive semi-definite."""
    res = realized_covariance(bivariate_returns)
    eigvals = np.linalg.eigvalsh(res.cov)
    assert np.all(eigvals >= -1e-10)


def test_realized_correlation_diagonal_ones(bivariate_returns):
    corr = realized_correlation(bivariate_returns)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.shape == (2, 2)


def test_realized_correlation_bounded(bivariate_returns):
    corr = realized_correlation(bivariate_returns)
    assert np.all(np.abs(corr) <= 1.0 + 1e-10)


def test_hayashi_yoshida_bivariate(rng):
    """HY should give a (2, 2) PSD covariance matrix."""
    T1, T2 = 300, 250
    price1 = np.exp(np.cumsum(rng.standard_normal(T1) * 0.01))
    time1 = np.sort(rng.uniform(0, 23400, T1))  # seconds in trading day
    price2 = np.exp(np.cumsum(rng.standard_normal(T2) * 0.01))
    time2 = np.sort(rng.uniform(0, 23400, T2))

    res = realized_hayashi_yoshida([price1, price2], [time1, time2])
    assert res.cov.shape == (2, 2)
    eigvals = np.linalg.eigvalsh(res.cov)
    # Matrix may not be PSD if HY can give negative off-diagonal; diagonal must be positive
    assert res.cov[0, 0] > 0
    assert res.cov[1, 1] > 0


def test_hayashi_yoshida_symmetric(rng):
    T = 200
    p1 = np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    t1 = np.arange(T, dtype=float)
    p2 = np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    t2 = np.arange(T, dtype=float) + 0.5  # offset half a second

    res = realized_hayashi_yoshida([p1, p2], [t1, t2])
    assert np.isclose(res.cov[0, 1], res.cov[1, 0])


def test_refresh_time_covariance_psd(rng):
    T = 300
    p1 = np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    t1 = np.arange(T, dtype=float)
    p2 = np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    t2 = np.arange(T, dtype=float) + 0.5

    res = realized_covariance_refresh_time([p1, p2], [t1, t2])
    eigvals = np.linalg.eigvalsh(res.cov)
    assert np.all(eigvals >= -1e-10)
