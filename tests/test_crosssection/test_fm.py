"""Tests for Fama-MacBeth."""

import numpy as np
import pytest

from mfe.crosssection.fm import fama_macbeth, rolling_betas


@pytest.fixture
def fm_data(rng):
    T, N, K = 120, 30, 3
    factors = rng.standard_normal((T, K)) * 0.01
    true_betas = rng.standard_normal((N, K))
    true_lambda = np.array([0.005, 0.003, -0.002])
    idio = rng.standard_normal((T, N)) * 0.02
    returns = (factors @ true_lambda)[: , None] + (factors @ true_betas.T) + idio
    return returns, true_betas, true_lambda


def test_fm_runs(fm_data, rng):
    returns, betas, _ = fm_data
    result = fama_macbeth(returns, betas)
    assert len(result.lambda_mean) == 4  # intercept + 3 factors
    assert np.all(np.isfinite(result.lambda_mean))


def test_fm_t_stats_finite(fm_data, rng):
    returns, betas, _ = fm_data
    result = fama_macbeth(returns, betas)
    assert np.all(np.isfinite(result.t_stats))


def test_fm_r_squared_bounded(fm_data, rng):
    returns, betas, _ = fm_data
    result = fama_macbeth(returns, betas)
    assert 0 <= result.r_squared_mean <= 1


def test_rolling_betas_shape(rng):
    T, N, K = 100, 20, 2
    returns = rng.standard_normal((T, N)) * 0.01
    factors = rng.standard_normal((T, K)) * 0.01
    betas = rolling_betas(returns, factors, window=60)
    assert betas.shape == (N, K)
