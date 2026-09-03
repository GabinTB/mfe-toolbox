"""Tests for HEAVY model."""
import numpy as np
import pytest
from mfe.univariate import HEAVY


@pytest.fixture(scope="module")
def heavy_data(rng):
    T = 400
    rv = np.abs(rng.standard_normal(T)) * 1e-4 + 1e-5
    returns = rng.standard_normal(T) * 0.01
    return returns, rv


@pytest.fixture(scope="module")
def heavy_result(heavy_data):
    r, rv = heavy_data
    return HEAVY().fit(r, rv)


def test_heavy_converged(heavy_result):
    assert heavy_result.converged

def test_heavy_param_count(heavy_result):
    assert len(heavy_result.params) == 6

def test_heavy_h_r_positive(heavy_result):
    assert np.all(heavy_result.h_returns > 0)

def test_heavy_h_rm_positive(heavy_result):
    assert np.all(heavy_result.h_realized > 0)

def test_heavy_ll_finite(heavy_result):
    assert np.isfinite(heavy_result.log_likelihood)

def test_heavy_aic_bic(heavy_result):
    assert np.isfinite(heavy_result.aic)
    assert np.isfinite(heavy_result.bic)
    assert heavy_result.bic > heavy_result.aic

def test_heavy_stationarity(heavy_result):
    """alpha_r + beta_r < 1 and alpha_rm + beta_rm < 1."""
    p = heavy_result.params
    assert p[1] + p[2] < 1.0   # alpha_r + beta_r
    assert p[4] + p[5] < 1.0   # alpha_rm + beta_rm

def test_heavy_forecast(heavy_result, heavy_data):
    r, rv = heavy_data
    h_r_fc, h_rm_fc = HEAVY().forecast(heavy_result, horizon=5, last_realized=rv[-1])
    assert h_r_fc.shape == (5,)
    assert h_rm_fc.shape == (5,)
    assert np.all(h_r_fc > 0)
    assert np.all(h_rm_fc > 0)

def test_heavy_raises_on_nonpositive_rv(rng):
    r = rng.standard_normal(100) * 0.01
    rv = rng.standard_normal(100) * 0.01  # can be negative
    with pytest.raises(ValueError, match="strictly positive"):
        HEAVY().fit(r, rv)

def test_heavy_raises_on_length_mismatch(rng):
    with pytest.raises(ValueError, match="same length"):
        HEAVY().fit(rng.standard_normal(100), rng.standard_normal(99) ** 2 + 1e-6)
