"""Tests for realized range estimator."""
import numpy as np
import pytest
from mfe.realized import realized_range, realized_range_from_ticks
from mfe.realized.range_ import RealizedRangeResult


def test_realized_range_positive(rng):
    H = np.exp(rng.standard_normal(100) * 0.01 + 0.005)
    L = H * np.exp(-np.abs(rng.standard_normal(100)) * 0.003)
    res = realized_range(H, L)
    assert res.value > 0

def test_realized_range_n_intervals(rng):
    H = np.exp(rng.standard_normal(50) * 0.01 + 0.005)
    L = H * np.exp(-np.abs(rng.standard_normal(50)) * 0.003)
    res = realized_range(H, L)
    assert res.n_intervals == 50

def test_realized_range_raises_on_h_less_than_l(rng):
    H = np.ones(10) * 100.0
    L = np.ones(10) * 101.0  # L > H — invalid
    with pytest.raises(ValueError, match="high must be >= low"):
        realized_range(H, L)

def test_realized_range_equal_hl_gives_zero():
    H = np.ones(50) * 100.0
    L = np.ones(50) * 100.0  # no range
    res = realized_range(H, L)
    assert res.value == 0.0

def test_realized_range_from_ticks(rng):
    N = 1000
    price = np.exp(np.cumsum(rng.standard_normal(N) * 0.001))
    time = np.arange(N, dtype=float)  # 1 tick per second
    res = realized_range_from_ticks(price, time, interval_seconds=100.0)
    assert res.value > 0
    assert res.n_intervals > 0

def test_realized_range_efficiency():
    """Efficiency attribute should be approximately 1.67."""
    H = np.exp(np.random.default_rng(0).standard_normal(100) * 0.01 + 0.005)
    L = H * np.exp(-np.abs(np.random.default_rng(1).standard_normal(100)) * 0.003)
    res = realized_range(H, L)
    assert 1.5 < res.efficiency < 1.8

def test_realized_range_scales_with_variance(rng):
    """Doubling the log-range should quadruple the estimate."""
    H1 = np.exp(rng.standard_normal(200) * 0.005 + 0.005)
    L1 = H1 * np.exp(-np.abs(rng.standard_normal(200)) * 0.002)
    H2 = np.exp(rng.standard_normal(200) * 0.010 + 0.005)
    L2 = H2 * np.exp(-np.abs(rng.standard_normal(200)) * 0.004)
    res1 = realized_range(H1, L1)
    res2 = realized_range(H2, L2)
    # Higher vol → higher realized range
    assert res2.value > res1.value
