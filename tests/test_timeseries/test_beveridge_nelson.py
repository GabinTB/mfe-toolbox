"""Tests for Beveridge-Nelson decomposition."""
import numpy as np
import pytest
from mfe.timeseries import beveridge_nelson, BNResult


@pytest.fixture(scope="module")
def random_walk_with_cycle(rng):
    T = 300
    drift = 0.1
    eps = rng.standard_normal(T) * 0.5   # permanent shocks
    eta = rng.standard_normal(T) * 0.3   # transitory shocks
    cycle = np.zeros(T)
    for t in range(1, T):
        cycle[t] = 0.7 * cycle[t-1] + eta[t]
    y = np.cumsum(drift + eps) + cycle
    return y


class TestBNDecomposition:
    def test_trend_plus_cycle_equals_original_ar(self, random_walk_with_cycle):
        y = random_walk_with_cycle
        res = beveridge_nelson(y, method="ar")
        np.testing.assert_allclose(res.trend + res.cycle, y, atol=1e-10)

    def test_trend_plus_cycle_equals_original_ss(self, random_walk_with_cycle):
        y = random_walk_with_cycle
        res = beveridge_nelson(y, method="state_space")
        np.testing.assert_allclose(res.trend + res.cycle, y, atol=1e-10)

    def test_cycle_is_stationary(self, random_walk_with_cycle):
        """Cycle should have zero or near-zero mean."""
        res = beveridge_nelson(random_walk_with_cycle, ar_order=4)
        assert abs(res.cycle.mean()) < 0.5  # small mean, not requirement to be exactly 0

    def test_shapes(self, random_walk_with_cycle):
        T = len(random_walk_with_cycle)
        res = beveridge_nelson(random_walk_with_cycle, ar_order=4)
        assert res.trend.shape == (T,)
        assert res.cycle.shape == (T,)
        assert res.original.shape == (T,)

    def test_ar_params_length(self, random_walk_with_cycle):
        res = beveridge_nelson(random_walk_with_cycle, ar_order=6)
        assert len(res.ar_params) == 6
        assert res.ar_order == 6

    def test_drift_close_to_true(self, rng):
        T = 500
        true_drift = 0.05
        y = np.cumsum(true_drift + rng.standard_normal(T) * 0.1)
        res = beveridge_nelson(y, ar_order=2)
        assert abs(res.drift - true_drift) < 0.02

    def test_ar_vs_ss_cycle_correlation(self, random_walk_with_cycle):
        """AR and state-space cycles should be highly correlated."""
        res_ar = beveridge_nelson(random_walk_with_cycle, ar_order=4, method="ar")
        res_ss = beveridge_nelson(random_walk_with_cycle, ar_order=4, method="state_space")
        corr = float(np.corrcoef(res_ar.cycle, res_ss.cycle)[0, 1])
        assert corr > 0.90

    def test_auto_order_selection_aic(self, random_walk_with_cycle):
        res = beveridge_nelson(random_walk_with_cycle, ic="aic")
        assert 1 <= res.ar_order <= 24

    def test_auto_order_selection_bic(self, random_walk_with_cycle):
        res = beveridge_nelson(random_walk_with_cycle, ic="bic")
        assert 1 <= res.ar_order <= 24

    def test_pure_rw_cycle_near_zero(self, rng):
        """Pure random walk (no cycle) should give near-zero cycle."""
        T = 300
        y = np.cumsum(0.05 + rng.standard_normal(T) * 0.5)
        res = beveridge_nelson(y, ar_order=1)
        # For a pure RW, dy is white noise → AR(1) ≈ 0 → very small cycle
        assert res.cycle.std() < 0.5

    def test_invalid_method_raises(self, random_walk_with_cycle):
        with pytest.raises(ValueError, match="method"):
            beveridge_nelson(random_walk_with_cycle, method="invalid")

    def test_method_field_stored(self, random_walk_with_cycle):
        for method in ["ar", "state_space"]:
            res = beveridge_nelson(random_walk_with_cycle, ar_order=3, method=method)
            assert res.method == method
