"""Tests for Hansen SPA test."""
import numpy as np
import pytest
from mfe.bootstrap import spa_test


@pytest.fixture
def spa_data(rng):
    T = 200
    loss_bench = rng.standard_normal(T) ** 2
    loss_models = rng.standard_normal((T, 5)) ** 2
    return loss_bench, loss_models


def test_spa_result_shape(spa_data, rng):
    lb, lm = spa_data
    res = spa_test(lb, lm, n_bootstrap=99, rng=rng)
    assert res.n_models == 5
    assert res.n_obs == 200
    assert len(res.d_bar) == 5
    assert len(res.t_stats) == 5


def test_spa_p_values_in_unit_interval(spa_data, rng):
    lb, lm = spa_data
    res = spa_test(lb, lm, n_bootstrap=99, rng=rng)
    for pval in [res.p_value_consistent, res.p_value_upper, res.p_value_lower]:
        assert 0.0 <= pval <= 1.0


def test_spa_ordering(spa_data, rng):
    """Consistent p-val <= upper p-val (consistent null is tighter)."""
    lb, lm = spa_data
    res = spa_test(lb, lm, n_bootstrap=199, rng=rng)
    # Upper should be >= consistent (Reality Check is conservative)
    assert res.p_value_upper >= res.p_value_consistent - 0.05  # allow small MC noise


def test_spa_dominant_model(rng):
    """A clearly better model should produce a small p-value."""
    T = 300
    loss_bench = np.ones(T) * 0.1 + rng.standard_normal(T) * 0.01
    # One model is uniformly better
    loss_models = np.column_stack([
        np.ones(T) * 0.05 + rng.standard_normal(T) * 0.01,   # better
        np.ones(T) * 0.15 + rng.standard_normal(T) * 0.01,   # worse
    ])
    res = spa_test(loss_bench, loss_models, n_bootstrap=299, rng=rng)
    # Better model exists → should reject H0 (small p-value)
    assert res.p_value_consistent < 0.10

def test_spa_scalar_models(rng):
    """Single alternative model (1-D) should work."""
    T = 150
    lb = rng.standard_normal(T) ** 2
    lm = rng.standard_normal(T) ** 2
    res = spa_test(lb, lm, n_bootstrap=99, rng=rng)
    assert res.n_models == 1
