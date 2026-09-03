"""Tests for Romano-Wolf StepM."""
import numpy as np
import pytest
from mfe.bootstrap import step_m


def test_stepm_shape(rng):
    T, M = 150, 4
    lb = rng.standard_normal(T) ** 2
    lm = rng.standard_normal((T, M)) ** 2
    res = step_m(lb, lm, n_bootstrap=99, rng=rng)
    assert res.n_models == M
    assert res.n_obs == T
    assert len(res.t_stats) == M
    assert len(res.p_values_adjusted) == M

def test_stepm_partition(rng):
    T, M = 150, 4
    lb = rng.standard_normal(T) ** 2
    lm = rng.standard_normal((T, M)) ** 2
    res = step_m(lb, lm, n_bootstrap=99, rng=rng)
    # rejected and accepted partition {0..M-1}
    all_idx = sorted(res.rejected + res.accepted)
    assert all_idx == list(range(M))

def test_stepm_detects_dominant_model(rng):
    T = 300
    lb = np.ones(T) * 0.10 + rng.standard_normal(T) * 0.01
    lm = np.column_stack([
        np.ones(T) * 0.05 + rng.standard_normal(T) * 0.01,  # clearly better
        np.ones(T) * 0.20 + rng.standard_normal(T) * 0.01,  # clearly worse
    ])
    res = step_m(lb, lm, alpha=0.10, n_bootstrap=299, rng=rng)
    assert 0 in res.rejected   # better model should be rejected (H0 false)

def test_stepm_p_values_in_unit_interval(rng):
    T, M = 150, 3
    lb = rng.standard_normal(T) ** 2
    lm = rng.standard_normal((T, M)) ** 2
    res = step_m(lb, lm, n_bootstrap=99, rng=rng)
    assert np.all((res.p_values_adjusted >= 0) & (res.p_values_adjusted <= 1))

def test_stepm_no_rejection_under_null(rng):
    """When all models are equivalent to benchmark, FWER <= alpha."""
    T, M = 200, 5
    rng2 = np.random.default_rng(0)
    rejections = 0
    n_trials = 30
    for _ in range(n_trials):
        lb = rng2.standard_normal(T) ** 2
        lm = rng2.standard_normal((T, M)) ** 2
        res = step_m(lb, lm, alpha=0.10, n_bootstrap=99, rng=rng2)
        if len(res.rejected) > 0:
            rejections += 1
    # FWER should be <= alpha; allow some Monte Carlo variation
    assert rejections / n_trials <= 0.40  # generous bound
