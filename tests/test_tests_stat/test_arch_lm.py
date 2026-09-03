"""Tests for ARCH-LM test."""
import numpy as np
import pytest
from mfe.tests_stat import arch_lm


def test_arch_lm_iid_high_pval(rng):
    """i.i.d. squared returns should not show ARCH."""
    e = rng.standard_normal(500)
    res = arch_lm(e, lags=5)
    assert res.lm_pval > 0.05

def test_arch_lm_garch_low_pval(rng):
    """Simulated GARCH should reject no-ARCH null."""
    T = 1000
    h = np.ones(T) * 0.01
    e = np.zeros(T)
    eps = rng.standard_normal(T)
    for t in range(1, T):
        h[t] = 0.01 + 0.15 * e[t-1]**2 + 0.80 * h[t-1]
        e[t] = eps[t] * np.sqrt(h[t])
    res = arch_lm(e, lags=5)
    assert res.lm_pval < 0.05

def test_arch_lm_returns_finite(rng):
    e = rng.standard_normal(300)
    res = arch_lm(e, lags=3)
    assert np.isfinite(res.lm_stat)
    assert np.isfinite(res.f_stat)
    assert 0 <= res.r_squared <= 1

def test_arch_lm_f_and_lm_agree(rng):
    """F and LM tests should agree on rejection."""
    T = 1000
    h = np.ones(T) * 0.01
    e = np.zeros(T)
    eps = rng.standard_normal(T)
    for t in range(1, T):
        h[t] = 0.01 + 0.15 * e[t-1]**2 + 0.80 * h[t-1]
        e[t] = eps[t] * np.sqrt(h[t])
    res = arch_lm(e, lags=5)
    # Both should reject
    assert (res.lm_pval < 0.05) == (res.f_pval < 0.05)
