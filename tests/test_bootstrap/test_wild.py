"""Tests for wild bootstrap."""

import numpy as np
import pytest

from mfe.bootstrap.wild import wild_bootstrap_rv, wild_bootstrap_test


def test_wild_bootstrap_rv_runs(rng):
    """Wild bootstrap with a sign-sensitive statistic (mean return) should give a proper CI."""
    r = rng.standard_normal(500) * 0.01
    # sum(r) is sign-sensitive: w_t * r_t flipped sign changes this
    mean_ret = lambda x: float(np.mean(x))
    result = wild_bootstrap_rv(r, statistic_fn=mean_ret, n_replications=99, rng=rng)
    assert np.isfinite(result.ci_lower)
    assert np.isfinite(result.ci_upper)
    assert result.ci_lower <= result.ci_upper
    assert result.n_replications == 99


def test_wild_bootstrap_rv_degenerate_for_rv(rng):
    """
    sum(r^2) is invariant to sign flips — bootstrap distribution is a single point.
    This is expected: the wild bootstrap for RV works via bias/centering,
    not via CI on the original statistic directly.
    """
    r = rng.standard_normal(200) * 0.01
    result = wild_bootstrap_rv(r, n_replications=49, rng=rng)
    # All bootstrap stats should equal the original RV (since w^2 = 1 always)
    assert np.allclose(result.bootstrap_distribution, result.statistic)


@pytest.mark.parametrize("multiplier", ["rademacher", "mammen", "normal"])
def test_wild_bootstrap_multipliers(rng, multiplier):
    r = rng.standard_normal(200) * 0.01
    result = wild_bootstrap_rv(r, n_replications=99, multiplier=multiplier, rng=rng)
    assert len(result.bootstrap_distribution) == 99
    assert result.multiplier == multiplier


def test_wild_bootstrap_custom_statistic(rng):
    r = rng.standard_normal(300) * 0.01
    # Custom statistic: BPV-like (simplified)
    def bpv(x):
        return float(np.sum(np.abs(x[:-1]) * np.abs(x[1:])))

    result = wild_bootstrap_rv(r, statistic_fn=bpv, n_replications=99, rng=rng)
    assert np.isfinite(result.statistic)
    assert np.all(np.isfinite(result.bootstrap_distribution))
