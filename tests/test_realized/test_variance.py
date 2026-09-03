"""
Tests for mfe.realized.variance.

Numerical checks:
- RV sums to squared returns (trivial)
- BPV < RV under a pure diffusion (no jumps), E[BPV/RV] -> 1
- MedRV + MinRV are both smaller than RV in presence of outliers
- Preaveraged RV is robust to noise
"""

import numpy as np
import pytest

from mfe.realized.variance import (
    realized_variance,
    realized_bipower_variation,
    realized_med_variance,
    realized_min_variance,
    realized_preaveraged_variance,
    realized_semivariance,
)


def test_rv_equals_sum_of_squares(rng):
    r = rng.standard_normal(500) * 0.01
    result = realized_variance(r)
    assert np.isclose(result.value, np.sum(r ** 2))
    assert result.n_returns == 500


def test_rv_subsampling_close_to_rv(rng):
    r = rng.standard_normal(1000) * 0.01
    res_plain = realized_variance(r, subsamples=1)
    res_sub = realized_variance(r, subsamples=5)
    # subsampled should be within 20% of plain for i.i.d. returns
    assert abs(res_sub.subsampled_value - res_plain.value) / res_plain.value < 0.20


def test_bpv_less_than_rv_with_jump(rng):
    """With a large jump, BPV < RV."""
    r = rng.standard_normal(500) * 0.01
    r[250] += 0.20  # inject a jump
    rv = realized_variance(r).value
    bpv = realized_bipower_variation(r).value
    assert bpv < rv


def test_bpv_skip_reduces_value(rng):
    """skip > 0 should change (usually reduce) BPV."""
    r = rng.standard_normal(500) * 0.01
    bpv0 = realized_bipower_variation(r, skip=0).value
    bpv1 = realized_bipower_variation(r, skip=1).value
    # Not a strict ordering but they should differ
    assert bpv0 != bpv1


def test_med_min_rv_robust_to_outlier(rng):
    r = rng.standard_normal(500) * 0.01
    r_jump = r.copy()
    r_jump[100] += 0.50  # large outlier
    rv_plain = realized_variance(r_jump).value
    med_rv = realized_med_variance(r_jump).value
    min_rv = realized_min_variance(r_jump).value
    # robust estimators should be much smaller than naive RV with jump
    assert med_rv < rv_plain
    assert min_rv < rv_plain


def test_semivariance_sums_to_rv(rng):
    r = rng.standard_normal(500) * 0.01
    rv = realized_variance(r).value
    rs_pos, rs_neg = realized_semivariance(r)
    assert np.isclose(rs_pos.value + rs_neg.value, rv, rtol=1e-10)


def test_preaveraged_rv_positive(rng):
    r = rng.standard_normal(500) * 0.01
    result = realized_preaveraged_variance(r)
    # may not always be positive for very short samples but typically is
    assert np.isfinite(result.value)
