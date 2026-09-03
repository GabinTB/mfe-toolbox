"""Tests for serial correlation tests."""
import numpy as np
import pytest
from mfe.tests_stat import ljung_box, lm_test


@pytest.fixture
def white_noise(rng):
    return rng.standard_normal(500) * 0.01


@pytest.fixture
def ar1_series(rng):
    T = 500
    x = np.zeros(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = 0.8 * x[t-1] + rng.standard_normal() * 0.1
    return x


class TestLjungBox:
    def test_shape(self, white_noise):
        res = ljung_box(white_noise, max_lags=10)
        assert len(res.statistics) == 10
        assert len(res.p_values) == 10

    def test_white_noise_high_pval(self, white_noise):
        res = ljung_box(white_noise, max_lags=5)
        # White noise should not reject at 1%
        assert np.all(res.p_values > 0.01)

    def test_ar1_low_pval(self, ar1_series):
        res = ljung_box(ar1_series, max_lags=5)
        # Strongly autocorrelated series should reject
        assert res.p_values[0] < 0.05

    def test_q_statistics_increasing(self, white_noise):
        res = ljung_box(white_noise, max_lags=10)
        # Q_k is cumulative — not strictly increasing but generally so
        assert res.statistics[-1] >= res.statistics[0]

    def test_p_values_in_unit_interval(self, white_noise):
        res = ljung_box(white_noise, max_lags=10)
        assert np.all((res.p_values >= 0) & (res.p_values <= 1))


class TestLMTest:
    def test_shape(self, white_noise):
        res = lm_test(white_noise, max_lags=10)
        assert len(res.statistics) == 10

    def test_white_noise_high_pval(self, white_noise):
        res = lm_test(white_noise, max_lags=5)
        assert np.all(res.p_values > 0.01)

    def test_ar1_low_pval(self, ar1_series):
        res = lm_test(ar1_series, max_lags=5)
        assert res.p_values[0] < 0.05

    def test_robust_vs_nonrobust(self, white_noise):
        r = lm_test(white_noise, max_lags=5, robust=True)
        nr = lm_test(white_noise, max_lags=5, robust=False)
        assert r.robust is True
        assert nr.robust is False
        # Both should give reasonable (non-NaN) results
        assert np.all(np.isfinite(r.statistics))
        assert np.all(np.isfinite(nr.statistics))
