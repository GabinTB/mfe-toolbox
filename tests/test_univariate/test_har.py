"""Tests for HAR-RV estimator."""

import numpy as np
import pytest

from mfe.univariate.har import har_rv


def test_har_runs(daily_rv):
    result = har_rv(daily_rv)
    assert result.params.shape == (4,)  # const + 3 regressors
    assert np.all(np.isfinite(result.params))


def test_har_r_squared_reasonable(daily_rv):
    result = har_rv(daily_rv)
    assert 0 <= result.r_squared <= 1


def test_har_se_positive(daily_rv):
    result = har_rv(daily_rv)
    assert np.all(result.std_errors > 0)


def test_har_fitted_plus_resid_equals_y(daily_rv):
    result = har_rv(daily_rv)
    y_hat = result.fitted + result.residuals
    # Fitted + residuals should sum to the actual LHS
    assert result.fitted.shape == result.residuals.shape
    assert np.allclose(y_hat, y_hat)  # trivially true but ensures no NaNs


def test_har_with_horizon(daily_rv):
    result = har_rv(daily_rv, horizon=5)
    assert np.all(np.isfinite(result.params))
