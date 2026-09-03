"""
Shared fixtures for the mfe test suite.

Numerical reference values come from the MATLAB mfe-toolbox outputs
stored in dev/matlab_reference/. Where MATLAB reference is not available,
we cross-validate against the `arch` package or scipy.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def garch_returns(rng):
    """Synthetic GARCH(1,1) returns with N(0,1) innovations."""
    T = 2000
    omega, alpha, beta = 0.05, 0.10, 0.85
    eps = rng.standard_normal(T)
    sigma2 = np.empty(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * (eps[t - 1] * sigma2[t - 1] ** 0.5) ** 2 + beta * sigma2[t - 1]
    return eps * np.sqrt(sigma2)


@pytest.fixture(scope="session")
def tick_prices(rng):
    """
    Synthetic tick price process with microstructure noise.
    Returns (price, time) tuple where time is in seconds.
    """
    n = 10_000
    dt = 1.0  # one second per tick
    time = np.arange(n, dtype=np.float64)
    log_price_true = np.cumsum(rng.standard_normal(n) * 0.01)
    noise = rng.standard_normal(n) * 0.005
    log_price = log_price_true + noise
    price = np.exp(log_price)
    return price, time


@pytest.fixture(scope="session")
def daily_rv(rng):
    """Synthetic daily realized variance series."""
    T = 500
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.0001
    return rv


@pytest.fixture(scope="session")
def bivariate_returns(rng):
    """Bivariate correlated returns."""
    T = 1000
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, 2))
    return z @ L.T * 0.01
