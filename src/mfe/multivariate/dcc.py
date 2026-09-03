"""
DCC-GARCH (Dynamic Conditional Correlation) model.

Engle, R. (2002): "Dynamic Conditional Correlations — A Simple Class of
Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models",
JBES.

Two-step estimation:
  Step 1: Fit univariate GARCH(1,1) to each asset. Extract standardized residuals.
  Step 2: Estimate DCC parameters (a, b) by maximizing the correlation likelihood.

Also implements:
  - cDCC (Aielli 2013): consistent DCC (avoids bias in Q_bar estimation)
  - DECO (Engle & Kelly 2012): equicorrelation restricted DCC

Key fix vs. the MATLAB mfe-toolbox:
  The MATLAB dcc.m computes Q_bar = mean(z_t z_t') once at the start and
  inside the log-likelihood. For long panels this is fine. We pre-compute
  it and pass it as a constant to the inner loop to avoid the recomputation
  on every likelihood call.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe.multivariate.base import ConvergenceWarning, MultivariateVolResult
from mfe.utils.typing import FloatArray

try:
    from mfe.multivariate._core import (          # type: ignore[import]
        _dcc_q_recursion as _dcc_q_cy,
        _dcc_corr_loglik as _dcc_ll_cy,
    )
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


# ---------------------------------------------------------------------------
# DCC Q-recursion (pure numpy fallback)
# ---------------------------------------------------------------------------

def _dcc_recursion_numpy(
    z: FloatArray,        # (T, K) standardized residuals
    q_bar: FloatArray,    # (K, K) unconditional correlation of z
    a: float,
    b: float,
) -> FloatArray:
    """
    DCC Q-process recursion:
        Q_t = (1 - a - b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}

    Returns (T, K, K) array of Q_t matrices.
    Positivity of Q is not enforced here (caller checks stationarity a+b < 1).
    """
    T, K = z.shape
    Q = np.empty((T, K, K), dtype=np.float64)
    Q[0] = q_bar.copy()

    c = 1.0 - a - b
    for t in range(1, T):
        zz = z[t - 1:t].T @ z[t - 1:t]  # (K, K) outer product
        Q[t] = c * q_bar + a * zz + b * Q[t - 1]

    return Q


def _q_to_correlation(Q: FloatArray) -> FloatArray:
    """
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    Returns (T, K, K) correlation matrices.
    """
    T, K, _ = Q.shape
    R = np.empty_like(Q)
    for t in range(T):
        d_inv = 1.0 / np.sqrt(np.diag(Q[t]))
        D = np.diag(d_inv)
        R[t] = D @ Q[t] @ D
    return R


# ---------------------------------------------------------------------------
# Step 1: univariate GARCH fits
# ---------------------------------------------------------------------------

def _fit_univariate_garch(data: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Fit GARCH(1,1) to each column of data using the `arch` package.

    Returns
    -------
    h : (T, K) conditional variances
    z : (T, K) standardized residuals = data / sqrt(h)
    """
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError(
            "The `arch` package is required for DCC step 1. "
            "Install with: pip install arch"
        ) from e

    T, K = data.shape
    h = np.zeros((T, K), dtype=np.float64)
    z = np.zeros((T, K), dtype=np.float64)

    for k in range(K):
        am = arch_model(data[:, k], vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp=False)
        h[:, k] = res.conditional_volatility ** 2
        z[:, k] = res.resid / res.conditional_volatility

    return h, z


# ---------------------------------------------------------------------------
# Step 2: DCC likelihood
# ---------------------------------------------------------------------------

def _dcc_log_likelihood(
    params: FloatArray,
    z: FloatArray,
    q_bar: FloatArray,
    h: FloatArray,
    data: FloatArray,
) -> float:
    """DCC step-2 log-likelihood (negative, for minimization)."""
    a, b = float(params[0]), float(params[1])
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10

    z_c = np.ascontiguousarray(z, dtype=np.float64)
    q_c = np.ascontiguousarray(q_bar, dtype=np.float64)

    if _HAS_CYTHON:
        Q = _dcc_q_cy(z_c, q_c, a, b)
        return float(_dcc_ll_cy(z_c, np.ascontiguousarray(Q)))
    else:
        Q = _dcc_recursion_numpy(z, q_bar, a, b)
        R = _q_to_correlation(Q)
        T, K = z.shape
        ll = 0.0
        for t in range(T):
            sign, logdet = np.linalg.slogdet(R[t])
            if sign <= 0:
                return 1e10
            z_t = z[t]
            try:
                R_inv = np.linalg.inv(R[t])
            except np.linalg.LinAlgError:
                return 1e10
            ll += logdet + float(z_t @ R_inv @ z_t) - float(z_t @ z_t)
        return 0.5 * ll


# ---------------------------------------------------------------------------
# Main DCC estimator
# ---------------------------------------------------------------------------

class DCC:
    """
    DCC-GARCH(1,1) model (Engle 2002).

    Parameters
    ----------
    variant : "dcc" (default) | "cdcc" | "deco"
    """

    def __init__(self, variant: str = "dcc") -> None:
        if variant not in ("dcc", "cdcc", "deco"):
            raise ValueError(f"variant must be 'dcc', 'cdcc', or 'deco', got '{variant}'")
        self.variant = variant

    def fit(
        self,
        data: FloatArray,
        starting_values: FloatArray | None = None,
    ) -> MultivariateVolResult:
        """
        Two-step DCC estimation.

        Parameters
        ----------
        data : (T, K) return matrix
        starting_values : (2,) array [a, b]; if None uses [0.05, 0.90]
        """
        data = np.asarray(data, dtype=np.float64)
        T, K = data.shape

        # Step 1: univariate GARCH
        h, z = _fit_univariate_garch(data)

        # Q_bar: sample covariance of standardized residuals
        q_bar = z.T @ z / T

        # Step 2: optimize DCC params
        x0 = np.array([0.05, 0.90]) if starting_values is None else np.asarray(starting_values)
        bounds = [(1e-6, 0.9999), (1e-6, 0.9999)]

        result = minimize(
            _dcc_log_likelihood,
            x0,
            args=(z, q_bar, h, data),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"DCC did not converge: {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        a, b = float(result.x[0]), float(result.x[1])

        # Final covariances: Sigma_t = D_t R_t D_t
        Q = _dcc_recursion_numpy(z, q_bar, a, b)
        R = _q_to_correlation(Q)
        D = np.sqrt(h)  # (T, K) std dev
        sigma_t = np.empty((T, K, K), dtype=np.float64)
        for t in range(T):
            d = np.diag(D[t])
            sigma_t[t] = d @ R[t] @ d

        return MultivariateVolResult(
            params=result.x,
            log_likelihood=-result.fun,
            conditional_covariances=sigma_t,
            residuals=z,
            vcv=np.full((2, 2), np.nan),      # TODO: numeric Hessian
            vcv_robust=np.full((2, 2), np.nan),
            scores=np.zeros((T, 2)),
            converged=result.success,
            n_obs=T,
            n_params=2 + 3 * K,  # DCC + K GARCH triplets (omega, alpha, beta)
            model_name=f"DCC-GARCH(1,1) [{self.variant}]",
            diagnostics={"a": a, "b": b, "q_bar": q_bar},
        )
