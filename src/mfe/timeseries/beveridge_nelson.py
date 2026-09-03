"""
Beveridge-Nelson Decomposition.

Beveridge, S. & Nelson, C.R. (1981): "A New Approach to Decomposition of
Economic Time Series into Permanent and Transitory Components with Particular
Attention to Measurement of the Business Cycle",
Journal of Monetary Economics, 7(2), 151-174.

The BN decomposition splits an I(1) series y_t into:
  y_t = tau_t + c_t

where:
  tau_t = permanent (trend) component — a random walk with drift
  c_t   = transitory (cycle) component — a zero-mean stationary process

The trend is defined as the long-run forecast:
  tau_t = lim_{h→∞} E[y_{t+h} - h*mu | I_t]

where mu = drift of y_t = E[Delta y_t].

The cycle is:
  c_t = y_t - tau_t = -sum_{j=1}^{∞} E[Delta y_{t+j} - mu | I_t]

Computation
-----------
Given a forecasting model for Delta y_t (typically an AR or ARMA), the BN
decomposition can be computed exactly without truncating infinite sums.

Two approaches are implemented:

1. State-space (exact): Cast the ARMA model into companion form and compute
   the long-run forecast analytically using the matrix (I - A)^{-1}.
   This matches the algorithm of Morley (2002) and the MFE MATLAB implementation.

2. Direct AR: Fit an AR(p) to Delta y by OLS and compute the BN trend via
   the standard formula c_t = -sum_{k=1}^{p} pi_k * Delta y_{t-k+1}
   (Stock & Watson 1988; Cogley 2001). Faster and simpler.

The MFE MATLAB beveridgenelson.m uses approach 2 (AR on first differences).
We implement both and default to approach 2.

References
----------
Morley, J.C. (2002): "A State–Space Approach to Calculating the Beveridge–Nelson
Decomposition", Economics Letters, 75(1), 123-127.

Newbold, P. (1990): "Precise and Efficient Computation of the Beveridge–Nelson
Decomposition of Economic Time Series", Journal of Monetary Economics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.typing import FloatArray


@dataclass
class BNResult:
    """Beveridge-Nelson decomposition result."""
    trend: FloatArray         # (T,) permanent component tau_t
    cycle: FloatArray         # (T,) transitory component c_t = y_t - tau_t
    original: FloatArray      # (T,) original series
    drift: float              # estimated drift mu = E[Delta y]
    ar_params: FloatArray     # AR parameters fitted to Delta y
    ar_order: int             # p (AR order used)
    method: str               # "ar" or "state_space"


def _fit_ar(
    dy: FloatArray,
    p: int,
) -> FloatArray:
    """
    Fit AR(p) to dy by OLS. Returns (p,) coefficient vector [phi_1, ..., phi_p].
    """
    T = len(dy)
    n = T - p
    # Build regressor matrix
    X = np.column_stack([dy[p - k - 1: T - k - 1] for k in range(p)])
    y = dy[p:]
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coefs


def beveridge_nelson(
    y: FloatArray,
    ar_order: int | None = None,
    method: str = "ar",
    ic: str = "aic",
) -> BNResult:
    """
    Beveridge-Nelson decomposition of an I(1) time series.

    Parameters
    ----------
    y        : (T,) level series (must be I(1) — i.e. Delta y should be stationary)
    ar_order : AR order p for the model of Delta y.
               If None, selects automatically by AIC/BIC up to min(T//4, 24).
    method   : "ar" (default) — direct AR on first differences (Cogley 2001)
               "state_space" — exact via companion form (Morley 2002)
    ic       : "aic" | "bic" — information criterion for automatic order selection

    Returns
    -------
    BNResult
        .trend  — permanent component tau_t (same length as y)
        .cycle  — transitory component c_t = y_t - tau_t
        .drift  — estimated drift of Delta y

    Notes
    -----
    The BN trend is NOT smooth — it inherits all the innovation variance of
    the series. If you want a smooth trend, use HP or BK filter instead.
    The BN cycle is zero-mean, stationary, and reflects the business-cycle
    component as defined by forecasts.
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    dy = np.diff(y)               # first differences
    mu = float(dy.mean())         # drift

    # Determine AR order
    max_p = min(T // 4, 24)
    if ar_order is None:
        ar_order = _select_ar_order(dy, max_p=max_p, ic=ic)

    p = ar_order

    if method == "ar":
        trend, cycle = _bn_ar(y, dy, mu, p)
    elif method == "state_space":
        trend, cycle = _bn_state_space(y, dy, mu, p)
    else:
        raise ValueError(f"method must be 'ar' or 'state_space', got '{method}'")

    phi = _fit_ar(dy - mu, p) if p > 0 else np.array([])

    return BNResult(
        trend=trend,
        cycle=cycle,
        original=y,
        drift=mu,
        ar_params=phi,
        ar_order=p,
        method=method,
    )


def _select_ar_order(dy: FloatArray, max_p: int = 24, ic: str = "aic") -> int:
    """
    Select AR order for dy by AIC or BIC.
    """
    T = len(dy)
    best_ic = np.inf
    best_p = 1

    dy_dm = dy - dy.mean()  # demean for AR fitting

    for p in range(1, max_p + 1):
        if p >= T // 2:
            break
        phi = _fit_ar(dy_dm, p)
        n = T - p
        # Residuals
        X = np.column_stack([dy_dm[p - k - 1: T - k - 1] for k in range(p)])
        resid = dy_dm[p:] - X @ phi
        sigma2 = float(np.sum(resid ** 2)) / n

        ll = -n / 2 * (1 + np.log(2 * np.pi * sigma2))
        if ic == "aic":
            ic_val = -2 * ll + 2 * p
        else:
            ic_val = -2 * ll + p * np.log(n)

        if ic_val < best_ic:
            best_ic = ic_val
            best_p = p

    return best_p


def _bn_ar(
    y: FloatArray,
    dy: FloatArray,
    mu: float,
    p: int,
) -> tuple[FloatArray, FloatArray]:
    """
    BN decomposition via direct AR formula (Cogley 2001; Stock & Watson 1988).

    For AR(p) with phi_1..phi_p fit to demeaned dy:
        c_t = -sum_{j=1}^{p} gamma_j * Delta y_{t-j+1} + mu_correction

    where gamma_j = sum_{k=j}^{p} phi_k  (cumulated coefficients).

    This gives the cycle directly. Trend = y - cycle.

    The formula comes from:
        c_t = -sum_{h=1}^{∞} E[Delta y_{t+h} - mu | I_t]
             = -sum_{h=1}^{∞} phi(L)^h * (dy_t - mu)  [h-step AR forecast]
    which, for AR(p), collapses to the finite sum above.
    """
    T = len(y)
    cycle = np.zeros(T, dtype=np.float64)

    if p == 0:
        # No AR dynamics: cycle = 0, trend = y
        return y.copy(), cycle

    dy_dm = dy - mu
    phi = _fit_ar(dy_dm, p)  # (p,) coefficients

    # Cumulated coefficients: gamma_j = sum_{k=j}^{p} phi_k
    gamma = np.array([float(np.sum(phi[j:])) for j in range(p)])

    # c_t = -sum_{j=1}^{p} gamma_{j-1} * dy_dm_{t-j}  (1-indexed lag)
    # We compute from t = p onwards (earlier obs use available lags)
    for t in range(T):
        c = 0.0
        for j in range(p):
            lag_idx = t - 1 - j  # Delta y at time t-1, t-2, ... (these are dy[t-1], dy[t-2],...)
            # dy[idx] = y[idx+1] - y[idx], so dy has length T-1
            # dy_dm has length T-1; dy_dm[0] = dy[0] - mu
            if 0 <= lag_idx < len(dy_dm):
                c -= gamma[j] * dy_dm[lag_idx]
        cycle[t] = c

    trend = y - cycle
    return trend, cycle


def _bn_state_space(
    y: FloatArray,
    dy: FloatArray,
    mu: float,
    p: int,
) -> tuple[FloatArray, FloatArray]:
    """
    BN decomposition via companion-form state-space (Morley 2002).

    Cast AR(p) for dy_dm into companion form:
        xi_t = A xi_{t-1} + e_t

    Long-run forecast of dy_dm from state xi_t:
        lim_{h→∞} E[dy_dm_{t+h}|I_t] = 0 (stationary, so → 0)

    Cumulative long-run forecast = (I - A)^{-1} A xi_t  (sum of h-step forecasts)

    c_t = -e1' * (I - A)^{-1} A * xi_t
    where e1 = [1, 0, ..., 0] picks out the first element.
    """
    T = len(y)
    dy_dm = dy - mu
    cycle = np.zeros(T, dtype=np.float64)

    if p == 0:
        return y.copy(), cycle

    phi = _fit_ar(dy_dm, p)

    # Companion matrix A (p × p)
    A = np.zeros((p, p), dtype=np.float64)
    A[0, :] = phi
    A[1:, :-1] = np.eye(p - 1)

    # Long-run forecast matrix: Psi = (I - A)^{-1} A
    try:
        Psi = np.linalg.solve(np.eye(p) - A, A)
    except np.linalg.LinAlgError:
        return _bn_ar(y, dy, mu, p)  # fallback to direct formula

    e1 = np.zeros(p, dtype=np.float64)
    e1[0] = 1.0

    # Long-run multiplier row: lambda = e1' Psi
    lam = e1 @ Psi  # (p,)

    # State vector xi_t = [dy_dm_t, dy_dm_{t-1}, ..., dy_dm_{t-p+1}]
    for t in range(T):
        xi = np.array([dy_dm[t - j - 1] if 0 <= t - j - 1 < len(dy_dm) else 0.0
                       for j in range(p)])
        cycle[t] = -float(lam @ xi)

    trend = y - cycle
    return trend, cycle
