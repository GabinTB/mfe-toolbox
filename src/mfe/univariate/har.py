"""
HAR-RV model and extensions.

Corsi, F. (2009): "A Simple Approximate Long-Memory Model of Realized
Volatility", JFEC.

Extensions vs. original har.py
--------------------------------
- Matrix interval notation: P=[[1,1],[2,5],[6,22]] (non-overlapping intervals)
- MODIFIED spec: non-overlapping reparameterisation (same fit, different interp)
- HAR-RV-J: jump-augmented HAR
- har_forecast: multi-step point forecast from fitted result
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mfe.utils.vcv import newey_west
from mfe.utils.typing import FloatArray


@dataclass
class HARResult:
    """HAR-RV estimation result."""
    params: FloatArray
    std_errors: FloatArray
    t_stats: FloatArray
    p_values: FloatArray
    r_squared: float
    r_squared_adj: float
    residuals: FloatArray
    fitted: FloatArray
    n_obs: int
    bandwidth: int
    param_names: list[str] = field(default_factory=list)
    spec: str = "standard"
    intervals: list[tuple[int, int]] = field(default_factory=list)


def _parse_intervals(p_arg, spec: str) -> list[tuple[int, int]]:
    if isinstance(p_arg[0], (list, tuple)):
        return [(int(r[0]), int(r[1])) for r in p_arg]
    p_vec = sorted(int(v) for v in p_arg)
    if spec == "modified":
        intervals = [(1, p_vec[0])]
        for i in range(1, len(p_vec)):
            intervals.append((p_vec[i - 1] + 1, p_vec[i]))
    else:
        intervals = [(1, p) for p in p_vec]
    return intervals


def _build_har_regressors(rv: FloatArray, intervals: list[tuple[int, int]]) -> tuple[FloatArray, int]:
    T = len(rv)
    max_end = max(e for _, e in intervals)
    n = T - max_end
    X = np.empty((n, len(intervals)), dtype=np.float64)
    for col, (start, end) in enumerate(intervals):
        for row in range(n):
            t = row + max_end
            X[row, col] = np.mean(rv[t - end: t - start + 1])
    return X, max_end


def _fit(y, X, nw_lags, n, k, intervals, spec):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    fitted = X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1 - r2) * (n - 1) / (n - k)
    try:
        XTX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XTX_inv = np.linalg.pinv(X.T @ X)
    B_nw = newey_west(X * resid[:, None], bandwidth=nw_lags)
    vcv = XTX_inv @ B_nw @ XTX_inv
    diag_vcv = np.diag(vcv)
    se = np.sqrt(np.maximum(diag_vcv, 0.0))
    t_stat = beta / np.where(se > 0, se, np.nan)
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
    return beta, se, t_stat, p_val, r2, r2_adj, resid, fitted


def har_rv(
    rv: FloatArray,
    p=(1, 5, 22),
    horizon: int = 1,
    nw_lags: int | None = None,
    spec: str = "standard",
) -> HARResult:
    """
    HAR-RV estimation by OLS with Newey-West standard errors.

    Parameters
    ----------
    rv      : (T,) daily realized variance
    p       : vector [1,5,22] or matrix [[1,1],[1,5],[1,22]] or [[1,1],[2,5],[6,22]]
    horizon : forecast horizon h (LHS is h-day forward average)
    nw_lags : Newey-West bandwidth; None => 2*horizon
    spec    : "standard" (overlapping) | "modified" (non-overlapping intervals)
    """
    rv = np.asarray(rv, dtype=np.float64)
    T = len(rv)
    intervals = _parse_intervals(list(p), spec)
    max_end = max(e for _, e in intervals)

    X_regs, _ = _build_har_regressors(rv, intervals)
    n = len(X_regs)

    if horizon == 1:
        y = rv[max_end: max_end + n]
    else:
        kernel = np.ones(horizon) / horizon
        rv_fwd = np.convolve(rv, kernel[::-1], mode="full")[:T]
        rv_fwd_shifted = np.roll(rv_fwd, -horizon)
        n = min(n, T - max_end - horizon)
        y = rv_fwd_shifted[max_end: max_end + n]
        X_regs = X_regs[:n]

    X = np.column_stack([np.ones(n), X_regs])
    k = X.shape[1]
    if nw_lags is None:
        nw_lags = max(1, 2 * horizon)

    beta, se, t_stat, p_val, r2, r2_adj, resid, fitted = _fit(y, X, nw_lags, n, k, intervals, spec)

    names = ["const"] + [
        f"RV_lag{s}" if s == e else f"RV_avg{s}to{e}"
        for s, e in intervals
    ]
    return HARResult(
        params=beta, std_errors=se, t_stats=t_stat, p_values=p_val,
        r_squared=float(r2), r_squared_adj=float(r2_adj),
        residuals=resid, fitted=fitted, n_obs=n, bandwidth=nw_lags,
        param_names=names, spec=spec, intervals=intervals,
    )


def har_rv_j(
    rv: FloatArray,
    jump: FloatArray,
    p=(1, 5, 22),
    horizon: int = 1,
    nw_lags: int | None = None,
) -> HARResult:
    """
    HAR-RV-J: HAR augmented with a jump component.

    Andersen, Bollerslev & Diebold (2007). The jump regressor is the daily
    jump contribution J_t = max(RV_t - BPV_t, 0).
    """
    rv = np.asarray(rv, dtype=np.float64)
    jump = np.asarray(jump, dtype=np.float64)
    intervals = _parse_intervals(list(p), "standard")
    max_end = max(e for _, e in intervals)

    X_regs, _ = _build_har_regressors(rv, intervals)
    n = len(X_regs)
    jump_lag = jump[max_end - 1: max_end - 1 + n]

    if horizon == 1:
        y = rv[max_end: max_end + n]
    else:
        T = len(rv)
        kernel = np.ones(horizon) / horizon
        rv_fwd = np.convolve(rv, kernel[::-1], mode="full")[:T]
        rv_fwd_shifted = np.roll(rv_fwd, -horizon)
        n = min(n, T - max_end - horizon)
        y = rv_fwd_shifted[max_end: max_end + n]
        X_regs = X_regs[:n]; jump_lag = jump_lag[:n]

    X = np.column_stack([np.ones(n), X_regs, jump_lag])
    k = X.shape[1]
    if nw_lags is None:
        nw_lags = max(1, 2 * horizon)

    beta, se, t_stat, p_val, r2, r2_adj, resid, fitted = _fit(y, X, nw_lags, n, k, intervals, "standard")

    names = ["const"] + [
        f"RV_lag{s}" if s == e else f"RV_avg{s}to{e}"
        for s, e in intervals
    ] + ["Jump_lag1"]
    return HARResult(
        params=beta, std_errors=se, t_stats=t_stat, p_values=p_val,
        r_squared=float(r2), r_squared_adj=float(r2_adj),
        residuals=resid, fitted=fitted, n_obs=n, bandwidth=nw_lags,
        param_names=names, spec="standard", intervals=intervals,
    )


def har_forecast(result: HARResult, last_rv: FloatArray, horizon: int = 1) -> FloatArray:
    """
    Multi-step HAR-RV point forecast.

    Parameters
    ----------
    result   : fitted HARResult
    last_rv  : recent RV history (at least max interval end observations)
    horizon  : steps ahead

    Returns
    -------
    (horizon,) forecast array
    """
    rv_hist = list(np.asarray(last_rv, dtype=np.float64))
    forecasts = []
    for _ in range(horizon):
        rv_arr = np.array(rv_hist)
        T = len(rv_arr)
        regs = []
        for start, end in result.intervals:
            end_c = min(end, T)
            start_c = min(start, T)
            regs.append(float(np.mean(rv_arr[T - end_c: T - start_c + 1])) if T >= end_c else float(rv_arr[-1]))
        x = np.array([1.0] + regs)
        fc = float(result.params[:len(x)] @ x)
        forecasts.append(fc)
        rv_hist.append(fc)
    return np.array(forecasts)
