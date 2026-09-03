"""
Fama-MacBeth two-pass cross-sectional regression.

Fama, E.F. & MacBeth, J.D. (1973): "Risk, Return, and Equilibrium: Empirical
Tests", Journal of Political Economy.

Two passes:
  Pass 1: For each time period t, regress cross-sectional returns on factor
          loadings (betas) to get factor risk premia lambda_t.
  Pass 2: Average lambda_t across time and compute t-statistics with
          Shanken (1992) correction for errors-in-variables.

Also implements rolling-window beta estimation (first step of pass 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.vcv import newey_west
from mfe.utils.typing import FloatArray


@dataclass
class FMResult:
    """Fama-MacBeth estimation result."""
    lambda_mean: FloatArray      # (K,) mean risk premia
    lambda_std: FloatArray       # (K,) time-series std of lambda_t
    t_stats: FloatArray          # (K,) FM t-statistics
    p_values: FloatArray         # (K,)
    t_stats_shanken: FloatArray  # (K,) Shanken-corrected t-stats
    p_values_shanken: FloatArray
    lambda_series: FloatArray    # (T, K) time series of estimated premia
    r_squared_mean: float        # mean cross-sectional R^2
    n_periods: int
    n_assets: int
    factor_names: list[str]


def fama_macbeth(
    returns: FloatArray,             # (T, N) — T periods, N assets
    betas: FloatArray,               # (N, K) — pre-estimated factor loadings
    include_intercept: bool = True,
    nw_lags: int = 0,
    shanken_correction: bool = True,
) -> FMResult:
    """
    Two-pass Fama-MacBeth regression.

    Parameters
    ----------
    returns          : (T, N) panel of asset returns
    betas            : (N, K) pre-estimated betas (from pass 1 or exogenous)
    include_intercept: add a constant to the cross-sectional regression
    nw_lags          : Newey-West lags for FM standard errors; 0 = no HAC
    shanken_correction: apply Shanken (1992) EIV correction to t-stats

    Returns
    -------
    FMResult
    """
    R = np.asarray(returns, dtype=np.float64)   # (T, N)
    B = np.asarray(betas, dtype=np.float64)     # (N, K)
    T, N = R.shape
    _, K = B.shape

    if include_intercept:
        X = np.column_stack([np.ones(N), B])  # (N, K+1)
        k_total = K + 1
    else:
        X = B
        k_total = K

    XTX_inv = np.linalg.pinv(X.T @ X)  # (k_total, k_total)

    # Pass 2: cross-sectional OLS at each t
    lambda_t = np.empty((T, k_total), dtype=np.float64)
    r2_t = np.empty(T, dtype=np.float64)

    for t in range(T):
        r_t = R[t]  # (N,)
        lam = XTX_inv @ (X.T @ r_t)
        lambda_t[t] = lam
        fitted = X @ lam
        ss_res = np.sum((r_t - fitted) ** 2)
        ss_tot = np.sum((r_t - np.mean(r_t)) ** 2)
        r2_t[t] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # FM standard errors
    lam_mean = np.mean(lambda_t, axis=0)
    lam_std = np.std(lambda_t, axis=0, ddof=1)

    if nw_lags > 0:
        # Newey-West on the lambda_t series
        centered = lambda_t - lam_mean[None, :]
        B_nw = newey_west(centered, bandwidth=nw_lags)
        fm_var = np.diag(B_nw) / T
    else:
        fm_var = lam_std ** 2 / T

    fm_se = np.sqrt(np.maximum(fm_var, 0.0))
    t_stats = np.where(fm_se > 0, lam_mean / fm_se, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=T - 1))

    # Shanken (1992) correction
    # c = 1 + lambda_f' Sigma_f^{-1} lambda_f  (where lambda_f are the premia on factors)
    if shanken_correction and K > 0:
        lam_f = lam_mean[1:] if include_intercept else lam_mean  # factor premia only
        try:
            Sigma_f = np.cov(lambda_t[:, 1:].T) if include_intercept else np.cov(lambda_t.T)
            c = 1.0 + float(lam_f @ np.linalg.solve(Sigma_f, lam_f))
        except np.linalg.LinAlgError:
            c = 1.0
        shanken_se = fm_se * np.sqrt(c)
        t_stats_s = np.where(shanken_se > 0, lam_mean / shanken_se, np.nan)
        p_values_s = 2 * (1 - stats.t.cdf(np.abs(t_stats_s), df=T - 1))
    else:
        t_stats_s = t_stats.copy()
        p_values_s = p_values.copy()

    factor_names = (["alpha"] if include_intercept else []) + [f"factor_{k+1}" for k in range(K)]

    return FMResult(
        lambda_mean=lam_mean,
        lambda_std=lam_std,
        t_stats=t_stats,
        p_values=p_values,
        t_stats_shanken=t_stats_s,
        p_values_shanken=p_values_s,
        lambda_series=lambda_t,
        r_squared_mean=float(np.mean(r2_t)),
        n_periods=T,
        n_assets=N,
        factor_names=factor_names,
    )


def rolling_betas(
    returns: FloatArray,    # (T, N)
    factors: FloatArray,    # (T, K)
    window: int = 60,
) -> FloatArray:
    """
    Rolling time-series OLS betas: for each asset n, regress returns on factors
    using a trailing window.

    Returns (N, K) array of end-of-sample betas.
    Useful for the first pass of Fama-MacBeth.
    """
    R = np.asarray(returns, dtype=np.float64)
    F = np.asarray(factors, dtype=np.float64)
    T, N = R.shape
    _, K = F.shape

    # Use full sample if T < window
    w = min(window, T)
    F_w = F[-w:]
    X = np.column_stack([np.ones(w), F_w])  # (w, K+1)

    betas = np.empty((N, K), dtype=np.float64)
    XTX_inv = np.linalg.pinv(X.T @ X)

    for n in range(N):
        r_w = R[-w:, n]
        coef = XTX_inv @ (X.T @ r_w)
        betas[n] = coef[1:]  # drop intercept

    return betas
