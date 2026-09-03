"""
OLS and OLSNW regression — MFE toolbox ols.m / olsnw.m equivalents.

Design rationale: statsmodels OLS exists but requires a DataFrame/array
and returns a result object with a non-trivial API. These functions are
thin, fast wrappers that match the MFE toolbox calling convention exactly
and fit naturally into pipelines that already use mfe.utils.vcv.

ols(Y, X)    — OLS with White heteroskedastic SEs
olsnw(Y, X)  — OLS with Newey-West HAC SEs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.vcv import newey_west
from mfe.utils.typing import FloatArray


@dataclass
class OLSResult:
    params: FloatArray         # (K+1,) or (K,) — const first if included
    t_stats: FloatArray        # heteroskedasticity-robust t-stats
    std_errors: FloatArray     # robust standard errors
    vcv: FloatArray            # (P, P) classical homoskedastic VCV
    vcv_robust: FloatArray     # (P, P) White or NW VCV
    r_squared: float
    r_squared_adj: float
    s_squared: float           # estimated error variance (df-adjusted)
    fitted: FloatArray         # (T,) fitted values
    residuals: FloatArray      # (T,) residuals
    n_obs: int
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    p_values: FloatArray


def _compute_ols_result(
    y: FloatArray,
    X: FloatArray,
    vcv_robust: FloatArray,
    include_const: bool,
) -> OLSResult:
    n, k = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if include_const else float(y @ y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1 - r2) * (n - 1) / (n - k) if include_const else r2
    s2 = ss_res / (n - k)

    vcv_classical = s2 * XTX_inv
    se_robust = np.sqrt(np.diag(vcv_robust))
    t_stats = beta / np.where(se_robust > 0, se_robust, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # Log-likelihood (Gaussian)
    ll = float(-0.5 * n * (1 + np.log(2 * np.pi) + np.log(max(ss_res / n, 1e-30))))
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    return OLSResult(
        params=beta,
        t_stats=t_stats,
        std_errors=se_robust,
        vcv=vcv_classical,
        vcv_robust=vcv_robust,
        r_squared=r2,
        r_squared_adj=r2_adj,
        s_squared=s2,
        fitted=fitted,
        residuals=resid,
        n_obs=n,
        n_params=k,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        p_values=p_values,
    )


def ols(
    y: FloatArray,
    X: FloatArray,
    include_const: bool = True,
) -> OLSResult:
    """
    OLS regression with White heteroskedasticity-robust standard errors.

    Parameters
    ----------
    y             : (T,) dependent variable
    X             : (T, K) regressors — do NOT include a constant column
    include_const : prepend a column of ones (default True)

    Returns
    -------
    OLSResult
        .params       — coefficient vector [const?, beta_1..beta_K]
        .t_stats      — White-robust t-statistics
        .std_errors   — White-robust standard errors
        .vcv          — classical (homoskedastic) VCV
        .vcv_robust   — White heteroskedasticity-robust VCV
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]

    if include_const:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta

    # White VCV: (X'X)^{-1} * (sum e_t^2 x_t x_t') * (X'X)^{-1}
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    vcv_white = XTX_inv @ S @ XTX_inv / n

    return _compute_ols_result(y, X, vcv_white, include_const)


def olsnw(
    y: FloatArray,
    X: FloatArray,
    include_const: bool = True,
    nw_lags: int | None = None,
) -> OLSResult:
    """
    OLS regression with Newey-West HAC standard errors.

    Parameters
    ----------
    y             : (T,) dependent variable
    X             : (T, K) regressors — do NOT include a constant column
    include_const : prepend a constant (default True)
    nw_lags       : Newey-West bandwidth; if None uses floor(T^{1/3})
                    Set to 0 for White-only (no serial correlation correction)

    Returns
    -------
    OLSResult with .vcv_robust = Newey-West VCV, .std_errors = NW standard errors.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]

    if include_const:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta

    if nw_lags is None:
        nw_lags = int(np.floor(n ** (1 / 3)))

    scores = X * resid[:, None]   # (n, k)
    B_nw = newey_west(scores, bandwidth=nw_lags)
    vcv_nw = XTX_inv @ B_nw @ XTX_inv / n

    return _compute_ols_result(y, X, vcv_nw, include_const)
