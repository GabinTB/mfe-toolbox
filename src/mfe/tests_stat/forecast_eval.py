"""
Forecast evaluation tests for volatility models.

Mincer-Zarnowitz (1969): regression-based evaluation of forecasts.
Diebold & Mariano (1995): test for equal predictive accuracy.
Hansen (2005): Superior Predictive Ability test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.vcv import newey_west
from mfe.utils.typing import FloatArray


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz regression
# ---------------------------------------------------------------------------

@dataclass
class MZResult:
    """Mincer-Zarnowitz regression output."""
    alpha: float           # intercept
    beta: float            # slope
    alpha_se: float
    beta_se: float
    t_stat_alpha: float    # test alpha = 0
    t_stat_beta: float     # test beta = 1
    f_stat: float          # joint F-test: alpha=0, beta=1
    f_pvalue: float
    r_squared: float
    n_obs: int


def mincer_zarnowitz(
    realized: FloatArray,
    forecast: FloatArray,
    nw_lags: int = 0,
) -> MZResult:
    """
    Mincer-Zarnowitz regression: realized = alpha + beta * forecast + eps.

    Tests alpha = 0, beta = 1 (unbiased forecast), and the joint H0.

    Parameters
    ----------
    realized : (T,) actual realized values (e.g. RV_t)
    forecast : (T,) model forecasts (e.g. h_{t|t-1})
    nw_lags  : Newey-West lags for HAC standard errors

    Returns
    -------
    MZResult
    """
    y = np.asarray(realized, dtype=np.float64)
    yhat = np.asarray(forecast, dtype=np.float64)
    T = len(y)

    X = np.column_stack([np.ones(T), yhat])  # (T, 2)
    XTX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XTX_inv @ (X.T @ y)
    resid = y - X @ beta_hat

    # R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot

    # Standard errors
    if nw_lags > 0:
        scores = X * resid[:, None]
        B_nw = newey_west(scores, bandwidth=nw_lags)
        vcv = XTX_inv @ B_nw @ XTX_inv
    else:
        s2 = np.sum(resid ** 2) / (T - 2)
        vcv = s2 * XTX_inv

    se = np.sqrt(np.diag(vcv))
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    alpha_se, beta_se = float(se[0]), float(se[1])

    t_alpha = alpha / alpha_se
    t_beta = (beta - 1.0) / beta_se

    # Joint F-test: alpha=0, beta=1
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    r_vec = np.array([0.0, 1.0])
    diff = R @ beta_hat - r_vec
    try:
        f_stat = float(diff @ np.linalg.solve(R @ vcv @ R.T, diff)) / 2
        f_pvalue = float(1 - stats.f.cdf(f_stat, dfn=2, dfd=T - 2))
    except np.linalg.LinAlgError:
        f_stat = np.nan
        f_pvalue = np.nan

    return MZResult(
        alpha=alpha,
        beta=beta,
        alpha_se=alpha_se,
        beta_se=beta_se,
        t_stat_alpha=t_alpha,
        t_stat_beta=t_beta,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        r_squared=float(r2),
        n_obs=T,
    )


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

@dataclass
class DMResult:
    statistic: float
    p_value: float
    loss_diff_mean: float   # mean of d_t = L1_t - L2_t
    n_obs: int


def diebold_mariano(
    errors1: FloatArray,
    errors2: FloatArray,
    loss: str = "mse",
    nw_lags: int | None = None,
    alternative: str = "two-sided",
) -> DMResult:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Diebold, F.X. & Mariano, R.S. (1995): "Comparing Predictive Accuracy",
    JBES.

    Parameters
    ----------
    errors1, errors2 : (T,) forecast error arrays from two models
    loss             : "mse" | "mae" | "qlike"
    nw_lags          : Newey-West lags; if None uses int(T^{1/3})
    alternative      : "two-sided" | "greater" | "less"
                       "greater" means model1 is less accurate (H1: model1 worse)
    """
    e1 = np.asarray(errors1, dtype=np.float64)
    e2 = np.asarray(errors2, dtype=np.float64)
    T = len(e1)

    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "qlike":
        # Patton (2011) QLIKE: L = sigma2/h - log(sigma2/h) - 1
        # For errors only (assuming h=1 or normalized): approximated as e^2 - log(e^2)
        d = e1 ** 2 - np.log(e1 ** 2 + 1e-30) - (e2 ** 2 - np.log(e2 ** 2 + 1e-30))
    else:
        raise ValueError(f"loss must be 'mse', 'mae', or 'qlike', got '{loss}'")

    d_bar = float(np.mean(d))

    if nw_lags is None:
        nw_lags = int(T ** (1 / 3))

    # HAC variance of d_bar
    d_centered = (d - d_bar)[:, None]
    B_nw = newey_west(d_centered, bandwidth=nw_lags)
    var_d = float(B_nw[0, 0]) / T
    se_d = np.sqrt(max(var_d, 0.0))

    dm_stat = d_bar / max(se_d, 1e-30)

    if alternative == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "greater":
        p_val = 1 - stats.norm.cdf(dm_stat)
    else:
        p_val = stats.norm.cdf(dm_stat)

    return DMResult(
        statistic=float(dm_stat),
        p_value=float(p_val),
        loss_diff_mean=d_bar,
        n_obs=T,
    )
