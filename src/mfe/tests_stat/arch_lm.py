"""
ARCH-LM test for conditional heteroskedasticity.

Engle, R.F. (1982): "Autoregressive Conditional Heteroscedasticity with
Estimates of the Variance of United Kingdom Inflation", Econometrica.

The test regresses squared residuals on their lagged values:
  eps_t^2 = alpha_0 + alpha_1 * eps_{t-1}^2 + ... + alpha_q * eps_{t-q}^2 + u_t

H0: alpha_1 = ... = alpha_q = 0 (no ARCH effects)

Test statistic: T * R^2 ~ chi2(q) under H0.

Also computes the F-form (more reliable in small samples).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.typing import FloatArray


@dataclass
class ARCHLMResult:
    lm_stat: float       # T * R^2
    lm_pval: float       # chi2(q) p-value
    f_stat: float        # F-version of the test
    f_pval: float
    lags: int
    r_squared: float


def arch_lm(
    residuals: FloatArray,
    lags: int = 5,
) -> ARCHLMResult:
    """
    Engle ARCH-LM test for conditional heteroskedasticity.

    Parameters
    ----------
    residuals : (T,) return or residual series
    lags      : number of lags q in the auxiliary regression

    Returns
    -------
    ARCHLMResult
        .lm_stat / .lm_pval : LM test (chi2 form, q df)
        .f_stat  / .f_pval  : F-test form (more reliable in small T)
    """
    e = np.asarray(residuals, dtype=np.float64)
    T = len(e)
    e2 = e ** 2

    # Build auxiliary regression: e2_t on constant + e2_{t-1..t-q}
    n = T - lags
    y = e2[lags:]                            # (n,)
    X = np.column_stack(                     # (n, lags+1)
        [np.ones(n)] + [e2[lags - j - 1: T - j - 1] for j in range(lags)]
    )

    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return ARCHLMResult(np.nan, np.nan, np.nan, np.nan, lags, np.nan)

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # LM stat = T * R^2
    lm = float(n * r2)
    lm_pval = float(1 - stats.chi2.cdf(lm, df=lags))

    # F stat
    k = lags       # number of restrictions
    denom = ss_res / (n - lags - 1) if n > lags + 1 else np.nan
    f_stat = ((ss_tot - ss_res) / k) / denom if np.isfinite(denom) and denom > 0 else np.nan
    f_pval = float(1 - stats.f.cdf(f_stat, dfn=k, dfd=n - k - 1)) if np.isfinite(f_stat) else np.nan

    return ARCHLMResult(
        lm_stat=lm,
        lm_pval=lm_pval,
        f_stat=f_stat,
        f_pval=f_pval,
        lags=lags,
        r_squared=r2,
    )
