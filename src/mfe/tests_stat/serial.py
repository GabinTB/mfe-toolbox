"""
Serial correlation tests for financial time series.

Ljung & Box (1978): Q-statistic for autocorrelation up to lag K.
Godfrey (1978) / Breusch (1978): LM test for serial correlation.

Key difference from statsmodels:
  statsmodels.stats.diagnostic.acorr_ljungbox only provides the standard LB test.
  lmtest here provides a HAC-robust LM variant (lmtest1 from MFE toolbox) which
  is appropriate for heteroskedastic series — the standard LB test is not.

The heteroskedasticity-robust LM test is essentially an LR-class test:
  LM = T * s_hat' * S_hat^{-1} * s_hat
where s_hat = T^{-1} X'eps_tilde (gradient under null) and S_hat is estimated
under the alternative using the White sandwich.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.utils.vcv import newey_west
from mfe.utils.typing import FloatArray


@dataclass
class LjungBoxResult:
    statistics: FloatArray   # (max_lags,) Q statistics
    p_values: FloatArray     # (max_lags,) p-values (chi2 with lag df)
    lags: FloatArray         # (max_lags,) lag indices tested


@dataclass
class LMTestResult:
    statistics: FloatArray   # (max_lags,) LM statistics
    p_values: FloatArray     # (max_lags,) chi2 p-values
    lags: FloatArray
    robust: bool             # True if HAC-robust


def ljung_box(
    data: FloatArray,
    max_lags: int = 10,
) -> LjungBoxResult:
    """
    Ljung-Box Q statistic for serial correlation.

    Q_k = T(T+2) * sum_{j=1}^{k} rho_hat_j^2 / (T - j)

    Under H0 of no autocorrelation, Q_k ~ chi2(k) asymptotically.

    NOTE: Not appropriate for heteroskedastic data (use lm_test instead).

    Parameters
    ----------
    data     : (T,) time series (demeaned or residuals)
    max_lags : number of lags to test; returns one statistic per lag 1..max_lags
    """
    x = np.asarray(data, dtype=np.float64)
    x = x - x.mean()
    T = len(x)

    # Sample autocorrelations
    acov0 = float(x @ x) / T
    rho = np.array([
        float(x[lag:] @ x[:T - lag]) / (T * acov0)
        for lag in range(1, max_lags + 1)
    ])

    lags_arr = np.arange(1, max_lags + 1)
    # Q_k = cumulative sum up to lag k
    q_terms = rho ** 2 / (T - lags_arr)
    Q = T * (T + 2) * np.cumsum(q_terms)

    p_vals = np.array([
        float(1 - stats.chi2.cdf(Q[k], df=k + 1))
        for k in range(max_lags)
    ])

    return LjungBoxResult(statistics=Q, p_values=p_vals, lags=lags_arr.astype(float))


def lm_test(
    data: FloatArray,
    max_lags: int = 10,
    robust: bool = True,
) -> LMTestResult:
    """
    LM test for serial correlation in up to max_lags lags.

    The test is an LM-test for testing the null that all of the regression
    coefficients are zero in the auxiliary regression of y_t on lags 1..Q.
    The null tested is H0: phi_1 = phi_2 = ... = phi_Q = 0.

    Parameters
    ----------
    data     : (T,) time series (typically GARCH residuals or raw returns)
    max_lags : maximum lag order to test
    robust   : if True (default), use heteroskedasticity-robust (White) VCV
               if False, use classical homoskedastic VCV

    Notes
    -----
    Equivalent to MFE toolbox lmtest1.m.
    """
    x = np.asarray(data, dtype=np.float64)
    T = len(x)
    x_dm = x - x.mean()

    stats_arr = np.empty(max_lags, dtype=np.float64)
    pvals_arr = np.empty(max_lags, dtype=np.float64)

    for q in range(1, max_lags + 1):
        # Build regressor matrix: T-q by q lags of x_dm
        n = T - q
        X = np.column_stack([x_dm[q - j - 1: T - j - 1] for j in range(q)])  # (n, q)
        eps_tilde = x_dm[q:]  # residual under null = demeaned data

        # Score: s_t = eps_tilde_t * X_t
        scores = X * eps_tilde[:, None]   # (n, q)
        s_bar = scores.mean(axis=0)       # (q,)

        if robust:
            # White sandwich: S = T^{-1} sum scores'*scores
            S = scores.T @ scores / n    # (q, q)
        else:
            # Homoskedastic: S = sigma^2 * T^{-1} X'X
            sigma2 = float(np.mean(eps_tilde ** 2))
            S = sigma2 * (X.T @ X) / n

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            stats_arr[q - 1] = np.nan
            pvals_arr[q - 1] = np.nan
            continue

        lm = float(n * s_bar @ S_inv @ s_bar)
        stats_arr[q - 1] = lm
        pvals_arr[q - 1] = float(1 - stats.chi2.cdf(lm, df=q))

    return LMTestResult(
        statistics=stats_arr,
        p_values=pvals_arr,
        lags=np.arange(1, max_lags + 1, dtype=float),
        robust=robust,
    )
