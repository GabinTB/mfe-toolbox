"""
Realized Quantile Variance.

Christensen, Oomen & Podolskij (2010): "Realised Quantile-Based Estimation
of the Integrated Variance", Journal of Econometrics, 159(1), 74-98.

The realized quantile variance uses the quantile of the distribution of
intra-period squared returns rather than their sum. It is jump-robust:
jumps appear as extreme outliers in the distribution of r_t^2 and are
downweighted by choosing an appropriate quantile below 1.

For a given quantile probability tau in (0, 1):

    RQV(tau) = c(tau) * mean_{t} ( r_t^2 * 1{r_t^2 <= q_tau} ) * T / floor(tau * T)

where q_tau is the empirical tau-quantile of {r_t^2} and c(tau) is a
calibration constant that ensures consistency under a pure diffusion:

    c(tau) = 1 / (chi2_cdf(chi2_ppf(tau, df=1), df=1) ← same as tau for chi2(1))
           = 1 / tau (asymptotically, to leading order)

More precisely: since r_t^2 / sigma^2 ~ chi2(1) under normality,
the expectation of the quantile-truncated version satisfies:

    E[r_t^2 * 1{r_t^2 <= q_tau}] = sigma^2 * gamma(3/2, chi2_ppf(tau, 1)/2) / Gamma(3/2)

where gamma is the lower incomplete gamma function. We use this to calibrate.

Practical default: tau = 0.50 (median-based), which gives good jump robustness
while retaining reasonable efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc, gamma
from scipy.stats import chi2

from mfe.utils.typing import FloatArray


@dataclass
class RealizedQuantileVarResult:
    value: float
    tau: float
    n_returns: int
    n_truncated: int      # number of returns used (below quantile)
    quantile_cutoff: float  # the q_tau cutoff on squared returns


def _calibration_constant(tau: float) -> float:
    """
    Calibration constant c(tau) such that RQV is unbiased under a pure BM.

    Under r_t ~ N(0, sigma^2):
        E[r^2 | r^2 <= q_tau] = sigma^2 * P(chi2(3) <= q_tau) / P(chi2(1) <= q_tau)

    So we need to scale up by 1 / (P(chi2(3) <= q_tau) / P(chi2(1) <= q_tau)).
    """
    q_tau = float(chi2.ppf(tau, df=1))
    p1 = float(chi2.cdf(q_tau, df=1))   # = tau by construction
    p3 = float(chi2.cdf(q_tau, df=3))   # P(chi2(3) <= q_tau)
    if p3 <= 0:
        return 1.0
    return p1 / p3  # = tau / p3


def realized_quantile_variance(
    returns: FloatArray,
    tau: float = 0.50,
) -> RealizedQuantileVarResult:
    """
    Realized quantile variance — jump-robust quadratic variation estimator.

    Parameters
    ----------
    returns : (T,) log-return array
    tau     : quantile probability in (0, 1); default 0.50 (median)
              Lower tau → more jump-robust, less efficient.
              tau = 1 → equivalent to realized variance (no truncation).

    Returns
    -------
    RealizedQuantileVarResult
        .value — RQV estimate of integrated variance
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    r = np.asarray(returns, dtype=np.float64)
    T = len(r)
    r2 = r ** 2

    # Quantile cutoff on squared returns
    q_tau = float(np.quantile(r2, tau))

    # Truncated sum: only squared returns below q_tau
    mask = r2 <= q_tau
    n_used = int(np.sum(mask))
    truncated_sum = float(np.sum(r2[mask]))

    if n_used == 0:
        return RealizedQuantileVarResult(
            value=np.nan, tau=tau, n_returns=T, n_truncated=0, quantile_cutoff=q_tau
        )

    # Raw estimate: scale so that the mean over ALL T observations is calibrated
    raw = truncated_sum / T

    # Calibration constant
    c = _calibration_constant(tau)

    rqv = raw / c if c > 0 else raw

    return RealizedQuantileVarResult(
        value=rqv,
        tau=tau,
        n_returns=T,
        n_truncated=n_used,
        quantile_cutoff=q_tau,
    )
