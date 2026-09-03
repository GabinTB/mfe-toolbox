"""
Two-Scale Realized Variance (TSRV).

Zhang, L., Mykland, P.A. & Ait-Sahalia, Y. (2005): "A Tale of Two Time Scales:
Determining Integrated Volatility With Noisy High-Frequency Data",
JASA, 100(472), 1394-1411.

TSRV is a bias-corrected realized variance that is consistent under
i.i.d. microstructure noise. It uses two sampling frequencies:

    TSRV = (1 / (1 - n_slow/n_fast)) * (RV_slow - (n_slow/n_fast) * RV_fast)

where:
  RV_slow = realized variance at a slower (e.g. 5-min) frequency
  RV_fast = realized variance at the fastest available frequency (all ticks)
  n_slow  = number of slow-scale returns
  n_fast  = number of fast-scale returns

The correction removes the leading noise bias term from RV_fast.

Contrast with preaveraged RV (Jacod et al. 2009):
  Pre-averaging is an alternative noise-robust estimator that is also
  consistent and semiparametrically efficient, but TSRV is simpler
  and easier to interpret as a bias correction of standard RV.

Multi-scale version (MSRV) is implemented as an extension.

Reference
---------
Zhang (2006): "Efficient Estimation of Stochastic Volatility Using Noisy
Observations: A Multi-Scale Approach", Bernoulli.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfe.realized.noise import estimate_noise_variance
from mfe.utils.typing import FloatArray


@dataclass
class TSRVResult:
    tsrv: float             # TSRV estimate
    rv_fast: float          # raw all-tick RV (biased upward)
    rv_slow: float          # slow-scale RV
    noise_variance: float   # implied noise variance estimate
    n_fast: int             # number of fast-scale returns used
    n_slow: int             # number of slow-scale returns
    K: int                  # sub-sampling scale used


def tsrv(
    returns: FloatArray,
    K: int | None = None,
) -> TSRVResult:
    """
    Two-Scale Realized Variance.

    Parameters
    ----------
    returns : (N,) all-tick log-returns (finest available frequency)
    K       : sub-sampling scale for the slow-scale estimator.
              If None, uses the optimal K* from Zhang et al. (2005):
              K* = (N/12)^{1/3} * (sigma_omega^2 / IQ)^{2/3}  (approx N^{1/3})

    Returns
    -------
    TSRVResult
        .tsrv  — the TSRV estimate of integrated variance
        .rv_fast — all-tick RV (noise-contaminated)
        .rv_slow — sub-sampled RV at scale K
    """
    r = np.asarray(returns, dtype=np.float64)
    N = len(r)

    if K is None:
        # Approximate optimal K: floor(N^{1/3}) from the asymptotic formula
        K = max(2, int(np.floor(N ** (1 / 3))))

    # Fast-scale RV: all ticks
    rv_fast = float(np.sum(r ** 2))
    n_fast = N

    # Slow-scale RV: average of K sub-grids at spacing K
    # Sub-grid s uses returns r[s], r[s+K], r[s+2K], ...
    # For each sub-grid, compute RV and average over s = 0..K-1
    rv_grids = np.empty(K, dtype=np.float64)
    n_slow_total = 0
    for s in range(K):
        r_sub = r[s::K]
        rv_grids[s] = float(np.sum(r_sub ** 2))
        n_slow_total += len(r_sub)

    rv_slow = float(np.mean(rv_grids))
    n_slow = n_slow_total // K  # average grid size

    # TSRV bias correction
    # RV_fast ≈ IQ_true + 2*N*omega^2  (noise bias)
    # RV_slow ≈ IQ_true + 2*n_slow*omega^2
    # TSRV = RV_slow - (n_slow/N) * RV_fast  (zeroes out the 2*omega^2 term)
    adj = n_slow / n_fast
    tsrv_raw = rv_slow - adj * rv_fast
    # Scale correction: 1/(1 - n_slow/N)
    scale = 1.0 / (1.0 - adj)
    tsrv_val = scale * tsrv_raw

    # Implied noise variance: omega^2 = (RV_fast - IQ_approx) / (2*N)
    # Use TSRV as the IQ approximation
    noise_var = max(0.0, (rv_fast - tsrv_val) / (2 * N))

    return TSRVResult(
        tsrv=tsrv_val,
        rv_fast=rv_fast,
        rv_slow=rv_slow,
        noise_variance=noise_var,
        n_fast=n_fast,
        n_slow=n_slow,
        K=K,
    )


@dataclass
class MSRVResult:
    msrv: float
    rv_fast: float
    n_scales: int
    weights: FloatArray      # (J,) optimal weights per scale
    rv_per_scale: FloatArray  # (J,) RV at each scale


def msrv(
    returns: FloatArray,
    n_scales: int | None = None,
) -> MSRVResult:
    """
    Multi-Scale Realized Variance (MSRV) — Zhang (2006).

    Combines J sub-sampled RVs with optimally chosen weights to achieve
    the best rate of convergence under i.i.d. microstructure noise.

    Parameters
    ----------
    returns  : (N,) all-tick log-returns
    n_scales : number of scales J; if None uses min(N^{1/2}, 30)
    """
    r = np.asarray(returns, dtype=np.float64)
    N = len(r)

    J = n_scales if n_scales is not None else min(30, max(2, int(np.sqrt(N))))

    # Sub-sampled RVs at scales K = 1, 2, ..., J
    rv_scales = np.empty(J, dtype=np.float64)
    n_returns = np.empty(J, dtype=np.float64)

    for j, K in enumerate(range(1, J + 1)):
        rv_grids = [float(np.sum(r[s::K] ** 2)) for s in range(K)]
        rv_scales[j] = float(np.mean(rv_grids))
        n_returns[j] = N / K

    # Optimal weights from Zhang (2006) eq. (3.13)
    # w_K = 12 * K * (K - 1/2) * (1/n_K) / (J(J+1)(2J+1))
    K_arr = np.arange(1, J + 1, dtype=float)
    denom = J * (J + 1) * (2 * J + 1)
    weights = 12 * K_arr * (K_arr - 0.5) / (n_returns * denom)

    msrv_val = float(weights @ rv_scales)
    rv_fast = float(np.sum(r ** 2))

    return MSRVResult(
        msrv=msrv_val,
        rv_fast=rv_fast,
        n_scales=J,
        weights=weights,
        rv_per_scale=rv_scales,
    )
