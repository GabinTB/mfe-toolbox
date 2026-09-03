"""
Realized covariance estimators for multivariate HFT data.

Andersen, Bollerslev, Diebold & Labys (2003): synchronous realized covariance.
Hayashi & Yoshida (2005): non-synchronous covariance estimator.
Barndorff-Nielsen et al. (2011): multivariate realized kernel.

Note on Hayashi-Yoshida for K > 2:
The MATLAB realized_hayashi_yoshida.m has a TODO comment for K > 2 assets.
We implement the general K-asset case by applying the bivariate HY estimator
to each (i, j) pair and assembling the full matrix. This is O(K^2 * max(N_i, N_j))
and correct, but not the most efficient possible implementation for large K.
"""

from __future__ import annotations

import numpy as np

from mfe.realized._types import RealizedCovarianceResult
from mfe.realized.sampling import refresh_time
from mfe.realized.variance import realized_variance
from mfe.utils.typing import FloatArray

try:
    from mfe.realized._core import _hy_sweep as _hy_sweep_cy   # type: ignore[import]
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


# ---------------------------------------------------------------------------
# Synchronous realized covariance
# ---------------------------------------------------------------------------

def realized_covariance(
    returns: FloatArray,
) -> RealizedCovarianceResult:
    """
    Standard realized covariance matrix from synchronous returns.

    Parameters
    ----------
    returns : (T, K) matrix of synchronous log-returns

    Returns
    -------
    RealizedCovarianceResult with .cov = (K, K) realized covariance matrix
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    T, K = r.shape

    cov = r.T @ r  # (K, K) — NOT divided by T, this is the quadratic variation

    return RealizedCovarianceResult(
        cov=cov,
        method="synchronous",
        n_assets=K,
        n_returns=T,
    )


def realized_correlation(returns: FloatArray) -> FloatArray:
    """
    Realized correlation matrix from synchronous returns.

    Returns (K, K) correlation matrix.
    """
    res = realized_covariance(returns)
    cov = res.cov
    d = np.sqrt(np.diag(cov))
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    return d_inv[:, None] * cov * d_inv[None, :]


# ---------------------------------------------------------------------------
# Hayashi-Yoshida (non-synchronous)
# ---------------------------------------------------------------------------

def _hy_bivariate(
    price1: FloatArray,
    time1: FloatArray,
    price2: FloatArray,
    time2: FloatArray,
) -> float:
    """
    Hayashi-Yoshida estimator for a single (i, j) pair.

    HY = sum_{s,t} r1_s * r2_t * 1{[a_s, b_s] ∩ [a_t, b_t] != empty}

    Strategy:
    - N1 * N2 <= 4M: fully vectorized boolean matrix (fast)
    - N1 * N2 > 4M:  searchsorted-per-row (O(N1 * log N2), moderate memory)
    - For N > 100K each: Cython sweep-line needed (O(N1 + N2)) — not yet compiled
    """
    r1 = np.diff(np.log(np.asarray(price1, dtype=np.float64)))
    r2 = np.diff(np.log(np.asarray(price2, dtype=np.float64)))
    a1, b1 = time1[:-1], time1[1:]
    a2, b2 = time2[:-1], time2[1:]
    N1, N2 = len(r1), len(r2)

    if _HAS_CYTHON:
        # O((N1+N2) log(N1+N2)) sweep-line — no Python overhead per event
        return float(_hy_sweep_cy(
            np.ascontiguousarray(r1, dtype=np.float64),
            np.ascontiguousarray(a1, dtype=np.float64),
            np.ascontiguousarray(b1, dtype=np.float64),
            np.ascontiguousarray(r2, dtype=np.float64),
            np.ascontiguousarray(a2, dtype=np.float64),
            np.ascontiguousarray(b2, dtype=np.float64),
        ))
    elif N1 * N2 <= 4_000_000:
        # Fully vectorized — bool matrix fits in ~4MB
        overlap = (a1[:, None] < b2[None, :]) & (a2[None, :] < b1[:, None])
        return float(r1 @ (overlap @ r2))
    else:
        # searchsorted per-row: O(N1 * log N2) — handles up to ~100K ticks
        total = 0.0
        b2_sorted = b2  # already sorted if times are sorted
        a2_sorted = a2
        for s in range(N1):
            # overlap: a1[s] < b2[t] AND a2[t] < b1[s]
            left = np.searchsorted(b2_sorted, a1[s], side="right")
            right = np.searchsorted(a2_sorted, b1[s], side="left")
            if left < right:
                total += r1[s] * float(np.sum(r2[left:right]))
        return total


def realized_hayashi_yoshida(
    prices: list[FloatArray],
    times: list[FloatArray],
) -> RealizedCovarianceResult:
    """
    Hayashi-Yoshida realized covariance for K non-synchronously observed assets.

    Hayashi, T. & Yoshida, N. (2005): "On Covariance Estimation of
    Non-Synchronously Observed Diffusion Processes", Bernoulli.

    Parameters
    ----------
    prices : list of K price arrays (lengths can differ)
    times  : list of K timestamp arrays

    Returns
    -------
    RealizedCovarianceResult with (K, K) covariance matrix.
        .method = "hayashi-yoshida"

    Notes
    -----
    Diagonal elements are the standard realized variance of each asset
    (computed from their own tick data, so no synchronization needed).

    K > 2 assets: implemented as O(K^2) bivariate calls. The MATLAB
    mfe-toolbox has a TODO here for the general case — we implement it.
    """
    K = len(prices)
    if K < 2:
        raise ValueError("Need at least 2 assets.")
    if len(times) != K:
        raise ValueError(f"len(times)={len(times)} != len(prices)={K}")

    cov = np.zeros((K, K), dtype=np.float64)

    # Diagonal: realized variance from each asset's own tick data
    for k in range(K):
        r_k = np.diff(np.log(np.asarray(prices[k], dtype=np.float64)))
        cov[k, k] = float(np.sum(r_k ** 2))

    # Off-diagonal: HY estimator for each pair
    for i in range(K):
        for j in range(i + 1, K):
            hy = _hy_bivariate(
                np.asarray(prices[i], dtype=np.float64),
                np.asarray(times[i], dtype=np.float64),
                np.asarray(prices[j], dtype=np.float64),
                np.asarray(times[j], dtype=np.float64),
            )
            cov[i, j] = hy
            cov[j, i] = hy

    n_returns = min(len(p) - 1 for p in prices)

    return RealizedCovarianceResult(
        cov=cov,
        method="hayashi-yoshida",
        n_assets=K,
        n_returns=n_returns,
    )


# ---------------------------------------------------------------------------
# Refresh-time realized covariance
# ---------------------------------------------------------------------------

def realized_covariance_refresh_time(
    prices: list[FloatArray],
    times: list[FloatArray],
) -> RealizedCovarianceResult:
    """
    Realized covariance using refresh-time synchronization.

    Barndorff-Nielsen, Hansen, Lunde & Shephard (2011).

    Synchronizes K asynchronous tick streams via refresh time, then computes
    the standard realized covariance on the synchronized returns.

    Less efficient than HY (more data loss from synchronization) but gives a
    positive semi-definite matrix by construction.
    """
    sync_prices, sync_times = refresh_time(prices, times)
    sync_returns = np.column_stack([
        np.diff(np.log(p)) for p in sync_prices
    ])
    return realized_covariance(sync_returns)
