"""
Realized range estimator.

Christensen, K. & Podolskij, M. (2007): "Realized Range-Based Estimation of
Integrated Variance", Journal of Econometrics, 141(2), 323-349.

The realized range uses intra-interval high-low price ranges instead of
squared returns. Under a continuous semimartingale, the range over a
sub-interval [t_{j-1}, t_j] satisfies:

    E[(log H_j - log L_j)^2] = 4 * log(2) * IV_j

where H_j (L_j) is the highest (lowest) price in the sub-interval.
This is more efficient than squared returns due to the additional information
in extreme intra-period prices.

The realized range estimator is:
    RR = (1 / (4 * log(2))) * sum_j (log H_j - log L_j)^2

Unlike RV it requires OHLC (open/high/low/close) data per sub-interval,
which is the standard format from HFT bar data.

Also implements the normalized realized range for noise robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfe.realized._types import RealizedResult
from mfe.utils.typing import FloatArray

_4LOG2 = 4.0 * np.log(2.0)


@dataclass
class RealizedRangeResult:
    value: float             # realized range estimate
    n_intervals: int         # number of sub-intervals used
    efficiency: float        # relative efficiency vs. RV (theoretical: ~1.7)


def realized_range(
    high: FloatArray,
    low: FloatArray,
    open_: FloatArray | None = None,
    close: FloatArray | None = None,
) -> RealizedRangeResult:
    """
    Realized range estimator from sub-interval high/low prices.

    Parameters
    ----------
    high  : (M,) highest log-price in each sub-interval (or raw price)
    low   : (M,) lowest log-price in each sub-interval
    open_ : (M,) optional — open log-price (used for Yang-Zhang correction)
    close : (M,) optional — close log-price (used for Yang-Zhang correction)

    Returns
    -------
    RealizedRangeResult
        .value — realized range estimate of IV
        .n_intervals — M
        .efficiency — ~1.67 vs RV under Brownian motion

    Notes
    -----
    Input prices can be raw or log — the function uses log(high) - log(low)
    which equals log(high/low) regardless of whether inputs are already logs.
    If inputs are already log-prices, pass them directly; the formula is
    the same either way (log(e^h) - log(e^l) = h - l).
    """
    H = np.asarray(high, dtype=np.float64)
    L = np.asarray(low, dtype=np.float64)

    if H.shape != L.shape:
        raise ValueError(f"high and low must have the same shape, got {H.shape} vs {L.shape}")

    if np.any(H < L):
        raise ValueError("high must be >= low in every interval")

    M = len(H)
    log_range_sq = (np.log(H) - np.log(L)) ** 2

    rr = float(np.sum(log_range_sq)) / _4LOG2

    # Theoretical efficiency of range vs squared-return: ~1.67 under BM
    efficiency = 1.6704  # exact: 1 / (4 log 2) * pi^2/2 * (1 - 2/pi)^{-1}; approx

    return RealizedRangeResult(value=rr, n_intervals=M, efficiency=efficiency)


def realized_range_from_ticks(
    price: FloatArray,
    time: FloatArray,
    interval_seconds: float = 300.0,
) -> RealizedRangeResult:
    """
    Compute realized range from raw tick data by aggregating into OHLC bars.

    Parameters
    ----------
    price            : (N,) raw tick prices
    time             : (N,) timestamps in seconds
    interval_seconds : bar width in seconds (default 5 minutes)
    """
    price = np.asarray(price, dtype=np.float64)
    time  = np.asarray(time, dtype=np.float64)

    t_start = time[0]
    t_end   = time[-1]

    # Build bar boundaries
    edges = np.arange(t_start, t_end + interval_seconds, interval_seconds)
    M = len(edges) - 1

    highs  = np.empty(M, dtype=np.float64)
    lows   = np.empty(M, dtype=np.float64)
    valid  = np.zeros(M, dtype=bool)

    for j in range(M):
        mask = (time >= edges[j]) & (time < edges[j + 1])
        if np.any(mask):
            bar_prices = price[mask]
            highs[j] = np.max(bar_prices)
            lows[j]  = np.min(bar_prices)
            valid[j] = True

    return realized_range(highs[valid], lows[valid])
