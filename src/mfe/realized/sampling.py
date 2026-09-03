"""
Price filtering and return computation for HFT tick data.

Implements the sampling schemes from the MATLAB mfe-toolbox realized module:
- Calendar-time sampling (fixed clock intervals)
- Business-time sampling (fixed tick intervals)
- Calendar-uniform (uniform in clock time via interpolation)
- Business-uniform (uniform in tick space)
- Fixed-grid sampling

All functions operate on raw tick data (price, timestamp) and return
a filtered (price, time) pair ready for return computation.
"""

from __future__ import annotations

import numpy as np

from mfe.realized._types import SamplingType, TimeType
from mfe.utils.typing import FloatArray, IntArray


def price_filter(
    price: FloatArray,
    time: FloatArray,
    time_type: TimeType = TimeType.SECONDS,
    sampling_type: SamplingType = SamplingType.CALENDAR_TIME,
    sampling_interval: float | int | FloatArray = 300,
) -> tuple[FloatArray, FloatArray]:
    """
    Filter raw tick prices to a regular grid.

    Parameters
    ----------
    price : (N,) array of log or raw prices (function is agnostic)
    time  : (N,) timestamps in units specified by time_type
    time_type : how timestamps are encoded
    sampling_type : sampling scheme
    sampling_interval :
        - CalendarTime: seconds between samples
        - BusinessTime: number of ticks between samples
        - CalendarUniform / BusinessUniform: number of obs in the filtered grid
        - Fixed: (M,) array of target times

    Returns
    -------
    (filtered_price, filtered_time) — both (M,) arrays
    """
    price = np.asarray(price, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    if price.shape != time.shape:
        raise ValueError(f"price and time must have the same length, got {price.shape} vs {time.shape}")

    if sampling_type == SamplingType.CALENDAR_TIME:
        return _sample_calendar_time(price, time, float(sampling_interval))
    elif sampling_type == SamplingType.BUSINESS_TIME:
        return _sample_business_time(price, time, int(sampling_interval))
    elif sampling_type == SamplingType.CALENDAR_UNIFORM:
        return _sample_calendar_uniform(price, time, int(sampling_interval))
    elif sampling_type == SamplingType.BUSINESS_UNIFORM:
        return _sample_business_uniform(price, time, int(sampling_interval))
    elif sampling_type == SamplingType.FIXED:
        target_times = np.asarray(sampling_interval, dtype=np.float64)
        return _sample_fixed(price, time, target_times)
    else:
        raise ValueError(f"Unknown sampling_type: {sampling_type}")


def _sample_calendar_time(
    price: FloatArray,
    time: FloatArray,
    interval_seconds: float,
) -> tuple[FloatArray, FloatArray]:
    """Previous-tick interpolation on a regular clock grid."""
    t_start = time[0]
    t_end = time[-1]
    grid = np.arange(t_start, t_end + interval_seconds, interval_seconds)
    # previous-tick: for each grid point find the last tick at or before it
    idx = np.searchsorted(time, grid, side="right") - 1
    idx = np.clip(idx, 0, len(price) - 1)
    return price[idx], grid[: len(idx)]


def _sample_business_time(
    price: FloatArray,
    time: FloatArray,
    interval_ticks: int,
) -> tuple[FloatArray, FloatArray]:
    """Sample every interval_ticks ticks."""
    idx = np.arange(0, len(price), interval_ticks)
    return price[idx], time[idx]


def _sample_calendar_uniform(
    price: FloatArray,
    time: FloatArray,
    n_obs: int,
) -> tuple[FloatArray, FloatArray]:
    """n_obs uniformly spaced points in clock time."""
    grid = np.linspace(time[0], time[-1], n_obs)
    idx = np.searchsorted(time, grid, side="right") - 1
    idx = np.clip(idx, 0, len(price) - 1)
    return price[idx], grid


def _sample_business_uniform(
    price: FloatArray,
    time: FloatArray,
    n_obs: int,
) -> tuple[FloatArray, FloatArray]:
    """n_obs uniformly spaced points in tick space."""
    idx = np.round(np.linspace(0, len(price) - 1, n_obs)).astype(int)
    return price[idx], time[idx]


def _sample_fixed(
    price: FloatArray,
    time: FloatArray,
    target_times: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Previous-tick at each target_time."""
    idx = np.searchsorted(time, target_times, side="right") - 1
    idx = np.clip(idx, 0, len(price) - 1)
    return price[idx], target_times


def returns_from_prices(price: FloatArray, log: bool = True) -> FloatArray:
    """
    Compute log or simple returns from a price series.

    Parameters
    ----------
    price : (M,) filtered price array
    log   : if True (default), use log-price differences

    Returns
    -------
    (M - 1,) return array
    """
    price = np.asarray(price, dtype=np.float64)
    if log:
        return np.diff(np.log(price))
    else:
        return np.diff(price) / price[:-1]


def refresh_time(
    prices: list[FloatArray],
    times: list[FloatArray],
) -> tuple[list[FloatArray], FloatArray]:
    """
    Synchronize K asynchronous price series via refresh-time sampling
    (Barndorff-Nielsen et al. 2011).

    For two assets this is O(N1 + N2) and vectorized.
    For K > 2 this loops over assets — TODO: Cython for K > 10.

    Parameters
    ----------
    prices : list of K (N_k,) price arrays
    times  : list of K (N_k,) time arrays (same units)

    Returns
    -------
    sync_prices : list of K (M,) synchronized price arrays
    sync_times  : (M,) refresh times
    """
    K = len(prices)
    if K < 2:
        raise ValueError("Need at least 2 assets for refresh-time synchronization.")

    # Initial refresh time: first time all assets have a quote
    t_start = max(t[0] for t in times)

    # Build the synchronized grid iteratively
    sync_times = []
    current_idx = [np.searchsorted(times[k], t_start, side="right") - 1 for k in range(K)]
    current_idx = [max(0, i) for i in current_idx]

    while True:
        # Current refresh time = max of current "last trade" times per asset
        t_refresh = max(times[k][current_idx[k]] for k in range(K))
        sync_times.append(t_refresh)

        # Advance each asset to the first tick at or after t_refresh
        new_idx = []
        for k in range(K):
            i = np.searchsorted(times[k], t_refresh, side="left")
            new_idx.append(i)

        # Check if we've exhausted any asset
        if any(new_idx[k] >= len(times[k]) for k in range(K)):
            break

        current_idx = new_idx

    sync_times_arr = np.array(sync_times, dtype=np.float64)
    sync_prices = []
    for k in range(K):
        idx = np.searchsorted(times[k], sync_times_arr, side="right") - 1
        idx = np.clip(idx, 0, len(prices[k]) - 1)
        sync_prices.append(prices[k][idx])

    return sync_prices, sync_times_arr
