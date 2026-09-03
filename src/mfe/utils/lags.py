"""
Vectorized lag-matrix utilities.

All functions operate on (T,) or (T, K) arrays and return views or
stride-tricks arrays where possible — no unnecessary copies.
"""

from __future__ import annotations

import numpy as np

from mfe.utils.typing import FloatArray, IntArray


def lag_matrix(x: FloatArray, lags: int | list[int], trim: bool = True) -> FloatArray:
    """
    Construct a lag matrix from a 1-D or 2-D array.

    Parameters
    ----------
    x : array of shape (T,) or (T, K)
    lags : int or list[int]
        If int, lags 1..lags are included.
        If list, exactly those lags are included.
    trim : bool
        If True (default), drop the leading NaN rows.

    Returns
    -------
    array of shape (T - max_lag, len(lags) * K) if trim else (T, ...)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T, K = x.shape

    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = list(lags)

    max_lag = max(lag_list)
    out = np.empty((T, len(lag_list) * K), dtype=np.float64)
    out[:] = np.nan

    for col, lag in enumerate(lag_list):
        out[lag:, col * K : (col + 1) * K] = x[: T - lag]

    if trim:
        out = out[max_lag:]

    return out


def har_lag_matrix(
    rv: FloatArray,
    horizons: tuple[int, int, int] = (1, 5, 22),
    trim: bool = True,
) -> FloatArray:
    """
    Build the HAR regressor matrix [RV_d, RV_w, RV_m] from daily RV.

    Uses rolling averages, not raw lags, matching the Corsi (2009) definition:
        RV_{t|t-h} = (1/h) * sum_{k=0}^{h-1} RV_{t-k}

    Parameters
    ----------
    rv : (T,) array of daily realized variances
    horizons : tuple of 3 ints (daily, weekly, monthly), default (1, 5, 22)
    trim : bool — drop leading NaNs

    Returns
    -------
    (T - max_h, 3) array: columns are [RV_d_lag1, RV_w, RV_m]
    """
    rv = np.asarray(rv, dtype=np.float64)
    T = len(rv)
    max_h = max(horizons)

    cols = []
    for h in horizons:
        # rolling mean over h observations, shifted by 1 (yesterday's average)
        kernel = np.ones(h) / h
        rolled = np.convolve(rv, kernel, mode="full")[:T]
        # shift right by 1: today's regressor is yesterday's rolling mean
        col = np.empty(T, dtype=np.float64)
        col[:] = np.nan
        col[1:] = rolled[:-1]
        cols.append(col)

    out = np.column_stack(cols)

    if trim:
        out = out[max_h:]

    return out
