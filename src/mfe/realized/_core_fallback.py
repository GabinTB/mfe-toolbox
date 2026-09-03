"""
Pure-numpy fallbacks for functions normally provided by _core.pyx.

Imported automatically if the Cython extension is not available.
All functions have identical signatures to their Cython counterparts.
"""

from __future__ import annotations

import numpy as np

from mfe.utils.typing import FloatArray


def _autocovariance_sum(returns: FloatArray, H: int) -> FloatArray:
    """
    Autocovariance sequence gamma_h for h = 0..H.
    Fallback: O(H * T), pure numpy.
    """
    r = np.asarray(returns, dtype=np.float64)
    T = len(r)
    gamma = np.empty(H + 1, dtype=np.float64)
    gamma[0] = float(r @ r)
    for h in range(1, H + 1):
        gamma[h] = float(r[h:] @ r[: T - h])
    return gamma
