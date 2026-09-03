"""
Realized quarticity and integrated quarticity estimators.

Used as inputs for CLT-based inference on RV and RK.
"""

from __future__ import annotations

import math

import numpy as np

from mfe.realized._types import RealizedResult
from mfe.utils.typing import FloatArray

_MU1 = np.sqrt(2 / np.pi)
_MU43 = 2 ** (2 / 3) * np.exp(math.lgamma(7 / 6) - math.lgamma(0.5))  # E[|Z|^{4/3}]


def realized_quarticity(returns: FloatArray) -> RealizedResult:
    """
    Realized quarticity: (n/3) * sum r_t^4

    Consistent estimator of integrated quarticity IQ = int_0^1 sigma_t^4 dt.
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    rq = float((n / 3) * np.sum(r ** 4))
    return RealizedResult(value=rq, n_returns=n)


def realized_tripower_quarticity(returns: FloatArray) -> RealizedResult:
    """
    Tripower quarticity — robust to occasional jumps.

    TPQ = n * mu_{4/3}^{-3} * mean(|r_{t-2}|^{4/3} |r_{t-1}|^{4/3} |r_t|^{4/3})
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    absr = np.abs(r)
    tpq = float(
        n * (_MU43 ** -3) * np.mean(absr[:-2] ** (4 / 3) * absr[1:-1] ** (4 / 3) * absr[2:] ** (4 / 3))
    )
    return RealizedResult(value=tpq, n_returns=n)
