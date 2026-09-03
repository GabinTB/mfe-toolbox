"""
Jump detection tests.

Barndorff-Nielsen & Shephard (2006):
  "Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation",
  JFEC.

Also includes the ratio-based test and the min/med variance jump test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mfe.realized.variance import realized_variance, realized_bipower_variation
from mfe.realized.quarticity import realized_tripower_quarticity
from mfe.utils.typing import FloatArray


@dataclass
class JumpTestResult:
    statistic: float
    p_value: float
    jump_variation: float     # RV - BPV
    continuous_variation: float  # BPV
    total_variation: float    # RV
    significant: bool         # at 5% level


def bns_jump_test(
    returns: FloatArray,
    alpha: float = 0.05,
) -> JumpTestResult:
    """
    Barndorff-Nielsen & Shephard (2006) jump test based on the ratio RV/BPV.

    Z = sqrt(n) * (RV/BPV - 1) / sqrt(omega_hat)

    Under the null of no jumps, Z -> N(0, 1).

    Parameters
    ----------
    returns : (M,) log-return array
    alpha   : significance level

    Returns
    -------
    JumpTestResult
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)

    rv = realized_variance(r).value
    bpv = realized_bipower_variation(r).value
    tpq = realized_tripower_quarticity(r).value

    # Consistent estimate of asymptotic variance (BN-S 2006, Theorem 1)
    # omega = (pi^2/4 + pi - 5) * max(1, TQ/BPV^2)
    pi = np.pi
    omega_hat = (pi ** 2 / 4 + pi - 5) * max(1.0, tpq / max(bpv ** 2, 1e-300))

    z_stat = float(np.sqrt(n) * (rv / max(bpv, 1e-300) - 1) / np.sqrt(max(omega_hat, 1e-300)))
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    jump_var = max(rv - bpv, 0.0)
    return JumpTestResult(
        statistic=z_stat,
        p_value=p_val,
        jump_variation=jump_var,
        continuous_variation=bpv,
        total_variation=rv,
        significant=(p_val < alpha),
    )
