"""
Microstructure noise variance estimation.

Two approaches:
1. Bandi & Russell (2006): noise_var = -0.5 * mean(r_t * r_{t-1})
2. Zhang, Mykland & Ait-Sahalia (2005): from the difference between full-
   frequency RV and a sub-sampled RV.

We default to the Bandi-Russell estimator as in the MATLAB mfe-toolbox.
"""

from __future__ import annotations

import numpy as np

from mfe.utils.typing import FloatArray


def estimate_noise_variance(
    returns: FloatArray,
    method: str = "bandi-russell",
) -> float:
    """
    Estimate the microstructure noise variance omega^2.

    Parameters
    ----------
    returns : (M,) log-return array at the finest available frequency
    method  : "bandi-russell" (default) or "zma"

    Returns
    -------
    float — noise variance estimate (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)

    if method == "bandi-russell":
        return _noise_bandi_russell(r)
    elif method == "zma":
        return _noise_zma(r)
    else:
        raise ValueError(f"Unknown noise estimation method: {method}")


def _noise_bandi_russell(r: FloatArray) -> float:
    """
    Bandi & Russell (2006): omega^2 = -0.5 * mean(r_t * r_{t+1})

    Estimator is consistent under i.i.d. noise and unbiased in large samples.
    Can be negative (set to 0 in that case — indicates noise is negligible).
    """
    acov1 = float(r[1:] @ r[:-1]) / (len(r) - 1)
    return max(0.0, -0.5 * acov1)


def _noise_zma(r: FloatArray) -> float:
    """
    Zhang, Mykland & Ait-Sahalia (2005) TSRV-based noise estimate.
    Uses difference between all-ticks RV and sub-sampled (sparse) RV.
    """
    n = len(r)
    rv_all = float(np.sum(r ** 2))
    # Sparse RV: every-other-tick
    rv_sparse = float(np.sum(r[::2] ** 2)) * 2
    # noise_var estimate
    noise_var = (rv_all - rv_sparse) / (2 * n)
    return max(0.0, noise_var)
