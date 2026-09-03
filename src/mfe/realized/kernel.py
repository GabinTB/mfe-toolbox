"""
Realized kernel estimator.

Barndorff-Nielsen, Hansen, Lunde & Shephard (2008):
  "Designing Realized Kernels to Measure the Ex-Post Variation of Equity
  Prices in the Presence of Noise", Econometrica.

The estimator is:
    RK = sum_{h=-H}^{H} k(h/(H+1)) * gamma_h
where gamma_h = sum_{t>|h|} r_t * r_{t-|h|} (autocovariance of returns).

Key design decisions vs. the MATLAB version:
- parameter validation is fully separated from the hot path
- bandwidth selection is a standalone function (easily unit-testable)
- the inner autocovariance loop is in a Cython extension (_core.pyx);
  if unavailable we fall back to the numpy path here
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe.realized._types import KernelType, RealizedKernelResult
from mfe.realized.noise import estimate_noise_variance
from mfe.utils.typing import FloatArray

try:
    from mfe.realized._core import _autocovariance_sum as _acov_fast  # type: ignore[import]
    _HAS_CYTHON = True
except ImportError:
    from mfe.realized._core_fallback import _autocovariance_sum as _acov_fast
    _HAS_CYTHON = False


# ---------------------------------------------------------------------------
# Kernel weight functions
# ---------------------------------------------------------------------------

def _kernel_weights(kernel_type: KernelType, H: int) -> FloatArray:
    """
    Compute kernel weights k(h/(H+1)) for h = 0, 1, ..., H.
    Returns array of length H+1 (h=0 always has weight 1.0).
    """
    h = np.arange(0, H + 1) / (H + 1)

    if kernel_type == KernelType.PARZEN:
        w = np.where(
            h <= 0.5,
            1 - 6 * h ** 2 + 6 * h ** 3,
            2 * (1 - h) ** 3,
        )
    elif kernel_type == KernelType.BARTLETT:
        w = 1 - h
    elif kernel_type == KernelType.TUKEY_HANNING:
        w = (1 + np.cos(np.pi * h)) / 2
    elif kernel_type == KernelType.CUBIC:
        w = 1 - 3 * h ** 2 + 2 * h ** 3
    elif kernel_type == KernelType.EPANECHNIKOV:
        w = 1 - h ** 2
    elif kernel_type == KernelType.FLAT_TOP:
        # Flat-top kernel: 1 for h<=0.1, linear taper to 0 at h=1
        w = np.where(h <= 0.1, 1.0, np.maximum(0.0, (1 - h) / 0.9))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return w.astype(np.float64)


# ---------------------------------------------------------------------------
# Bandwidth selection (Barndorff-Nielsen et al. 2009 optimal formula)
# ---------------------------------------------------------------------------

def select_bandwidth(
    returns: FloatArray,
    kernel_type: KernelType = KernelType.PARZEN,
    noise_variance: float | None = None,
    iq_lower_bound: float | None = None,
) -> int:
    """
    Optimal bandwidth H for the realized kernel.

    H* = c_star * xi^{4/5} * n^{3/5}

    where xi = noise_variance / sqrt(IQ), and c_star depends on the kernel.
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(r)

    # Lower bound for IQ: use tripower variation as proxy
    if iq_lower_bound is None:
        from mfe.realized.quarticity import realized_quarticity
        iq_lower_bound = realized_quarticity(r).value

    # c_star depends on kernel (from Table 1 of BNHLS 2009)
    c_star_map = {
        KernelType.PARZEN: 3.51,
        KernelType.BARTLETT: 2.16,
        KernelType.TUKEY_HANNING: 3.68,
        KernelType.CUBIC: 3.71,
        KernelType.EPANECHNIKOV: 3.28,
        KernelType.FLAT_TOP: 2.78,
    }
    c_star = c_star_map.get(kernel_type, 3.51)

    xi = noise_variance / max(iq_lower_bound ** 0.5, 1e-30)
    H = max(1, int(np.round(c_star * (xi ** 0.4) * (n ** 0.6))))

    return H


# ---------------------------------------------------------------------------
# Core autocovariance sum — numpy fallback
# ---------------------------------------------------------------------------

def _autocovariance_numpy(returns: FloatArray, H: int) -> FloatArray:
    """
    Compute gamma_h = sum_{t} r_t * r_{t-h} for h = 0, 1, ..., H.
    Returns array of length H+1.

    This is the pure-numpy fallback. The Cython version is faster by ~10x
    because it avoids allocating intermediate slices.
    """
    r = returns
    T = len(r)
    gamma = np.empty(H + 1, dtype=np.float64)
    gamma[0] = float(r @ r)
    for h in range(1, H + 1):
        gamma[h] = float(r[h:] @ r[:T - h])
    return gamma


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def realized_kernel(
    returns: FloatArray,
    kernel_type: KernelType = KernelType.PARZEN,
    bandwidth: int | None = None,
    jitter: bool = True,
) -> RealizedKernelResult:
    """
    Realized kernel estimator for quadratic variation.

    Parameters
    ----------
    returns      : (M,) log-return array (already filtered/sampled)
    kernel_type  : which kernel weight function to use
    bandwidth    : H; if None, uses automatic selector
    jitter       : if True, apply end-point jittering (noise correction)
                   as in the BNHLS paper; adds a small fraction of the
                   noise variance to handle the boundary bias

    Returns
    -------
    RealizedKernelResult
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)

    # Noise variance (needed for bandwidth selection and jitter)
    noise_var = estimate_noise_variance(r)

    if bandwidth is None:
        H = select_bandwidth(r, kernel_type=kernel_type, noise_variance=noise_var)
    else:
        H = int(bandwidth)

    if H >= n:
        warnings.warn(
            f"Bandwidth H={H} >= n={n}; clipping to n//2.",
            RuntimeWarning,
            stacklevel=2,
        )
        H = n // 2

    # Compute autocovariances
    if _HAS_CYTHON:
        gamma = _acov_fast(r, H)
    else:
        gamma = _autocovariance_numpy(r, H)

    # Kernel weights
    w = _kernel_weights(kernel_type, H)

    # RK = gamma_0 + 2 * sum_{h=1}^{H} k(h/(H+1)) * gamma_h
    rk = gamma[0] + 2.0 * float(w[1:] @ gamma[1:])

    # End-point (jitter) correction: adjusts for noise at boundaries
    # See BNHLS eq. (29); adds 2 * noise_var
    rk_adjusted = rk - 2.0 * noise_var if jitter else rk

    rk_adjusted = max(rk_adjusted, 0.0)  # enforce positivity

    return RealizedKernelResult(
        rk=rk,
        rk_adjusted=rk_adjusted,
        bandwidth=H,
        noise_variance=noise_var,
        iq_lower_bound=0.0,  # populated by caller if needed
        kernel_type=kernel_type,
        n_returns=n,
    )
