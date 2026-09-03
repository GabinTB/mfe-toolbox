"""
Multivariate Realized Kernel.

Barndorff-Nielsen, Hansen, Lunde & Shephard (2011): "Multivariate Realised
Kernels: Consistent Positive Semi-Definite Estimators of the Covariation of
Equity Prices with Noise and Non-Synchronous Trading",
Journal of Econometrics, 162(2), 149-169.

The multivariate realized kernel generalizes the univariate realized kernel
to estimate the entire (K, K) covariance matrix simultaneously from
synchronised returns (after refresh-time or calendar-time sampling).

    RK_{ij} = sum_{h=-H}^{H} k(h/(H+1)) * Gamma_{h,ij}

where Gamma_{h,ij} = sum_{t>|h|} r_{i,t} * r_{j,t-|h|} is the cross-autocovariance.

The key property: the full (K, K) matrix is positive semi-definite by construction
because all kernels are symmetric and the weight function k satisfies k(0)=1 and
the matrix {Gamma_h} has a positive-semidefinite kernel-weighted combination.

Contrast with pairwise HY:
  HY applied to each (i,j) pair is not guaranteed to give a PSD matrix.
  The multivariate realized kernel IS PSD.

Computational note:
  The inner loop is O(H * T * K^2). For K=10, T=50K, H=30: ~1.5B ops.
  The Cython _core.pyx extension covers the K=2 case analytically; for
  general K we use numpy's matmul broadcasting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mfe.realized.kernel import _kernel_weights, select_bandwidth
from mfe.realized._types import KernelType
from mfe.realized.noise import estimate_noise_variance
from mfe.utils.typing import FloatArray


@dataclass
class MultivariateKernelResult:
    rk: FloatArray              # (K, K) realized kernel covariance matrix
    rk_adjusted: FloatArray     # (K, K) end-point corrected version
    bandwidth: int              # H used
    noise_variances: FloatArray  # (K,) diagonal noise variances
    kernel_type: KernelType
    n_returns: int
    n_vars: int


def realized_multivariate_kernel(
    returns: FloatArray,
    kernel_type: KernelType = KernelType.PARZEN,
    bandwidth: int | None = None,
    jitter: bool = True,
) -> MultivariateKernelResult:
    """
    Multivariate realized kernel for the full (K, K) covariance matrix.

    Parameters
    ----------
    returns     : (T, K) synchronized log-return matrix
    kernel_type : kernel weight function
    bandwidth   : H; if None, uses average of per-asset optimal bandwidths
    jitter      : if True, apply end-point noise correction (subtract
                  2 * diag(noise_var) from the diagonal)

    Returns
    -------
    MultivariateKernelResult
        .rk          — raw (K, K) realized kernel matrix
        .rk_adjusted — noise-corrected version (PSD enforced)
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim == 1:
        raise ValueError("returns must be (T, K) with K >= 2 for multivariate kernel")
    T, K = R.shape

    # Per-asset noise variances and optimal bandwidths
    noise_vars = np.array([estimate_noise_variance(R[:, k]) for k in range(K)])

    if bandwidth is None:
        H_per_asset = [select_bandwidth(R[:, k], kernel_type=kernel_type,
                                        noise_variance=noise_vars[k]) for k in range(K)]
        H = max(1, int(np.round(np.mean(H_per_asset))))
    else:
        H = int(bandwidth)

    if H >= T:
        warnings.warn(f"Bandwidth H={H} >= T={T}; clipping to T//2.", RuntimeWarning, stacklevel=2)
        H = T // 2

    # Kernel weights
    w = _kernel_weights(kernel_type, H)   # (H+1,)

    # Cross-autocovariance matrices Gamma_h = R[h:].T @ R[:T-h]  for h=0..H
    # Stack: (H+1, K, K) then weight-sum
    RK = np.zeros((K, K), dtype=np.float64)

    # h = 0: R.T @ R
    RK += w[0] * (R.T @ R)

    # h = 1..H: w[h] * (Gamma_h + Gamma_h.T) — symmetric kernel
    for h in range(1, H + 1):
        Gamma_h = R[h:].T @ R[:T - h]   # (K, K)
        RK += w[h] * (Gamma_h + Gamma_h.T)

    # Jitter correction: subtract 2 * diag(noise_var) per asset
    if jitter:
        noise_correction = np.diag(2.0 * noise_vars)
        rk_adj = RK - noise_correction
    else:
        rk_adj = RK.copy()

    # Enforce PSD: project onto cone of PSD matrices
    eigvals, eigvecs = np.linalg.eigh(rk_adj)
    if np.any(eigvals < 0):
        eigvals_clipped = np.maximum(eigvals, 0.0)
        rk_adj = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    return MultivariateKernelResult(
        rk=RK,
        rk_adjusted=rk_adj,
        bandwidth=H,
        noise_variances=noise_vars,
        kernel_type=kernel_type,
        n_returns=T,
        n_vars=K,
    )
