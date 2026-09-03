"""
BEKK-GARCH models (Engle & Kroner 1995).

Three variants:
  Scalar BEKK:   H_t = C'C + a^2 * eps_{t-1} eps_{t-1}' + b^2 * H_{t-1}
  Diagonal BEKK: H_t = C'C + A' * eps eps' * A + B' * H_{t-1} * B  (A,B diagonal)
  Full BEKK:     H_t = C'C + A' * eps eps' * A + B' * H_{t-1} * B  (A,B full K×K)

Estimation via QMLE (two-step or direct).
We implement the scalar and diagonal variants first; full BEKK is numerically
expensive (O(K^4) per recursion step) and relegated to Phase 2.

Key performance note:
- Scalar BEKK inner loop: O(K^2 * T) — fast enough in numpy for K <= 20
- Diagonal BEKK: O(K^2 * T) — same
- Full BEKK: O(K^4 * T) — needs Cython for K > 5

References
----------
Engle, R.F. & Kroner, K.F. (1995): "Multivariate Simultaneous Generalized ARCH",
Econometric Theory.

Noureldin, D., Shephard, N. & Sheppard, K. (2012): "Multivariate High-Frequency-Based
Volatility (HEAVY) Models", JoE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky

from mfe.multivariate.base import ConvergenceWarning, MultivariateVolResult
from mfe.utils.typing import FloatArray

try:
    from mfe.multivariate._core import (          # type: ignore[import]
        _bekk_scalar_recursion as _bekk_scalar_cy,
        _bekk_diagonal_recursion as _bekk_diag_cy,
    )
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


class BEKKVariant(str, Enum):
    SCALAR = "scalar"
    DIAGONAL = "diagonal"
    FULL = "full"


# ---------------------------------------------------------------------------
# BEKK recursions — numpy fallbacks
# ---------------------------------------------------------------------------

def _bekk_scalar_recursion_numpy(
    eps: FloatArray,     # (T, K) residuals
    C: FloatArray,       # (K, K) lower-triangular intercept
    a: float,
    b: float,
    H0: FloatArray,      # (K, K) initial covariance (backcast)
) -> tuple[FloatArray, float]:
    """
    Scalar BEKK recursion:
        H_t = C'C + a^2 * eps_{t-1} eps_{t-1}' + b^2 * H_{t-1}

    Returns (H_series, log_likelihood).
    H_series is (T, K, K).
    Log-likelihood is the Gaussian QML.
    """
    T, K = eps.shape
    CC = C.T @ C  # constant matrix
    a2 = a ** 2
    b2 = b ** 2

    H = np.empty((T, K, K), dtype=np.float64)
    H[0] = H0.copy()

    ll = 0.0
    for t in range(1, T):
        outer = eps[t - 1:t].T @ eps[t - 1:t]  # (K, K)
        H[t] = CC + a2 * outer + b2 * H[t - 1]

    # Log-likelihood: sum_t [-K/2 * log(2pi) - 0.5 * log|H_t| - 0.5 * eps_t' H_t^{-1} eps_t]
    for t in range(T):
        sign, logdet = np.linalg.slogdet(H[t])
        if sign <= 0:
            return H, 1e10
        try:
            H_inv = np.linalg.inv(H[t])
        except np.linalg.LinAlgError:
            return H, 1e10
        quad = float(eps[t] @ H_inv @ eps[t])
        ll += logdet + quad

    ll = 0.5 * (T * K * np.log(2 * np.pi) + ll)
    return H, ll


def _bekk_diagonal_recursion_numpy(
    eps: FloatArray,
    C: FloatArray,       # (K, K) lower-triangular
    A_diag: FloatArray,  # (K,) diagonal of A
    B_diag: FloatArray,  # (K,) diagonal of B
    H0: FloatArray,
) -> tuple[FloatArray, float]:
    """
    Diagonal BEKK recursion:
        H_t = C'C + diag(A)' * eps_{t-1} eps_{t-1}' * diag(A)
                  + diag(B)' * H_{t-1} * diag(B)
    """
    T, K = eps.shape
    CC = C.T @ C
    A = np.diag(A_diag)
    B = np.diag(B_diag)
    AtA = A.T @ A  # elementwise: A_diag^2 as outer product
    BtB = B.T @ B

    H = np.empty((T, K, K), dtype=np.float64)
    H[0] = H0.copy()

    ll = 0.0
    for t in range(1, T):
        outer = eps[t - 1:t].T @ eps[t - 1:t]
        H[t] = CC + AtA * outer + BtB * H[t - 1]

    for t in range(T):
        sign, logdet = np.linalg.slogdet(H[t])
        if sign <= 0:
            return H, 1e10
        try:
            H_inv = np.linalg.inv(H[t])
        except np.linalg.LinAlgError:
            return H, 1e10
        quad = float(eps[t] @ H_inv @ eps[t])
        ll += logdet + quad

    ll = 0.5 * (T * K * np.log(2 * np.pi) + ll)
    return H, ll


# ---------------------------------------------------------------------------
# Backcast: initial H_0 estimate
# ---------------------------------------------------------------------------

def _backcast(eps: FloatArray, weight: float = 0.06) -> FloatArray:
    """
    Exponentially weighted backcast for H_0.
    Standard approach: H_0 = sum_t lambda^t * eps_t eps_t' / sum_t lambda^t.
    """
    T, K = eps.shape
    H0 = np.zeros((K, K), dtype=np.float64)
    total_w = 0.0
    lam = 1.0 - weight
    for t in range(min(T, 100)):
        w = lam ** t
        total_w += w
        H0 += w * np.outer(eps[t], eps[t])
    return H0 / total_w


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def _pack_scalar(C: FloatArray, a: float, b: float) -> FloatArray:
    """Pack C lower-triangle + a + b into a flat vector."""
    K = C.shape[0]
    c_vec = C[np.tril_indices(K)]
    return np.concatenate([c_vec, [a, b]])


def _unpack_scalar(params: FloatArray, K: int) -> tuple[FloatArray, float, float]:
    n_c = K * (K + 1) // 2
    C = np.zeros((K, K), dtype=np.float64)
    C[np.tril_indices(K)] = params[:n_c]
    a = float(params[n_c])
    b = float(params[n_c + 1])
    return C, a, b


def _pack_diagonal(C: FloatArray, A_diag: FloatArray, B_diag: FloatArray) -> FloatArray:
    K = C.shape[0]
    c_vec = C[np.tril_indices(K)]
    return np.concatenate([c_vec, A_diag, B_diag])


def _unpack_diagonal(params: FloatArray, K: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_c = K * (K + 1) // 2
    C = np.zeros((K, K), dtype=np.float64)
    C[np.tril_indices(K)] = params[:n_c]
    A_diag = params[n_c : n_c + K]
    B_diag = params[n_c + K : n_c + 2 * K]
    return C, A_diag, B_diag


# ---------------------------------------------------------------------------
# Scalar starting values: target-based initialisation
# ---------------------------------------------------------------------------

def _scalar_starting_values(eps: FloatArray) -> FloatArray:
    """
    Starting values using covariance targeting.
    C is set so that C'C = (1 - a0^2 - b0^2) * sample_cov.
    """
    K = eps.shape[1]
    a0, b0 = 0.10, 0.85
    S = eps.T @ eps / len(eps)
    scale = 1.0 - a0 ** 2 - b0 ** 2
    try:
        C0 = np.linalg.cholesky(scale * S).T  # upper triangular
        C0 = C0.T  # lower triangular
    except np.linalg.LinAlgError:
        C0 = np.eye(K) * np.sqrt(scale * np.mean(np.diag(S)))
    return _pack_scalar(C0, a0, b0)


def _diagonal_starting_values(eps: FloatArray) -> FloatArray:
    K = eps.shape[1]
    a0_diag = np.full(K, 0.10)
    b0_diag = np.full(K, 0.85)
    S = eps.T @ eps / len(eps)
    scale = 1.0 - 0.10 ** 2 - 0.85 ** 2
    try:
        C0 = np.linalg.cholesky(scale * S).T.T
    except np.linalg.LinAlgError:
        C0 = np.eye(K) * np.sqrt(scale * np.mean(np.diag(S)))
    return _pack_diagonal(C0, a0_diag, b0_diag)


# ---------------------------------------------------------------------------
# Main BEKK class
# ---------------------------------------------------------------------------

class BEKK:
    """
    BEKK-GARCH model (Engle & Kroner 1995).

    Parameters
    ----------
    variant : "scalar" | "diagonal" | "full"
        "full" is not yet implemented.
    """

    def __init__(self, variant: str | BEKKVariant = BEKKVariant.SCALAR) -> None:
        self.variant = BEKKVariant(variant)
        if self.variant == BEKKVariant.FULL:
            raise NotImplementedError(
                "Full BEKK is on the Phase 2 roadmap. "
                "Use variant='scalar' or 'diagonal'."
            )

    def fit(
        self,
        data: FloatArray,
        starting_values: FloatArray | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> MultivariateVolResult:
        """
        Estimate BEKK parameters via QMLE.

        Parameters
        ----------
        data : (T, K) return matrix (demeaned)
        """
        eps = np.asarray(data, dtype=np.float64)
        T, K = eps.shape
        H0 = _backcast(eps)

        if self.variant == BEKKVariant.SCALAR:
            return self._fit_scalar(eps, K, T, H0, starting_values, method, options)
        else:
            return self._fit_diagonal(eps, K, T, H0, starting_values, method, options)

    def _fit_scalar(self, eps, K, T, H0, sv, method, options):
        if sv is None:
            x0 = _scalar_starting_values(eps)
        else:
            x0 = np.asarray(sv, dtype=np.float64)

        n_c = K * (K + 1) // 2
        bounds = [(None, None)] * n_c + [(1e-6, 0.9999), (1e-6, 0.9999)]

        def neg_ll(params):
            C, a, b = _unpack_scalar(params, K)
            if a + b >= 1.0:
                return 1e10
            CC = C.T @ C
            a2, b2 = a ** 2, b ** 2
            if _HAS_CYTHON:
                _, ll = _bekk_scalar_cy(
                    np.ascontiguousarray(eps, dtype=np.float64),
                    np.ascontiguousarray(CC, dtype=np.float64),
                    a2, b2,
                    np.ascontiguousarray(H0, dtype=np.float64),
                )
            else:
                _, ll = _bekk_scalar_recursion_numpy(eps, C, a, b, H0)
            return ll

        result = minimize(
            neg_ll, x0, method=method,
            bounds=bounds,
            options=options or {"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"Scalar BEKK did not converge: {result.message}",
                ConvergenceWarning, stacklevel=3,
            )

        C, a, b = _unpack_scalar(result.x, K)
        CC = C.T @ C
        if _HAS_CYTHON:
            H_t, _ = _bekk_scalar_cy(
                np.ascontiguousarray(eps, dtype=np.float64),
                np.ascontiguousarray(CC, dtype=np.float64),
                a ** 2, b ** 2,
                np.ascontiguousarray(H0, dtype=np.float64),
            )
        else:
            H_t, _ = _bekk_scalar_recursion_numpy(eps, C, a, b, H0)
        P = len(result.x)

        return MultivariateVolResult(
            params=result.x,
            log_likelihood=-result.fun,
            conditional_covariances=H_t,
            residuals=eps,
            vcv=np.full((P, P), np.nan),
            vcv_robust=np.full((P, P), np.nan),
            scores=np.zeros((T, P)),
            converged=result.success,
            n_obs=T,
            n_params=P,
            model_name=f"Scalar BEKK-GARCH(1,1), K={K}",
            diagnostics={"a": a, "b": b, "C": C},
        )

    def _fit_diagonal(self, eps, K, T, H0, sv, method, options):
        if sv is None:
            x0 = _diagonal_starting_values(eps)
        else:
            x0 = np.asarray(sv, dtype=np.float64)

        n_c = K * (K + 1) // 2
        bounds = [(None, None)] * n_c + [(-0.9999, 0.9999)] * K + [(-0.9999, 0.9999)] * K

        def neg_ll(params):
            C, A_diag, B_diag = _unpack_diagonal(params, K)
            if np.any(A_diag ** 2 + B_diag ** 2 >= 1.0):
                return 1e10
            CC = C.T @ C
            AtA = np.outer(A_diag, A_diag)
            BtB = np.outer(B_diag, B_diag)
            if _HAS_CYTHON:
                _, ll = _bekk_diag_cy(
                    np.ascontiguousarray(eps, dtype=np.float64),
                    np.ascontiguousarray(CC, dtype=np.float64),
                    np.ascontiguousarray(AtA, dtype=np.float64),
                    np.ascontiguousarray(BtB, dtype=np.float64),
                    np.ascontiguousarray(H0, dtype=np.float64),
                )
            else:
                _, ll = _bekk_diagonal_recursion_numpy(eps, C, A_diag, B_diag, H0)
            return ll

        result = minimize(
            neg_ll, x0, method=method,
            bounds=bounds,
            options=options or {"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"Diagonal BEKK did not converge: {result.message}",
                ConvergenceWarning, stacklevel=3,
            )

        C, A_diag, B_diag = _unpack_diagonal(result.x, K)
        CC  = C.T @ C
        AtA = np.outer(A_diag, A_diag)
        BtB = np.outer(B_diag, B_diag)
        if _HAS_CYTHON:
            H_t, _ = _bekk_diag_cy(
                np.ascontiguousarray(eps, dtype=np.float64),
                np.ascontiguousarray(CC, dtype=np.float64),
                np.ascontiguousarray(AtA, dtype=np.float64),
                np.ascontiguousarray(BtB, dtype=np.float64),
                np.ascontiguousarray(H0, dtype=np.float64),
            )
        else:
            H_t, _ = _bekk_diagonal_recursion_numpy(eps, C, A_diag, B_diag, H0)
        P = len(result.x)

        return MultivariateVolResult(
            params=result.x,
            log_likelihood=-result.fun,
            conditional_covariances=H_t,
            residuals=eps,
            vcv=np.full((P, P), np.nan),
            vcv_robust=np.full((P, P), np.nan),
            scores=np.zeros((T, P)),
            converged=result.success,
            n_obs=T,
            n_params=P,
            model_name=f"Diagonal BEKK-GARCH(1,1), K={K}",
            diagnostics={"A_diag": A_diag, "B_diag": B_diag, "C": C},
        )
