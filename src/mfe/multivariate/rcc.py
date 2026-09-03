"""
Rotated Conditional Correlation (RCC) model.

Noureldin, D., Shephard, N. & Sheppard, K. (2014): "Multivariate Rotated
ARCH Models", Journal of Econometrics, 179(1), 16-30.

Core idea
---------
Standard DCC operates on the correlation structure of standardised returns.
RCC instead:

1. Rotates raw returns by the inverse square-root of their unconditional
   covariance:  u_t = P^{-1/2} r_t  where P = E[r_t r_t']
   Under the true DGP, u_t has unconditional covariance I_K.

2. Fits a BEKK-type process to u_t u_t' in the *rotated* space:
     G_t = (I_K - A - B) + A * u_{t-1} u_{t-1}' * A + B * G_{t-1} * B
   With covariance targeting, the unconditional mean of G_t is I_K by
   construction, which removes the free intercept.

3. The DCC-type RCC sets A = a*I_K, B = b*I_K (scalar):
     G_t = (1 - a - b) I_K + a u_{t-1}u_{t-1}' + b G_{t-1}

4. Reconstructs the conditional covariance of original returns:
     Sigma_t = P^{1/2} G_t P^{1/2}

Advantages over DCC
-------------------
- Covariance targeting is exact by construction (no approximate initialisation).
- Estimation is more stable for large K: scalar RCC has only 2 free parameters
  (a, b) regardless of dimension, compared to DCC which also has 2 but its
  inner Q-bar computation can be numerically unstable.
- The rotated space has a cleaner likelihood because the rotated residuals u_t
  have unconditional identity covariance, so the step-2 likelihood simplifies.

Two-step estimation
-------------------
Step 1: Estimate P = sample covariance of returns. Compute P^{1/2} (Cholesky
        or spectral decomp). Rotate: u_t = P^{-1/2} r_t.
Step 2: Estimate scalar (a, b) by maximising the correlation log-likelihood
        using the G_t recursion.

This matches the MFE MATLAB rcc.m implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from mfe.multivariate.base import ConvergenceWarning, MultivariateVolResult
from mfe.utils.typing import FloatArray

try:
    from mfe.multivariate._core import _dcc_q_recursion  # type: ignore[import]
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def _rotation_matrix(P: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Compute P^{1/2} and P^{-1/2} via eigendecomposition (symmetric square root).

    Eigendecomposition is preferred over Cholesky here because we need the
    *symmetric* square root (not the lower-triangular one) so that
    P^{1/2} P^{1/2} = P and P^{-1/2} P P^{-1/2} = I exactly.

    Returns (P_half, P_inv_half).
    """
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-14)
    sqrt_vals = np.sqrt(eigvals)
    P_half     = eigvecs @ np.diag(sqrt_vals)     @ eigvecs.T
    P_inv_half = eigvecs @ np.diag(1.0 / sqrt_vals) @ eigvecs.T
    return P_half, P_inv_half


# ---------------------------------------------------------------------------
# G_t recursion (in rotated space) — numpy fallback
# ---------------------------------------------------------------------------

def _rcc_recursion_numpy(
    u: FloatArray,     # (T, K) rotated residuals
    a: float,
    b: float,
) -> FloatArray:
    """
    Scalar RCC recursion:
        G_t = (1-a-b)*I + a*u_{t-1}u_{t-1}' + b*G_{t-1}

    G_0 = I_K (consistent with covariance targeting).
    Returns (T, K, K) array.
    """
    T, K = u.shape
    c = 1.0 - a - b
    G = np.empty((T, K, K), dtype=np.float64)
    G[0] = np.eye(K)

    for t in range(1, T):
        outer = np.outer(u[t - 1], u[t - 1])
        G[t] = c * np.eye(K) + a * outer + b * G[t - 1]

    return G


# ---------------------------------------------------------------------------
# Correlation log-likelihood in rotated space
# ---------------------------------------------------------------------------

def _rcc_loglik(
    params: FloatArray,
    u: FloatArray,
) -> float:
    """
    Step-2 RCC log-likelihood (negative, for minimization).

    L2 = 0.5 * sum_t [log|G_t| + u_t' G_t^{-1} u_t - u_t'u_t]

    The last term u_t'u_t cancels the step-1 contribution and is kept for
    consistency with Engle (2002) DCC log-likelihood decomposition.
    """
    a, b = float(params[0]), float(params[1])
    if a < 0 or b < 0 or a + b >= 1.0:
        return 1e10

    T, K = u.shape

    if _HAS_CYTHON:
        # Re-use the DCC Q recursion — same structure with q_bar = I_K
        G = np.asarray(_dcc_q_recursion(
            np.ascontiguousarray(u, dtype=np.float64),
            np.ascontiguousarray(np.eye(K), dtype=np.float64),
            a, b,
        ))
    else:
        G = _rcc_recursion_numpy(u, a, b)

    ll = 0.0
    for t in range(T):
        sign, logdet = np.linalg.slogdet(G[t])
        if sign <= 0:
            return 1e10
        try:
            G_inv = np.linalg.inv(G[t])
        except np.linalg.LinAlgError:
            return 1e10
        quad_G = float(u[t] @ G_inv @ u[t])
        quad_I = float(u[t] @ u[t])
        ll += logdet + quad_G - quad_I

    return 0.5 * ll


# ---------------------------------------------------------------------------
# RCC result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RCCResult:
    """RCC model estimation result."""
    params: FloatArray            # (2,) [a, b] scalar RCC params
    log_likelihood: float
    conditional_covariances: FloatArray   # (T, K, K) Sigma_t in original space
    G_t: FloatArray               # (T, K, K) conditional covariance in rotated space
    u_t: FloatArray               # (T, K) rotated standardized residuals
    P: FloatArray                 # (K, K) unconditional covariance (rotation target)
    P_half: FloatArray            # (K, K) symmetric square root of P
    converged: bool
    n_obs: int
    n_vars: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def a(self) -> float:
        return float(self.params[0])

    @property
    def b(self) -> float:
        return float(self.params[1])

    @property
    def aic(self) -> float:
        return -2 * self.log_likelihood + 2 * 2

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + 2 * np.log(self.n_obs)

    def conditional_correlations(self) -> FloatArray:
        """
        Extract (T, K, K) conditional correlation matrices from Sigma_t.
        """
        T, K, _ = self.conditional_covariances.shape
        R = np.empty_like(self.conditional_covariances)
        for t in range(T):
            S = self.conditional_covariances[t]
            d = np.sqrt(np.diag(S))
            d_inv = np.where(d > 0, 1.0 / d, 0.0)
            R[t] = d_inv[:, None] * S * d_inv[None, :]
        return R


# ---------------------------------------------------------------------------
# Main RCC class
# ---------------------------------------------------------------------------

class RCC:
    """
    Rotated Conditional Correlation (RCC) model.

    Noureldin, Shephard & Sheppard (2014). Scalar parameterisation only
    (full RARCH — fully parametric A, B matrices — is left as a future extension).

    Parameters
    ----------
    rotation : "symmetric" (default) | "cholesky"
        How to compute P^{1/2}:
        "symmetric" — symmetric (spectral) square root. PSD-preserving, recommended.
        "cholesky"  — lower-triangular Cholesky. Faster but ordering-dependent.
    """

    def __init__(self, rotation: str = "symmetric") -> None:
        if rotation not in ("symmetric", "cholesky"):
            raise ValueError(f"rotation must be 'symmetric' or 'cholesky', got '{rotation}'")
        self.rotation = rotation

    def fit(
        self,
        data: FloatArray,
        starting_values: FloatArray | None = None,
        options: dict | None = None,
    ) -> RCCResult:
        """
        Estimate RCC by two-step QML.

        Parameters
        ----------
        data             : (T, K) return matrix (demeaned)
        starting_values  : (2,) [a0, b0]; if None uses [0.05, 0.90]
        options          : passed to scipy.optimize.minimize

        Returns
        -------
        RCCResult
        """
        data = np.asarray(data, dtype=np.float64)
        T, K = data.shape

        # Step 1: Rotation
        P = data.T @ data / T  # unconditional covariance

        if self.rotation == "symmetric":
            P_half, P_inv_half = _rotation_matrix(P)
        else:
            try:
                L = np.linalg.cholesky(P)
                P_half = L
                P_inv_half = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                P_half, P_inv_half = _rotation_matrix(P)

        u = data @ P_inv_half.T   # (T, K) rotated residuals; u_t = P^{-T/2} r_t
        # Verify: u.T @ u / T ≈ I_K
        # (P_inv_half.T @ P @ P_inv_half = I if symmetric, approx otherwise)

        # Step 2: Optimize (a, b) in rotated space
        x0 = np.array([0.05, 0.90]) if starting_values is None else np.asarray(starting_values)
        bounds = [(1e-6, 0.9999), (1e-6, 0.9999)]

        result = minimize(
            _rcc_loglik,
            x0,
            args=(u,),
            method="L-BFGS-B",
            bounds=bounds,
            options=options or {"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"RCC did not converge: {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        a, b = float(result.x[0]), float(result.x[1])

        # Final G_t in rotated space
        if _HAS_CYTHON:
            G_t = np.asarray(_dcc_q_recursion(
                np.ascontiguousarray(u, dtype=np.float64),
                np.ascontiguousarray(np.eye(K), dtype=np.float64),
                a, b,
            ))
        else:
            G_t = _rcc_recursion_numpy(u, a, b)

        # Reconstruct Sigma_t = P^{1/2} G_t P^{1/2}
        sigma_t = np.empty((T, K, K), dtype=np.float64)
        for t in range(T):
            sigma_t[t] = P_half @ G_t[t] @ P_half.T

        # Log-likelihood (step 1 + step 2)
        # Step 1: Gaussian log-lik with const cov P
        sign, logdet_P = np.linalg.slogdet(P)
        ll_step1 = -0.5 * T * (K * np.log(2 * np.pi) + logdet_P) - 0.5 * float(np.sum(u ** 2))
        ll_step2 = -result.fun  # stored as negative by _rcc_loglik
        ll_total = ll_step1 + ll_step2

        return RCCResult(
            params=result.x,
            log_likelihood=ll_total,
            conditional_covariances=sigma_t,
            G_t=G_t,
            u_t=u,
            P=P,
            P_half=P_half,
            converged=result.success,
            n_obs=T,
            n_vars=K,
            diagnostics={"a": a, "b": b, "rotation": self.rotation},
        )
