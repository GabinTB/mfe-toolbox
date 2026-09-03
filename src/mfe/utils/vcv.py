"""
Robust covariance matrix estimators.

- sandwich (QMLE robust, a.k.a. Huber-White)
- newey_west (HAC)
- Both accept a (T, P) score matrix and optionally a (P, P) Hessian.
"""

from __future__ import annotations

import numpy as np

from mfe.utils.typing import FloatArray


def sandwich(
    scores: FloatArray,
    hessian: FloatArray,
) -> FloatArray:
    """
    Sandwich (robust) covariance: H^{-1} B H^{-1}
    where B = scores.T @ scores / T and H is the (negative) Hessian / T.

    Parameters
    ----------
    scores  : (T, P) score matrix
    hessian : (P, P) negative Hessian evaluated at MLE

    Returns
    -------
    (P, P) robust covariance matrix
    """
    T = scores.shape[0]
    B = scores.T @ scores / T
    H_inv = np.linalg.inv(hessian / T)
    return H_inv @ B @ H_inv / T


def newey_west(
    scores: FloatArray,
    bandwidth: int | None = None,
    hessian: FloatArray | None = None,
) -> FloatArray:
    """
    Newey-West (HAC) covariance.

    Parameters
    ----------
    scores    : (T, P) score matrix
    bandwidth : number of lags; if None uses Andrews (1991) automatic selector
    hessian   : (P, P) — if provided, returns sandwich; otherwise returns B_hat only

    Returns
    -------
    (P, P) matrix: B_hat (HAC meat) if hessian is None, else full sandwich
    """
    T, P = scores.shape

    if bandwidth is None:
        bandwidth = int(np.floor(4 * (T / 100) ** (2 / 9)))

    # Newey-West weights: 1 - h/(bandwidth+1)
    B = scores.T @ scores / T
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1)
        gamma = scores[lag:].T @ scores[:T - lag] / T
        B += w * (gamma + gamma.T)

    if hessian is None:
        return B

    H_inv = np.linalg.inv(hessian / T)
    return H_inv @ B @ H_inv / T
