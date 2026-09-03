"""
Multivariate normal log-likelihood and related utilities.

MFE MATLAB mvnormloglik.m equivalent. Used internally in multivariate
GARCH estimation but exposed here as a clean public function because
it comes up constantly in empirical work.

Also provides:
- mvnorm_loglik     — exact Gaussian log-likelihood for a given Sigma_t sequence
- mvnorm_qmle       — QMLE with a fixed Sigma_t from any multivariate model
- standardize_mvn   — extract Mahalanobis residuals from (T, K, K) covariances
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfe.utils.typing import FloatArray


_LOG2PI = np.log(2 * np.pi)


def mvnorm_loglik(
    data: FloatArray,
    sigma_t: FloatArray,
) -> float:
    """
    Gaussian multivariate log-likelihood for a time-varying covariance sequence.

    L = sum_t [ -K/2 * log(2pi) - 0.5 * log|Sigma_t| - 0.5 * x_t' Sigma_t^{-1} x_t ]

    Parameters
    ----------
    data    : (T, K) return matrix (demeaned)
    sigma_t : (T, K, K) or (K, K) conditional covariance matrices
              If (K, K), treated as a constant covariance

    Returns
    -------
    float — total log-likelihood
    """
    data = np.asarray(data, dtype=np.float64)
    sigma_t = np.asarray(sigma_t, dtype=np.float64)
    T, K = data.shape

    if sigma_t.ndim == 2:
        # Constant covariance
        sigma_t = np.broadcast_to(sigma_t[None], (T, K, K))

    ll = 0.0
    const = -0.5 * K * _LOG2PI

    for t in range(T):
        S = sigma_t[t]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return -np.inf
        try:
            quad = float(data[t] @ np.linalg.solve(S, data[t]))
        except np.linalg.LinAlgError:
            return -np.inf
        ll += const - 0.5 * (logdet + quad)

    return ll


def mvnorm_loglik_t(
    data_t: FloatArray,
    sigma: FloatArray,
) -> float:
    """
    Single-observation Gaussian log-likelihood.

    Parameters
    ----------
    data_t : (K,) single observation
    sigma  : (K, K) covariance matrix

    Returns
    -------
    float — log-likelihood of this observation
    """
    K = len(data_t)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return -np.inf
    try:
        quad = float(data_t @ np.linalg.solve(sigma, data_t))
    except np.linalg.LinAlgError:
        return -np.inf
    return -0.5 * (K * _LOG2PI + logdet + quad)


def mahalanobis(
    data: FloatArray,
    sigma_t: FloatArray,
) -> FloatArray:
    """
    Mahalanobis distances from the conditional mean.

    d_t = sqrt( x_t' Sigma_t^{-1} x_t )

    Parameters
    ----------
    data    : (T, K)
    sigma_t : (T, K, K) or (K, K)

    Returns
    -------
    (T,) array of Mahalanobis distances
    """
    data = np.asarray(data, dtype=np.float64)
    sigma_t = np.asarray(sigma_t, dtype=np.float64)
    T, K = data.shape

    if sigma_t.ndim == 2:
        sigma_t = np.broadcast_to(sigma_t[None], (T, K, K))

    d = np.empty(T, dtype=np.float64)
    for t in range(T):
        try:
            q = float(data[t] @ np.linalg.solve(sigma_t[t], data[t]))
        except np.linalg.LinAlgError:
            q = np.nan
        d[t] = np.sqrt(max(q, 0.0))

    return d


def standardize_mvn(
    data: FloatArray,
    sigma_t: FloatArray,
) -> FloatArray:
    """
    Extract standardized residuals: z_t = L_t^{-1} x_t where L_t L_t' = Sigma_t.

    Useful for diagnostic checking: if the model is correct, z_t should be
    approximately i.i.d. N(0, I_K).

    Parameters
    ----------
    data    : (T, K) return matrix
    sigma_t : (T, K, K) conditional covariance matrices

    Returns
    -------
    (T, K) standardized residuals
    """
    data = np.asarray(data, dtype=np.float64)
    sigma_t = np.asarray(sigma_t, dtype=np.float64)
    T, K = data.shape

    if sigma_t.ndim == 2:
        sigma_t = np.broadcast_to(sigma_t[None], (T, K, K))

    z = np.empty((T, K), dtype=np.float64)
    for t in range(T):
        try:
            L = np.linalg.cholesky(sigma_t[t])
            z[t] = np.linalg.solve(L, data[t])
        except np.linalg.LinAlgError:
            z[t] = np.nan
    return z
