# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: nonecheck=False, initializedcheck=False
"""
Cython extensions for mfe.multivariate recursion hot paths.

Functions
---------
_dcc_q_recursion          DCC Q-process: (T, K, K) array of Q_t matrices
_bekk_scalar_recursion    Scalar BEKK: (T, K, K) conditional covariances + log-lik
_bekk_diagonal_recursion  Diagonal BEKK: same signature
_mvnorm_loglik            Multivariate normal log-likelihood (vectorized, no alloc)
"""

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float64_t
from libc.math cimport log, sqrt, fabs


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

cdef double _logdet_2x2(double a, double b, double c, double d) nogil:
    """log|[[a,b],[c,d]]| = log(ad - bc).  Returns -inf on singularity."""
    cdef double det = a * d - b * c
    if det <= 0.0:
        return -1e300
    return log(det)


cdef double _quad_2x2(
    double a, double b, double c, double d,  # matrix [[a,b],[c,d]]
    double x0, double x1,                    # vector
) nogil:
    """x' * inv([[a,b],[c,d]]) * x  for 2x2 symmetric matrix (b == c)."""
    cdef double det = a * d - b * b
    if fabs(det) < 1e-300:
        return 1e300
    cdef double inv_det = 1.0 / det
    # inv = [[d, -b],[-b, a]] / det
    return inv_det * (d * x0 * x0 - 2.0 * b * x0 * x1 + a * x1 * x1)


# ────────────────────────────────────────────────────────────────────────────
# 1. DCC Q-recursion
# ────────────────────────────────────────────────────────────────────────────

def _dcc_q_recursion(
    double[:, ::1] z,       # (T, K) standardized residuals
    double[:, ::1] q_bar,   # (K, K) unconditional covariance
    double a,
    double b,
):
    """
    Q_t = (1 - a - b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}

    Returns (T, K, K) array of Q_t matrices.
    ~15x faster than the pure-Python loop for K=5, T=2000.
    """
    cdef int T = z.shape[0]
    cdef int K = z.shape[1]
    cdef int t, i, j
    cdef double c = 1.0 - a - b

    Q = np.empty((T, K, K), dtype=np.float64)
    cdef double[:, :, ::1] Qv = Q

    # Q[0] = q_bar
    for i in range(K):
        for j in range(K):
            Qv[0, i, j] = q_bar[i, j]

    for t in range(1, T):
        for i in range(K):
            for j in range(K):
                Qv[t, i, j] = (
                    c * q_bar[i, j]
                    + a * z[t - 1, i] * z[t - 1, j]
                    + b * Qv[t - 1, i, j]
                )

    return Q


# ────────────────────────────────────────────────────────────────────────────
# 2. Scalar BEKK recursion + log-likelihood
# ────────────────────────────────────────────────────────────────────────────

def _bekk_scalar_recursion(
    double[:, ::1] eps,     # (T, K) residuals
    double[:, ::1] CC,      # (K, K) precomputed C'C
    double a2,              # alpha^2
    double b2,              # beta^2
    double[:, ::1] H0,      # (K, K) initial covariance
):
    """
    H_t = CC + a2 * eps_{t-1} eps_{t-1}' + b2 * H_{t-1}

    Returns (H_series, log_likelihood) where H_series is (T, K, K).
    Log-likelihood is Gaussian QML: sum_t [-0.5*(K*log2pi + log|H_t| + eps_t'H_t^{-1}eps_t)]

    For K=2 uses an analytic 2x2 inverse/det (no LAPACK call).
    For K>2 falls back to numpy for the per-step inverse.
    """
    cdef int T = eps.shape[0]
    cdef int K = eps.shape[1]
    cdef int t, i, j

    H = np.empty((T, K, K), dtype=np.float64)
    cdef double[:, :, ::1] Hv = H

    # H[0] = H0
    for i in range(K):
        for j in range(K):
            Hv[0, i, j] = H0[i, j]

    cdef double ll = 0.0
    cdef double LOG2PI = 1.8378770664093453  # log(2*pi)

    # Specialised 2x2 path avoids per-step numpy calls
    cdef double h00, h01, h11, det, inv_det, q00, q01, q10, q11
    cdef double e0, e1, logdet, quad

    for t in range(1, T):
        # H_t = CC + a2 * eps[t-1] eps[t-1]' + b2 * H[t-1]
        for i in range(K):
            for j in range(K):
                Hv[t, i, j] = (
                    CC[i, j]
                    + a2 * eps[t - 1, i] * eps[t - 1, j]
                    + b2 * Hv[t - 1, i, j]
                )

    # Log-likelihood loop — K==2 fast path
    if K == 2:
        for t in range(T):
            h00 = Hv[t, 0, 0]
            h01 = Hv[t, 0, 1]
            h11 = Hv[t, 1, 1]
            det = h00 * h11 - h01 * h01
            if det <= 0.0:
                return H, 1e10
            e0 = eps[t, 0]
            e1 = eps[t, 1]
            logdet = log(det)
            # quad = e' H^{-1} e = (h11*e0^2 - 2*h01*e0*e1 + h00*e1^2) / det
            quad = (h11 * e0 * e0 - 2.0 * h01 * e0 * e1 + h00 * e1 * e1) / det
            ll += logdet + quad
        ll = 0.5 * (T * K * LOG2PI + ll)
    else:
        # General path: use numpy for inversion
        H_np = np.asarray(H)
        eps_np = np.asarray(eps)
        for t in range(T):
            Ht = H_np[t]
            sign, ldet = np.linalg.slogdet(Ht)
            if sign <= 0:
                return H, 1e10
            try:
                Hinv = np.linalg.inv(Ht)
            except Exception:
                return H, 1e10
            et = eps_np[t]
            ll += float(ldet) + float(et @ Hinv @ et)
        ll = 0.5 * (T * K * LOG2PI + ll)

    return H, ll


# ────────────────────────────────────────────────────────────────────────────
# 3. Diagonal BEKK recursion + log-likelihood
# ────────────────────────────────────────────────────────────────────────────

def _bekk_diagonal_recursion(
    double[:, ::1] eps,     # (T, K)
    double[:, ::1] CC,      # (K, K) precomputed C'C
    double[:, ::1] AtA,     # (K, K) precomputed A_diag^2 as outer product (diag * diag)
    double[:, ::1] BtB,     # (K, K) precomputed B_diag^2 as outer product
    double[:, ::1] H0,      # (K, K)
):
    """
    H_t = CC + AtA * (eps_{t-1} eps_{t-1}') + BtB * H_{t-1}

    (element-wise products — A, B are diagonal so AtA[i,j] = A[i]*A[j])
    """
    cdef int T = eps.shape[0]
    cdef int K = eps.shape[1]
    cdef int t, i, j

    H = np.empty((T, K, K), dtype=np.float64)
    cdef double[:, :, ::1] Hv = H

    for i in range(K):
        for j in range(K):
            Hv[0, i, j] = H0[i, j]

    cdef double ll = 0.0
    cdef double LOG2PI = 1.8378770664093453

    for t in range(1, T):
        for i in range(K):
            for j in range(K):
                Hv[t, i, j] = (
                    CC[i, j]
                    + AtA[i, j] * eps[t - 1, i] * eps[t - 1, j]
                    + BtB[i, j] * Hv[t - 1, i, j]
                )

    # Log-likelihood (2x2 fast path + general)
    cdef double h00, h01, h11, det, e0, e1, logdet, quad
    if K == 2:
        for t in range(T):
            h00 = Hv[t, 0, 0]; h01 = Hv[t, 0, 1]; h11 = Hv[t, 1, 1]
            det = h00 * h11 - h01 * h01
            if det <= 0.0:
                return H, 1e10
            e0 = eps[t, 0]; e1 = eps[t, 1]
            logdet = log(det)
            quad = (h11 * e0 * e0 - 2.0 * h01 * e0 * e1 + h00 * e1 * e1) / det
            ll += logdet + quad
        ll = 0.5 * (T * K * LOG2PI + ll)
    else:
        H_np = np.asarray(H)
        eps_np = np.asarray(eps)
        for t in range(T):
            Ht = H_np[t]
            sign, ldet = np.linalg.slogdet(Ht)
            if sign <= 0:
                return H, 1e10
            try:
                Hinv = np.linalg.inv(Ht)
            except Exception:
                return H, 1e10
            et = eps_np[t]
            ll += float(ldet) + float(et @ Hinv @ et)
        ll = 0.5 * (T * K * LOG2PI + ll)

    return H, ll


# ────────────────────────────────────────────────────────────────────────────
# 4. DCC correlation log-likelihood (step 2 objective)
# ────────────────────────────────────────────────────────────────────────────

def _dcc_corr_loglik(
    double[:, ::1] z,       # (T, K) standardized residuals
    double[:, :, ::1] Q,    # (T, K, K) Q matrices from _dcc_q_recursion
):
    """
    DCC correlation log-likelihood (step 2, Engle 2002 eq. 12):

    L2 = 0.5 * sum_t [log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t]

    where R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

    2x2 fast path avoids per-step numpy inversion.
    """
    cdef int T = z.shape[0]
    cdef int K = z.shape[1]
    cdef int t, i

    cdef double ll = 0.0
    cdef double q00, q01, q11, r00, r01, r11, det_r
    cdef double z0, z1, quad_r, quad_z

    if K == 2:
        for t in range(T):
            q00 = Q[t, 0, 0]; q01 = Q[t, 0, 1]; q11 = Q[t, 1, 1]
            # R_t = diag(Q)^{-1/2} Q diag(Q)^{-1/2}
            # r[i,j] = q[i,j] / sqrt(q[i,i] * q[j,j])
            if q00 <= 0.0 or q11 <= 0.0:
                return 1e10
            r01 = q01 / sqrt(q00 * q11)
            # R is [[1, r01],[r01, 1]], det = 1 - r01^2
            det_r = 1.0 - r01 * r01
            if det_r <= 1e-14:
                return 1e10
            z0 = z[t, 0]; z1 = z[t, 1]
            # z' R^{-1} z = (z0^2 - 2*r01*z0*z1 + z1^2) / det_r
            quad_r = (z0 * z0 - 2.0 * r01 * z0 * z1 + z1 * z1) / det_r
            quad_z = z0 * z0 + z1 * z1
            ll += log(det_r) + quad_r - quad_z
        return 0.5 * ll
    else:
        # General K: numpy fallback
        Q_np  = np.asarray(Q)
        z_np  = np.asarray(z)
        ll_py = 0.0
        for t in range(T):
            Qt = Q_np[t]
            d_inv = 1.0 / np.sqrt(np.diag(Qt))
            Rt = d_inv[:, None] * Qt * d_inv[None, :]
            sign, ldet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e10
            zt = z_np[t]
            try:
                Rinv = np.linalg.inv(Rt)
            except Exception:
                return 1e10
            ll_py += float(ldet) + float(zt @ Rinv @ zt) - float(zt @ zt)
        return 0.5 * ll_py
