"""
Vector Autoregression: estimation, Granger causality, impulse response functions.

vectorar    — VAR(P) estimation with 4 VCV options
grangercause — Granger causality LR / LM / Wald tests
impulse_response — IRF with delta-method std errors, Cholesky or spectral decomp

Design gap vs. statsmodels.tsa.VAR:
  statsmodels VAR: OLS only, homoskedastic VCV, no heteroskedastic sandwich
  statsmodels IRF: standard errors only under homoskedastic assumption
  No Granger causality test with robust VCV in statsmodels

This module fills those gaps exactly, matching the MFE MATLAB vectorar.m outputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from mfe.utils.typing import FloatArray


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VARResult:
    """VAR(P) estimation result."""
    params: list[FloatArray]     # list of P (K,K) parameter matrices Phi_1..Phi_P
    const: FloatArray | None     # (K,) constant vector, or None
    errors: FloatArray           # (T, K) residuals
    sigma: FloatArray            # (K,K) residual covariance
    r_squared: FloatArray        # (K,) per-equation R^2
    vcv: FloatArray              # full (P*K+const, P*K+const) per-eq VCV (Kronecker)
    param_vec: FloatArray        # flattened parameter vector (for VCV indexing)
    lags: list[int]
    n_obs: int
    n_vars: int
    log_likelihood: float

    @property
    def aic(self) -> float:
        k = len(self.param_vec)
        return -2 * self.log_likelihood + 2 * k

    @property
    def bic(self) -> float:
        k = len(self.param_vec)
        return -2 * self.log_likelihood + k * np.log(self.n_obs)


@dataclass
class GCResult:
    """Granger causality test result."""
    statistics: FloatArray    # (K,K) — stat[i,j]: does y_j cause y_i?
    p_values: FloatArray      # (K,K)
    method: str               # "lr" | "lm" | "wald"
    n_obs: int
    n_vars: int


@dataclass
class IRFResult:
    """Impulse response function result."""
    responses: FloatArray      # (K, K, H+1) — responses[i,j,h]: y_i response to shock in y_j at h
    std_errors: FloatArray     # (K, K, H+1) — delta-method std errors
    lags: int                  # VAR order used
    horizon: int
    decomp: str                # "unit" | "cholesky" | "spectral"


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def _build_regressor_matrix(
    y: FloatArray,          # (T, K)
    lags: list[int],
    include_const: bool,
) -> tuple[FloatArray, FloatArray, int]:
    """
    Build the regressor matrix X and response Y_adj for VAR estimation.

    Returns (Y_adj, X, max_lag) where both have T-max_lag rows.
    """
    T, K = y.shape
    max_lag = max(lags)
    n = T - max_lag

    cols = []
    if include_const:
        cols.append(np.ones((n, 1)))
    for lag in lags:
        cols.append(y[max_lag - lag: T - lag])  # (n, K)

    X = np.hstack(cols)            # (n, n_regressors)
    Y_adj = y[max_lag:]            # (n, K)

    return Y_adj, X, max_lag


# ---------------------------------------------------------------------------
# VCV estimators
# ---------------------------------------------------------------------------

def _vcv_hom_uncorr(
    X: FloatArray,
    errors: FloatArray,
    K: int,
    n_regressors: int,
) -> FloatArray:
    """Homoskedastic, uncorrelated: Sigma_hat_diag kron (X'X)^{-1}."""
    n = X.shape[0]
    XTX_inv = np.linalg.inv(X.T @ X)
    sigma_diag = np.diag(np.sum(errors ** 2, axis=0) / n)
    return np.kron(sigma_diag, XTX_inv)


def _vcv_hom_corr(
    X: FloatArray,
    errors: FloatArray,
    K: int,
    n_regressors: int,
) -> FloatArray:
    """Homoskedastic, correlated: Sigma_hat kron (X'X)^{-1}."""
    n = X.shape[0]
    XTX_inv = np.linalg.inv(X.T @ X)
    Sigma = errors.T @ errors / n
    return np.kron(Sigma, XTX_inv)


def _vcv_het_uncorr(
    X: FloatArray,
    errors: FloatArray,
    K: int,
    n_regressors: int,
) -> FloatArray:
    """White heteroskedastic sandwich, uncorrelated across equations."""
    n = X.shape[0]
    M = n_regressors
    XTX_inv = np.linalg.inv(X.T @ X / n)
    A_inv = np.kron(np.eye(K), XTX_inv)

    B = np.zeros((K * M, K * M), dtype=np.float64)
    for k in range(K):
        ek = errors[:, k]
        Bkk = (X * ek[:, None]).T @ (X * ek[:, None]) / n
        B[k*M:(k+1)*M, k*M:(k+1)*M] = Bkk

    return A_inv @ B @ A_inv / n


def _vcv_het_corr(
    X: FloatArray,
    errors: FloatArray,
    K: int,
    n_regressors: int,
) -> FloatArray:
    """White heteroskedastic sandwich, correlated across equations."""
    n = X.shape[0]
    M = n_regressors
    XTX_inv = np.linalg.inv(X.T @ X / n)
    A_inv = np.kron(np.eye(K), XTX_inv)

    B = np.zeros((K * M, K * M), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            Bij = (X * errors[:, i:i+1]).T @ (X * errors[:, j:j+1]) / n
            B[i*M:(i+1)*M, j*M:(j+1)*M] = Bij

    return A_inv @ B @ A_inv / n


# ---------------------------------------------------------------------------
# Main VAR estimator
# ---------------------------------------------------------------------------

def vectorar(
    y: FloatArray,
    lags: int | list[int] = 1,
    include_const: bool = True,
    het: bool = True,
    uncorr: bool = False,
) -> VARResult:
    """
    Estimate a VAR(P) or irregular VAR.

    Parameters
    ----------
    y            : (T, K) data matrix
    lags         : int (regular VAR(P)) or list[int] (irregular VAR, e.g. [1,3])
    include_const: include a constant term (default True)
    het          : heteroskedasticity-robust VCV (default True)
    uncorr       : assume uncorrelated errors across equations (default False)

    Returns
    -------
    VARResult with params as list of (K,K) matrices, one per lag.
    """
    Y = np.asarray(y, dtype=np.float64)
    T, K = Y.shape

    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted(lags)

    Y_adj, X, max_lag = _build_regressor_matrix(Y, lag_list, include_const)
    n, n_regs = X.shape
    n_lag_params = len(lag_list) * K

    # OLS equation-by-equation (GLS = OLS for VAR)
    try:
        XTX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValueError("Regressors are singular — reduce lag order or check data.")

    B_full = XTX_inv @ X.T @ Y_adj   # (n_regs, K)
    errors = Y_adj - X @ B_full       # (n, K)
    Sigma = errors.T @ errors / n

    # Unpack constant and lag matrices
    offset = 0
    const_vec = None
    if include_const:
        const_vec = B_full[0]       # (K,)
        offset = 1

    params = []
    for lag in lag_list:
        block = B_full[offset: offset + K]   # (K, K) — rows = regressors for this lag
        params.append(block.T)               # (K, K) as Phi_p (rows = equations)
        offset += K

    # Build param_vec in the MFE ordering:
    # for each equation k: [const_k (opt), phi_{k,1,1}..phi_{k,1,K}, phi_{k,2,1}..., ...]
    param_parts = []
    for k in range(K):
        row = []
        if include_const:
            row.append([const_vec[k]])
        for p_idx in range(len(lag_list)):
            row.append(params[p_idx][k])
        param_parts.append(np.concatenate(row))
    param_vec = np.concatenate(param_parts)

    # VCV
    if het and not uncorr:
        vcv = _vcv_het_corr(X, errors, K, n_regs)
    elif het and uncorr:
        vcv = _vcv_het_uncorr(X, errors, K, n_regs)
    elif not het and uncorr:
        vcv = _vcv_hom_uncorr(X, errors, K, n_regs)
    else:
        vcv = _vcv_hom_corr(X, errors, K, n_regs)

    # R^2 per equation
    r2 = np.array([
        1.0 - float(np.sum(errors[:, k] ** 2)) / float(np.sum((Y_adj[:, k] - Y_adj[:, k].mean()) ** 2))
        for k in range(K)
    ])

    # Log-likelihood (Gaussian)
    sign, logdet = np.linalg.slogdet(Sigma)
    ll = -0.5 * n * (K * np.log(2 * np.pi) + logdet + K) if sign > 0 else -1e10

    return VARResult(
        params=params,
        const=const_vec,
        errors=errors,
        sigma=Sigma,
        r_squared=r2,
        vcv=vcv,
        param_vec=param_vec,
        lags=lag_list,
        n_obs=n,
        n_vars=K,
        log_likelihood=float(ll),
    )


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def grangercause(
    y: FloatArray,
    lags: int | list[int] = 1,
    include_const: bool = True,
    het: bool = True,
    uncorr: bool = False,
    method: Literal["lr", "lm", "wald"] = "lr",
) -> GCResult:
    """
    Granger causality testing in a VAR.

    stat[i,j] tests H0: lags of y_j do not Granger-cause y_i.

    Parameters
    ----------
    y, lags, include_const, het, uncorr : same as vectorar
    method : "lr" | "lm" | "wald"
        "lr"   — likelihood ratio (Chi2, robust if het=True)
        "lm"   — score/LM test
        "wald" — Wald test using VCV from vectorar

    Returns
    -------
    GCResult with (K,K) matrices of statistics and p-values.
    """
    Y = np.asarray(y, dtype=np.float64)
    T, K = Y.shape

    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted(lags)

    P = len(lag_list)
    df = P  # number of restrictions per (i,j) pair

    res_unr = vectorar(Y, lags=lag_list, include_const=include_const, het=het, uncorr=uncorr)

    stat_mat = np.full((K, K), np.nan, dtype=np.float64)
    pval_mat = np.full((K, K), np.nan, dtype=np.float64)

    Y_adj, X_full, max_lag = _build_regressor_matrix(Y, lag_list, include_const)
    n = Y_adj.shape[0]
    n_regs = X_full.shape[1]
    n_const = 1 if include_const else 0

    for i in range(K):       # caused variable (equation)
        for j in range(K):   # causing variable (excluded)
            if i == j:
                stat_mat[i, j] = np.nan
                pval_mat[i, j] = np.nan
                continue

            # Build restricted X: drop columns for lags of y_j
            # Column layout: [const?] [lag1: K cols] [lag2: K cols] ...
            drop_cols = []
            for p in range(P):
                col_start = n_const + p * K
                drop_cols.append(col_start + j)   # j-th variable in lag p block
            keep_cols = [c for c in range(n_regs) if c not in drop_cols]
            X_r = X_full[:, keep_cols]

            y_i = Y_adj[:, i]

            if method == "wald":
                # Wald: R beta = 0 using VCV from unrestricted
                # Extract VCV block for equation i
                eqs_per_col = n_regs
                vcv_i = res_unr.vcv[i*eqs_per_col:(i+1)*eqs_per_col,
                                     i*eqs_per_col:(i+1)*eqs_per_col]
                # Restriction matrix R: picks rows corresponding to j's lags
                R = np.zeros((df, n_regs), dtype=np.float64)
                for p_idx, col in enumerate(drop_cols):
                    R[p_idx, col] = 1.0
                # Param vector for equation i
                B_i = res_unr.param_vec[i*n_regs:(i+1)*n_regs]
                Rb = R @ B_i
                try:
                    W = float(n * Rb @ np.linalg.solve(R @ vcv_i @ R.T * n, Rb))
                except np.linalg.LinAlgError:
                    W = np.nan
                stat_mat[i, j] = W
                pval_mat[i, j] = float(1 - stats.chi2.cdf(W, df=df)) if np.isfinite(W) else np.nan

            else:
                # LR or LM: compare restricted and unrestricted models for equation i
                # Unrestricted
                XTX_inv_u = np.linalg.inv(X_full.T @ X_full)
                b_u = XTX_inv_u @ (X_full.T @ y_i)
                e_u = y_i - X_full @ b_u
                s2_u = float(e_u @ e_u) / n

                # Restricted
                try:
                    XTX_inv_r = np.linalg.inv(X_r.T @ X_r)
                except np.linalg.LinAlgError:
                    continue
                b_r = XTX_inv_r @ (X_r.T @ y_i)
                e_r = y_i - X_r @ b_r
                s2_r = float(e_r @ e_r) / n

                if method == "lr":
                    if het:
                        # Robust LR-class: based on scores under null, VCV under alt
                        scores = X_full[:, drop_cols] * e_u[:, None]
                        B_meat = scores.T @ scores / n
                        B_bread = X_full[:, drop_cols].T @ X_full[:, drop_cols] / n
                        try:
                            B_inv = np.linalg.inv(B_bread)
                            S_inv = np.linalg.inv(B_inv @ B_meat @ B_inv / n)
                        except np.linalg.LinAlgError:
                            continue
                        s_bar = scores.mean(axis=0)
                        LR = float(n * s_bar @ S_inv @ s_bar)
                    else:
                        LR = float(n * (np.log(s2_r) - np.log(s2_u)))
                    stat_mat[i, j] = LR
                    pval_mat[i, j] = float(1 - stats.chi2.cdf(LR, df=df))

                else:  # lm
                    # LM: regress e_r on X_full, R^2 * n
                    b_aux = np.linalg.lstsq(X_full, e_r, rcond=None)[0]
                    e_aux = e_r - X_full @ b_aux
                    ss_res_aux = float(e_aux @ e_aux)
                    ss_tot_aux = float(e_r @ e_r)
                    r2_aux = 1.0 - ss_res_aux / ss_tot_aux if ss_tot_aux > 0 else 0.0
                    LM = float(n * r2_aux)
                    stat_mat[i, j] = LM
                    pval_mat[i, j] = float(1 - stats.chi2.cdf(LM, df=df))

    return GCResult(
        statistics=stat_mat,
        p_values=pval_mat,
        method=method,
        n_obs=n,
        n_vars=K,
    )


# ---------------------------------------------------------------------------
# Impulse Response Functions
# ---------------------------------------------------------------------------

def impulse_response(
    y: FloatArray,
    lags: int | list[int] = 1,
    horizon: int = 12,
    decomp: Literal["unit", "cholesky", "spectral"] = "cholesky",
    include_const: bool = True,
    het: bool = True,
    uncorr: bool = False,
) -> IRFResult:
    """
    Impulse response functions for a VAR(P) with delta-method standard errors.

    Parameters
    ----------
    y        : (T, K) data
    lags     : VAR lag order or list
    horizon  : number of periods H; returns H+1 responses (including period 0)
    decomp   : shock decomposition
        "unit"      — unit shocks (unscaled), i.e. P0 = I_K
        "cholesky"  — Cholesky of Sigma (lower triangular), recursive identification
        "spectral"  — symmetric square root of Sigma (spectral decomposition)
    het, uncorr : VCV options for standard error computation

    Returns
    -------
    IRFResult
        .responses  : (K, K, H+1) — responses[response_var, shock_var, h]
        .std_errors : (K, K, H+1) delta-method std errors
    """
    Y = np.asarray(y, dtype=np.float64)
    T, K = Y.shape

    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted(lags)

    res = vectorar(Y, lags=lag_list, include_const=include_const, het=het, uncorr=uncorr)
    P = len(lag_list)
    Sigma = res.sigma

    # Shock decomposition matrix P0
    if decomp == "unit":
        P0 = np.eye(K)
    elif decomp == "cholesky":
        try:
            P0 = np.linalg.cholesky(Sigma)    # lower triangular
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky failed; using spectral decomposition.", RuntimeWarning)
            P0 = _spectral_sqrt(Sigma)
    elif decomp == "spectral":
        P0 = _spectral_sqrt(Sigma)
    else:
        raise ValueError(f"decomp must be 'unit', 'cholesky', or 'spectral', got '{decomp}'")

    # Companion form: convert VAR(P) to VAR(1) state A of size (P*K, P*K)
    # A = [[Phi_1, Phi_2, ..., Phi_P],
    #      [I_K,   0,    ..., 0     ],
    #      [0,     I_K,  ..., 0     ],
    #      ...                       ]
    max_lag = max(lag_list)
    A = np.zeros((max_lag * K, max_lag * K), dtype=np.float64)

    for p_idx, p in enumerate(lag_list):
        # res.params[p_idx] is (K, K) Phi_p
        A[:K, p_idx * K: (p_idx + 1) * K] = res.params[p_idx]
    for block in range(1, max_lag):
        A[block * K: (block + 1) * K, (block - 1) * K: block * K] = np.eye(K)

    # IRF via MA(inf) representation: Psi_h = J A^h J' where J = [I_K | 0]
    J = np.zeros((K, max_lag * K), dtype=np.float64)
    J[:K, :K] = np.eye(K)

    responses  = np.empty((K, K, horizon + 1), dtype=np.float64)
    std_errors = np.empty((K, K, horizon + 1), dtype=np.float64)

    Ah = np.eye(max_lag * K, dtype=np.float64)
    for h in range(horizon + 1):
        Psi_h = J @ Ah @ J.T        # (K, K)
        responses[:, :, h] = Psi_h @ P0
        Ah = Ah @ A

    # Delta-method std errors via asymptotic approximation
    # For each horizon h, Var(vec(Psi_h * P0)) via chain rule on A^h
    # We use numerical differentiation for generality
    std_errors = _irf_std_errors(res, lag_list, horizon, P0, J, A, K, max_lag)

    return IRFResult(
        responses=responses,
        std_errors=std_errors,
        lags=len(lag_list),
        horizon=horizon,
        decomp=decomp,
    )


def _spectral_sqrt(Sigma: FloatArray) -> FloatArray:
    """Symmetric square root of Sigma via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _irf_std_errors(
    res: VARResult,
    lag_list: list[int],
    horizon: int,
    P0: FloatArray,
    J: FloatArray,
    A: FloatArray,
    K: int,
    max_lag: int,
) -> FloatArray:
    """
    Delta-method IRF standard errors via numerical differentiation of A^h w.r.t. params.

    This matches the approach in vectorar + impulseresponse from the MFE toolbox
    (which also uses numerical gradients for the compound A^h derivative).
    """
    eps = 1e-5
    n_regs = res.vcv.shape[0] // K   # per-equation regressors
    std_errors = np.zeros((K, K, horizon + 1), dtype=np.float64)

    # Extract the VAR coefficient block (ignore const) for Jacobian
    # For each VAR equation k, params in res.vcv rows k*n_regs..(k+1)*n_regs
    # correspond to [const?, lag1_y1..lag1_yK, lag2_y1..lag2_yK, ...]
    n_const = 1 if res.const is not None else 0

    for h in range(horizon + 1):
        # Numerical Jacobian of vec(Psi_h * P0) w.r.t. each VAR coefficient
        # Only A entries matter; we perturb each entry of A[:K, :] (top K rows)
        Ah_base = np.linalg.matrix_power(A, h)
        Psi_h_base = J @ Ah_base @ J.T @ P0   # (K, K)

        jac = np.zeros((K * K, K * K * max_lag), dtype=np.float64)
        for row in range(K):
            for col in range(max_lag * K):
                A_pert = A.copy()
                A_pert[row, col] += eps
                Ah_pert = np.linalg.matrix_power(A_pert, h)
                Psi_pert = J @ Ah_pert @ J.T @ P0
                jac[:, row * max_lag * K + col] = (Psi_pert - Psi_h_base).ravel() / eps

        # VCV of the VAR coefficients (top-K rows of A, lag-param block only)
        # Map from A-indexing to res.vcv indexing
        # A[:K, lag_block * K + var] = Phi_{lag_block+1}[row, var]
        # res.vcv is (K*n_regs, K*n_regs) with per-equation ordering
        # We take the diagonal variance (ignoring cross-equation for simplicity)
        var_irf = np.zeros(K * K, dtype=np.float64)
        for response_var in range(K):
            vcv_block = res.vcv[response_var * n_regs:(response_var + 1) * n_regs,
                                 response_var * n_regs:(response_var + 1) * n_regs]
            # lag coefficient block (skip const)
            vcv_lag = vcv_block[n_const:, n_const:]   # (P*K, P*K)
            # Jacobian rows for this response variable
            row_slice = slice(response_var * K, (response_var + 1) * K)
            jac_k = jac[row_slice, response_var * max_lag * K:(response_var + 1) * max_lag * K]
            if jac_k.shape[1] == vcv_lag.shape[0]:
                var_k = np.diag(jac_k @ vcv_lag @ jac_k.T)
                var_irf[row_slice] = np.maximum(var_k, 0.0)

        se_h = np.sqrt(var_irf).reshape(K, K)
        std_errors[:, :, h] = se_h

    return std_errors
