"""
Principal Component Analysis for financial returns.

Standalone PCA module with the interface matching the MFE MATLAB pca.m:
- eigendecomposition of the sample covariance
- proportion of variance explained per component
- factor scores (principal components)
- loadings matrix
- reconstruction of the original data from K_c components

This is a clean public API wrapping the internals already used in
mfe.multivariate.gogarch. The MFE MATLAB pca.m is documented but
rarely exposed directly — we make it first-class here.

Distinct from sklearn.decomposition.PCA in that:
- we expose the covariance structure (not correlation by default)
- we match financial conventions: K_c components from covariance,
  not correlation, and we return eigenvalues in variance units
  (not explained variance ratio) alongside the standard stats
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfe.utils.typing import FloatArray


@dataclass
class PCAResult:
    """Principal component analysis result."""
    eigenvalues: FloatArray      # (K,) descending eigenvalues of sample covariance
    eigenvectors: FloatArray     # (K, K) columns = eigenvectors (loadings)
    factors: FloatArray          # (T, K_c) factor scores (principal components)
    loadings: FloatArray         # (K, K_c) = eigenvectors[:, :K_c] * sqrt(eigenvalues[:K_c])
    explained_variance: FloatArray   # (K_c,) proportion of total variance per component
    cumulative_variance: FloatArray  # (K_c,) cumulative
    n_components: int
    n_obs: int
    n_vars: int
    mean: FloatArray             # (K,) sample mean (subtracted before decomposition)

    def reconstruct(self, k_c: int | None = None) -> FloatArray:
        """
        Reconstruct data from the first k_c PCA components.

        Parameters
        ----------
        k_c : number of components to use; if None uses all n_components

        Returns
        -------
        (T, K) reconstructed data matrix (in original scale, mean added back)
        """
        k_c = k_c or self.n_components
        k_c = min(k_c, self.n_components)
        # factors = Xc @ eigvecs[:, :k_c]  (projection)
        # reconstruction = factors @ eigvecs[:, :k_c].T + mean
        # eigvecs[:, :k_c] from PCAResult.eigenvectors
        evecs = self.eigenvectors[:, :k_c]
        return self.factors[:, :k_c] @ evecs.T + self.mean[None, :]


def pca(
    data: FloatArray,
    n_components: int | None = None,
    demean: bool = True,
    standardize: bool = False,
) -> PCAResult:
    """
    Principal component analysis of a (T, K) data matrix.

    Parameters
    ----------
    data         : (T, K) matrix (e.g. asset returns)
    n_components : number of components to retain; if None, keeps all K
    demean       : subtract column means before decomposition (default True)
    standardize  : divide by column std after demeaning (correlation PCA);
                   if False (default), operates on covariance matrix

    Returns
    -------
    PCAResult
        .eigenvalues  — (K,) eigenvalues of sample covariance/correlation matrix
        .eigenvectors — (K, K) eigenvector matrix (columns sorted descending)
        .factors      — (T, K_c) factor scores = demeaned_data @ eigenvectors[:, :K_c]
        .loadings     — (K, K_c) factor loadings scaled by sqrt(eigenvalue)
        .explained_variance — proportion of total variance per component
    """
    X = np.asarray(data, dtype=np.float64)
    T, K = X.shape
    K_c = K if n_components is None else min(n_components, K)

    # Demean
    mu = X.mean(axis=0) if demean else np.zeros(K, dtype=np.float64)
    Xc = X - mu[None, :]

    # Optionally standardize
    if standardize:
        std = Xc.std(axis=0, ddof=1)
        std = np.where(std > 0, std, 1.0)
        Xc = Xc / std[None, :]
    else:
        std = np.ones(K, dtype=np.float64)

    # Sample covariance (no Bessel correction to match MATLAB mfe convention)
    S = Xc.T @ Xc / T

    # Eigendecomposition (eigh is stable for symmetric matrices)
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Clip near-zero eigenvalues
    eigvals_safe = np.maximum(eigvals, 0.0)

    # Factor scores: projection of centered data onto eigenvectors
    factors = Xc @ eigvecs[:, :K_c]  # (T, K_c) — unit-variance if we scale

    # Loadings: eigenvectors scaled by sqrt(eigenvalue) → covariance structure
    loadings = eigvecs[:, :K_c] * np.sqrt(eigvals_safe[:K_c])[None, :]  # (K, K_c)

    # Explained variance
    total_var = float(np.sum(eigvals_safe))
    exp_var = eigvals_safe[:K_c] / max(total_var, 1e-30)
    cum_var = np.cumsum(exp_var)

    return PCAResult(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        factors=factors,
        loadings=loadings,
        explained_variance=exp_var,
        cumulative_variance=cum_var,
        n_components=K_c,
        n_obs=T,
        n_vars=K,
        mean=mu,
    )
