"""
CCC-GARCH (Constant Conditional Correlation).

Bollerslev, T. (1990): "Modelling the Coherence in Short-Run Nominal Exchange
Rates: A Multivariate Generalized ARCH Model", Review of Economics and Statistics.

H_t = D_t R D_t

Where D_t = diag(sigma_{1,t}, ..., sigma_{K,t}) from K independent GARCH(1,1)
and R = unconditional correlation matrix (constant).

Two-step estimation:
  Step 1: Fit GARCH(1,1) to each series independently.
  Step 2: Compute R = sample correlation of standardized residuals.

This is just DCC with a=b=0 (zero dynamics in the correlation), but it's
worth having as an explicit model for testing (CCC vs DCC likelihood ratio test).
"""

from __future__ import annotations

import numpy as np

from mfe.multivariate.base import MultivariateVolResult
from mfe.multivariate.dcc import _fit_univariate_garch
from mfe.utils.typing import FloatArray


class CCC:
    """
    CCC-GARCH(1,1) model (Bollerslev 1990).

    No free parameters beyond the K univariate GARCH models.
    """

    def fit(self, data: FloatArray) -> MultivariateVolResult:
        """
        Estimate CCC-GARCH.

        Parameters
        ----------
        data : (T, K) return matrix
        """
        data = np.asarray(data, dtype=np.float64)
        T, K = data.shape

        # Step 1: univariate GARCH
        h, z = _fit_univariate_garch(data)

        # Step 2: unconditional correlation of standardized residuals
        R = np.corrcoef(z.T)  # (K, K)

        # Sigma_t = D_t R D_t
        D = np.sqrt(h)  # (T, K)
        sigma_t = np.empty((T, K, K), dtype=np.float64)
        for t in range(T):
            d = np.diag(D[t])
            sigma_t[t] = d @ R @ d

        # Log-likelihood
        ll = 0.0
        for t in range(T):
            sign, logdet = np.linalg.slogdet(sigma_t[t])
            if sign <= 0:
                ll = -1e10
                break
            H_inv = np.linalg.inv(sigma_t[t])
            ll += -0.5 * (K * np.log(2 * np.pi) + logdet + float(data[t] @ H_inv @ data[t]))

        P = 3 * K  # omega, alpha, beta per asset; R is not a free param in estimation
        return MultivariateVolResult(
            params=np.array([]),  # R embedded in diagnostics
            log_likelihood=ll,
            conditional_covariances=sigma_t,
            residuals=z,
            vcv=np.zeros((0, 0)),
            vcv_robust=np.zeros((0, 0)),
            scores=np.zeros((T, 0)),
            converged=True,
            n_obs=T,
            n_params=P,
            model_name=f"CCC-GARCH(1,1), K={K}",
            diagnostics={"R": R},
        )
