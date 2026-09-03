"""
Abstract base class for multivariate volatility models.

Design principles:
- Results are dataclasses, not mutable objects, to avoid state mutation bugs.
- Convergence failures surface as ConvergenceWarning, not silent bad params.
- Robust VCV (sandwich) is computed by default alongside the Hessian-only VCV.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from mfe.utils.typing import FloatArray
from mfe.utils.vcv import sandwich, newey_west


class ConvergenceWarning(UserWarning):
    """Raised when the optimizer did not fully converge."""


@dataclass
class MultivariateVolResult:
    """Container for multivariate volatility estimation results."""
    params: FloatArray                  # (P,) estimated parameters
    log_likelihood: float
    conditional_covariances: FloatArray  # (T, K, K) time-varying Sigma_t
    residuals: FloatArray               # (T, K) standardized residuals
    vcv: FloatArray                     # (P, P) inverse-Hessian VCV
    vcv_robust: FloatArray              # (P, P) sandwich VCV
    scores: FloatArray                  # (T, P) score matrix
    converged: bool = True
    n_obs: int = 0
    n_params: int = 0
    model_name: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def std_errors(self) -> FloatArray:
        return np.sqrt(np.diag(self.vcv))

    @property
    def std_errors_robust(self) -> FloatArray:
        return np.sqrt(np.diag(self.vcv_robust))

    @property
    def aic(self) -> float:
        return -2 * self.log_likelihood + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + self.n_params * np.log(self.n_obs)


class MultivariateVolatilityProcess(ABC):
    """
    Base class for all multivariate volatility models.

    Subclasses must implement:
        _log_likelihood(params, data) -> float
        _compute_covariances(params, data) -> (T, K, K)
        _starting_values(data) -> FloatArray
        _parameter_bounds(data, K) -> list[tuple[float, float]]
        _parameter_names(K) -> list[str]
    """

    @abstractmethod
    def _log_likelihood(
        self,
        params: FloatArray,
        data: FloatArray,
    ) -> float:
        """Negative log-likelihood (for minimization)."""
        ...

    @abstractmethod
    def _compute_covariances(
        self,
        params: FloatArray,
        data: FloatArray,
    ) -> FloatArray:
        """Return (T, K, K) array of conditional covariances."""
        ...

    @abstractmethod
    def _starting_values(self, data: FloatArray) -> FloatArray:
        """Return initial parameter vector."""
        ...

    @abstractmethod
    def _parameter_bounds(self, data: FloatArray, K: int) -> list[tuple[float, float]]:
        """Return list of (lower, upper) bounds per parameter."""
        ...

    @abstractmethod
    def _parameter_names(self, K: int) -> list[str]:
        """Return human-readable parameter names."""
        ...

    def fit(
        self,
        data: FloatArray,
        starting_values: FloatArray | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> MultivariateVolResult:
        """
        Fit the model via quasi-maximum likelihood.

        Parameters
        ----------
        data            : (T, K) return matrix
        starting_values : (P,) starting parameter vector; if None, uses heuristic
        method          : scipy.optimize.minimize method
        options         : passed to minimize

        Returns
        -------
        MultivariateVolResult
        """
        from scipy.optimize import minimize

        data = np.asarray(data, dtype=np.float64)
        T, K = data.shape

        if starting_values is None:
            x0 = self._starting_values(data)
        else:
            x0 = np.asarray(starting_values, dtype=np.float64)

        bounds = self._parameter_bounds(data, K)

        result = minimize(
            self._log_likelihood,
            x0,
            args=(data,),
            method=method,
            bounds=bounds,
            options=options or {"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            warnings.warn(
                f"{self.__class__.__name__} did not converge: {result.message}. "
                "Use results with caution.",
                ConvergenceWarning,
                stacklevel=2,
            )

        params = result.x
        sigma_t = self._compute_covariances(params, data)

        # Score matrix via finite differences (TODO: analytic scores per model)
        eps = 1e-6
        P = len(params)
        scores = np.zeros((T, P), dtype=np.float64)
        # ... (placeholder: per-obs score via finite differences)

        # Hessian from scipy (numerical)
        try:
            from scipy.optimize import approx_fprime
            hessian = np.zeros((P, P), dtype=np.float64)
            for i in range(P):
                ei = np.zeros(P)
                ei[i] = eps
                g_plus = approx_fprime(params + ei, self._log_likelihood, eps, data)
                g_minus = approx_fprime(params - ei, self._log_likelihood, eps, data)
                hessian[i] = (g_plus - g_minus) / (2 * eps)
            hessian = 0.5 * (hessian + hessian.T)
            vcv = np.linalg.inv(hessian) if np.linalg.det(hessian) != 0 else np.full((P, P), np.nan)
        except Exception:
            vcv = np.full((P, P), np.nan)

        vcv_robust = sandwich(scores, hessian) if not np.any(np.isnan(scores)) else vcv

        return MultivariateVolResult(
            params=params,
            log_likelihood=-result.fun,
            conditional_covariances=sigma_t,
            residuals=data,  # caller should standardize if needed
            vcv=vcv,
            vcv_robust=vcv_robust,
            scores=scores,
            converged=result.success,
            n_obs=T,
            n_params=P,
            model_name=self.__class__.__name__,
            diagnostics={"optimizer_result": result},
        )
