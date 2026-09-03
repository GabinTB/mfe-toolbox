"""
HEAVY (High frEquency bAsed VolatilitY) model.

Shephard, N. & Sheppard, K. (2010): "Realising the Future: Forecasting with
High-Frequency-Based Volatility (HEAVY) Models",
Journal of Applied Econometrics, 25(2), 197-231.

Model specification
-------------------
The HEAVY model jointly models daily returns r_t and realized measures RM_t
(e.g. realized variance) using two equations:

  h_{r,t}  = omega_r  + alpha_r * RM_{t-1}  + beta_r * h_{r,t-1}
  h_{RM,t} = omega_RM + alpha_RM * RM_{t-1} + beta_RM * h_{RM,t-1}

where h_{r,t} is the conditional variance of returns and h_{RM,t} is the
conditional mean of the realized measure.

The second equation is a separate GARCH-like model for RM itself.

Estimation: joint QML assuming:
  r_t | F_{t-1} ~ N(0, h_{r,t})
  RM_t | F_{t-1} ~ Gamma(nu, nu / h_{RM,t})  (variance = h_{RM,t}^2 / nu)

The joint log-likelihood is:
  L = L_r + L_RM
  L_r   = -0.5 * sum_t [log(h_{r,t}) + r_t^2 / h_{r,t}]
  L_RM  = -0.5 * nu * sum_t [log(h_{RM,t}) + RM_t/h_{RM,t} - log(RM_t/h_{RM,t}) - 1]
         (Gamma log-likelihood in terms of scale; simplified constant dropped)

This is the model that `arch` partially stubs (ResearchModel) but never completes.

Key innovation vs. standard GARCH
----------------------------------
By using realized measures in the variance equation, HEAVY produces forecasts
that update faster: when overnight volatility is high (RM_{t-1} large), h_{r,t}
responds immediately rather than waiting for the squared return signal.

Reference implementation cross-check
--------------------------------------
MATLAB mfe-toolbox: heavy.m
The MATLAB version computes analytic gradients. We compute numerical gradients
via scipy; analytic scores are a future optimization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from mfe.multivariate.base import ConvergenceWarning
from mfe.utils.typing import FloatArray


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HEAVYResult:
    """HEAVY model estimation result."""
    params: FloatArray         # (6,) [omega_r, alpha_r, beta_r, omega_rm, alpha_rm, beta_rm]
    log_likelihood: float
    h_returns: FloatArray      # (T,) conditional variance of returns
    h_realized: FloatArray     # (T,) conditional mean of realized measure
    residuals_r: FloatArray    # (T,) standardized return residuals r_t / sqrt(h_{r,t})
    residuals_rm: FloatArray   # (T,) RM_t / h_{RM,t} (standardized realized)
    converged: bool
    n_obs: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def aic(self) -> float:
        return -2 * self.log_likelihood + 2 * 6

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + 6 * np.log(self.n_obs)

    @property
    def param_names(self) -> list[str]:
        return ["omega_r", "alpha_r", "beta_r", "omega_rm", "alpha_rm", "beta_rm"]


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------

def _heavy_recursion(
    params: FloatArray,
    returns: FloatArray,
    realized: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Run the HEAVY recursion given parameters.

    Parameters are [omega_r, alpha_r, beta_r, omega_rm, alpha_rm, beta_rm].
    Returns (h_r, h_rm) — (T,) arrays of conditional variances/means.
    """
    omega_r, alpha_r, beta_r, omega_rm, alpha_rm, beta_rm = params
    T = len(returns)

    h_r  = np.empty(T, dtype=np.float64)
    h_rm = np.empty(T, dtype=np.float64)

    # Backcast: unconditional mean as initial value
    rm_bar = float(np.mean(realized))
    r2_bar = float(np.mean(returns ** 2))

    h_r[0]  = r2_bar
    h_rm[0] = rm_bar

    for t in range(1, T):
        h_r[t]  = omega_r  + alpha_r  * realized[t - 1] + beta_r  * h_r[t - 1]
        h_rm[t] = omega_rm + alpha_rm * realized[t - 1] + beta_rm * h_rm[t - 1]

    # Enforce positivity
    h_r  = np.maximum(h_r,  1e-10)
    h_rm = np.maximum(h_rm, 1e-10)

    return h_r, h_rm


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def _heavy_loglik(
    params: FloatArray,
    returns: FloatArray,
    realized: FloatArray,
) -> float:
    """
    Negative joint QML log-likelihood (for minimization).

    Return equation: Gaussian QML
    Realized equation: Gamma QML (Barndorff-Nielsen & Shephard 2002)
    """
    omega_r, alpha_r, beta_r, omega_rm, alpha_rm, beta_rm = params

    # Stationarity and positivity constraints
    if (omega_r <= 0 or alpha_r < 0 or beta_r < 0 or
            alpha_r + beta_r >= 1.0 or
            omega_rm <= 0 or alpha_rm < 0 or beta_rm < 0 or
            alpha_rm + beta_rm >= 1.0):
        return 1e10

    h_r, h_rm = _heavy_recursion(params, returns, realized)

    # Return likelihood: -0.5 * sum [log(h_r) + r^2 / h_r]
    ll_r = -0.5 * float(np.sum(np.log(h_r) + returns ** 2 / h_r))

    # Realized likelihood: Gamma parameterized so E[RM] = h_rm, Var[RM] = h_rm^2/nu
    # With nu -> inf (large shape) this approaches log-normal; we use nu fixed at
    # the MoM estimate from the data, or treat it as a nuisance parameter with nu=4
    # (Shephard & Sheppard 2010 empirical default).
    nu = 4.0
    u = realized / h_rm   # standardized realized
    ll_rm = -0.5 * nu * float(np.sum(np.log(h_rm) + u - np.log(u) - 1.0))

    return -(ll_r + ll_rm)


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

def _heavy_starting_values(
    returns: FloatArray,
    realized: FloatArray,
) -> FloatArray:
    """
    Covariance-targeting starting values for HEAVY.
    """
    rm_bar = float(np.mean(realized))
    r2_bar = float(np.mean(returns ** 2))

    alpha_r0, beta_r0 = 0.20, 0.70
    omega_r0 = r2_bar * (1.0 - alpha_r0 - beta_r0)

    alpha_rm0, beta_rm0 = 0.20, 0.70
    omega_rm0 = rm_bar * (1.0 - alpha_rm0 - beta_rm0)

    return np.array([
        max(omega_r0, 1e-6), alpha_r0, beta_r0,
        max(omega_rm0, 1e-6), alpha_rm0, beta_rm0,
    ])


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class HEAVY:
    """
    HEAVY model — joint model of returns and realized variance.

    Shephard & Sheppard (2010). Not available in the `arch` package (stubbed
    but never completed as of 2026).

    Parameters
    ----------
    realized_measure : "rv" | "bpv" | "kernel"
        Which realized measure to use (informational, does not change estimation).
    """

    def __init__(self, realized_measure: str = "rv") -> None:
        self.realized_measure = realized_measure

    def fit(
        self,
        returns: FloatArray,
        realized: FloatArray,
        starting_values: FloatArray | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> HEAVYResult:
        """
        Estimate HEAVY by joint QML.

        Parameters
        ----------
        returns  : (T,) daily return series (demeaned)
        realized : (T,) daily realized variance series (same frequency)
        """
        r = np.asarray(returns, dtype=np.float64)
        rm = np.asarray(realized, dtype=np.float64)
        T = len(r)

        if len(rm) != T:
            raise ValueError(f"returns and realized must have same length, got {T} vs {len(rm)}")
        if np.any(rm <= 0):
            raise ValueError("realized must be strictly positive (use realized variance, not returns)")

        x0 = starting_values if starting_values is not None else _heavy_starting_values(r, rm)
        x0 = np.asarray(x0, dtype=np.float64)

        # Bounds: all positive, alpha+beta < 1
        bounds = [
            (1e-8, None),   # omega_r
            (1e-6, 0.999),  # alpha_r
            (1e-6, 0.999),  # beta_r
            (1e-8, None),   # omega_rm
            (1e-6, 0.999),  # alpha_rm
            (1e-6, 0.999),  # beta_rm
        ]

        result = minimize(
            _heavy_loglik,
            x0,
            args=(r, rm),
            method=method,
            bounds=bounds,
            options=options or {"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
        )

        if not result.success:
            warnings.warn(
                f"HEAVY did not converge: {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        params = result.x
        h_r, h_rm = _heavy_recursion(params, r, rm)

        resid_r  = r / np.sqrt(h_r)
        resid_rm = rm / h_rm

        return HEAVYResult(
            params=params,
            log_likelihood=-result.fun,
            h_returns=h_r,
            h_realized=h_rm,
            residuals_r=resid_r,
            residuals_rm=resid_rm,
            converged=result.success,
            n_obs=T,
            diagnostics={
                "optimizer_result": result,
                "realized_measure": self.realized_measure,
            },
        )

    def forecast(
        self,
        result: HEAVYResult,
        horizon: int = 1,
        last_realized: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Multi-step ahead forecasts of h_r and h_rm.

        Parameters
        ----------
        result         : fitted HEAVYResult
        horizon        : number of steps ahead
        last_realized  : RM value at time T (for recursion start);
                         if None, uses the last value from the estimation sample

        Returns
        -------
        (h_r_forecast, h_rm_forecast) — both (horizon,) arrays
        """
        params = result.params
        omega_r, alpha_r, beta_r, omega_rm, alpha_rm, beta_rm = params

        h_r_last  = float(result.h_returns[-1])
        h_rm_last = float(result.h_realized[-1])

        if last_realized is None:
            # Use the last fitted h_rm as a proxy for E[RM_T | F_{T-1}]
            rm_last = h_rm_last
        else:
            rm_last = float(last_realized)

        h_r_fc  = np.empty(horizon, dtype=np.float64)
        h_rm_fc = np.empty(horizon, dtype=np.float64)

        # One step ahead: use last_realized directly
        h_r_fc[0]  = omega_r  + alpha_r  * rm_last + beta_r  * h_r_last
        h_rm_fc[0] = omega_rm + alpha_rm * rm_last + beta_rm * h_rm_last

        # Further steps: replace RM_{t-1} with its conditional expectation h_{RM,t-1}
        for h in range(1, horizon):
            h_r_fc[h]  = omega_r  + alpha_r  * h_rm_fc[h - 1] + beta_r  * h_r_fc[h - 1]
            h_rm_fc[h] = omega_rm + alpha_rm * h_rm_fc[h - 1] + beta_rm * h_rm_fc[h - 1]

        return h_r_fc, h_rm_fc
