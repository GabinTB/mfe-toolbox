"""
Wild bootstrap for realized volatility and related statistics.

Gonçalves, S. & Meddahi, N. (2009): "Bootstrapping Realized Volatility",
Econometrica, 77(1), 283-306.

The wild bootstrap resamples by multiplying each squared return by an i.i.d.
multiplier w_t drawn from a two-point distribution that matches the first
two moments of the standard normal.

This is appropriate for realized volatility statistics because:
1. The squared-return sequence has heterogeneous conditional variance.
2. Block resampling destroys the i.i.d.-ness of squared returns under the null.
3. The wild bootstrap is consistent for RV-based test statistics even in the
   presence of microstructure noise (with appropriate pre-averaging).

Two-point Rademacher multiplier: w_t = +1 or -1 with prob 1/2.
Mammen (1993) multiplier: w_t = -(sqrt(5)-1)/2 or (sqrt(5)+1)/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mfe.utils.typing import FloatArray


MultiplierType = type  # "rademacher" | "mammen" | "normal"


def _rademacher(n: int, rng: np.random.Generator) -> FloatArray:
    """w_t in {-1, +1} with prob 1/2."""
    return rng.choice([-1.0, 1.0], size=n).astype(np.float64)


def _mammen(n: int, rng: np.random.Generator) -> FloatArray:
    """Mammen (1993) two-point: matches first three moments of N(0,1)."""
    sqrt5 = np.sqrt(5.0)
    p = (sqrt5 + 1) / (2 * sqrt5)
    a = -(sqrt5 - 1) / 2
    b = (sqrt5 + 1) / 2
    u = rng.random(n)
    return np.where(u < p, a, b).astype(np.float64)


def _normal(n: int, rng: np.random.Generator) -> FloatArray:
    return rng.standard_normal(n)


_MULTIPLIER_FUNCS = {
    "rademacher": _rademacher,
    "mammen": _mammen,
    "normal": _normal,
}


@dataclass
class WildBootstrapResult:
    """Result from a wild bootstrap confidence interval computation."""
    statistic: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    bootstrap_distribution: FloatArray
    n_replications: int
    multiplier: str


def wild_bootstrap_rv(
    returns: FloatArray,
    statistic_fn: Callable[[FloatArray], float] | None = None,
    n_replications: int = 999,
    ci_level: float = 0.95,
    multiplier: str = "rademacher",
    rng: np.random.Generator | None = None,
) -> WildBootstrapResult:
    """
    Wild bootstrap confidence interval for a realized volatility statistic.

    The bootstrap DGP is:
        r_t^* = w_t * r_t

    where w_t is i.i.d. from the specified multiplier distribution.
    The statistic is re-evaluated on {r_t^*}.

    Parameters
    ----------
    returns        : (T,) log-return array
    statistic_fn   : function mapping returns -> float; default is sum(r^2) (RV)
    n_replications : number of bootstrap replications
    ci_level       : confidence level (e.g. 0.95 for 95% CI)
    multiplier     : "rademacher" | "mammen" | "normal"
    rng            : numpy random generator; if None, uses default_rng()

    Returns
    -------
    WildBootstrapResult
    """
    r = np.asarray(returns, dtype=np.float64)
    T = len(r)

    if rng is None:
        rng = np.random.default_rng()

    if statistic_fn is None:
        def statistic_fn(x: FloatArray) -> float:
            return float(np.sum(x ** 2))

    mult_func = _MULTIPLIER_FUNCS.get(multiplier)
    if mult_func is None:
        raise ValueError(f"multiplier must be one of {list(_MULTIPLIER_FUNCS)}, got '{multiplier}'")

    # Point estimate
    stat0 = statistic_fn(r)

    # Bootstrap distribution
    boot_stats = np.empty(n_replications, dtype=np.float64)
    for b in range(n_replications):
        w = mult_func(T, rng)
        r_star = w * r
        boot_stats[b] = statistic_fn(r_star)

    alpha = 1.0 - ci_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return WildBootstrapResult(
        statistic=stat0,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        bootstrap_distribution=boot_stats,
        n_replications=n_replications,
        multiplier=multiplier,
    )


def wild_bootstrap_test(
    returns: FloatArray,
    null_statistic: float,
    statistic_fn: Callable[[FloatArray], float] | None = None,
    n_replications: int = 999,
    multiplier: str = "rademacher",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Wild bootstrap p-value for a two-sided hypothesis test.

    Parameters
    ----------
    null_statistic : the value of the statistic under the null hypothesis

    Returns
    -------
    (observed_statistic, bootstrap_p_value)
    """
    result = wild_bootstrap_rv(
        returns,
        statistic_fn=statistic_fn,
        n_replications=n_replications,
        multiplier=multiplier,
        rng=rng,
    )
    p_val = float(np.mean(np.abs(result.bootstrap_distribution - null_statistic) >=
                          abs(result.statistic - null_statistic)))
    return result.statistic, p_val
