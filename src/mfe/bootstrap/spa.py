"""
Superior Predictive Ability (SPA) test — Hansen (2005).

Hansen, P.R. (2005): "A Test for Superior Predictive Ability",
Journal of Business & Economic Statistics, 23(4), 365-380.

Also implements White (2000) Reality Check as a special case.

Setup
-----
Given M models and a benchmark, let d_{k,t} = L(y_t, f_{0,t}) - L(y_t, f_{k,t})
be the loss differential at time t for model k vs. benchmark (model 0).
d_{k,t} > 0 means model k is better than benchmark at time t.

H0: max_k E[d_{k,t}] <= 0  (no model beats the benchmark on average)
H1: max_k E[d_{k,t}] > 0   (at least one model is strictly better)

Test statistic
--------------
T_SPA = max_k ( sqrt(T) * d_bar_k / sigma_k )
where d_bar_k = mean(d_{k,t}) and sigma_k^2 is the long-run variance of d_{k,t}.

Under H0, T_SPA has a distribution that depends on the correlation structure of
{d_{k,t}} across k. P-values are computed by the stationary bootstrap.

Hansen's SPA uses a "studentized" version with three variants of the null:
  - "consistent" (default): removes irrelevant models from the null (d_bar_k << 0)
  - "upper": retains all models (equivalent to White's Reality Check)
  - "lower": most conservative, all models treated as tied with benchmark

References
----------
White, H. (2000): "A Reality Check for Data Snooping", Econometrica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mfe.utils.typing import FloatArray


@dataclass
class SPAResult:
    statistic: float          # max_k t_k = max_k sqrt(T)*d_bar_k/sigma_k
    p_value_consistent: float # Hansen consistent p-value (default to report)
    p_value_upper: float      # White Reality Check p-value
    p_value_lower: float      # lower bound p-value
    d_bar: FloatArray         # (M,) mean loss differentials
    t_stats: FloatArray       # (M,) studentized stats per model
    n_models: int
    n_obs: int
    n_bootstrap: int
    bootstrap_distribution: FloatArray = field(default_factory=lambda: np.array([]))


def _stationary_bootstrap_indices(
    T: int,
    n_boot: int,
    avg_block_len: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary bootstrap of Politis & Romano (1994).
    Returns (n_boot, T) array of resample indices.
    p = 1/avg_block_len is the geometric block-end probability.
    """
    p = 1.0 / avg_block_len
    idx = np.empty((n_boot, T), dtype=np.int64)
    for b in range(n_boot):
        i = rng.integers(0, T)
        for t in range(T):
            idx[b, t] = i
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T
    return idx


def _long_run_variance(
    d: FloatArray,
    bandwidth: int | None = None,
) -> FloatArray:
    """
    Newey-West long-run variance for each column of d.
    Returns (M,) array of sigma_k^2.
    """
    T, M = d.shape
    if bandwidth is None:
        bandwidth = int(np.ceil(1.2 * T ** (1 / 3)))

    sigma2 = np.empty(M, dtype=np.float64)
    for k in range(M):
        dk = d[:, k] - d[:, k].mean()
        var = float(dk @ dk) / T
        for lag in range(1, bandwidth + 1):
            w = 1.0 - lag / (bandwidth + 1)
            gamma = float(dk[lag:] @ dk[:T - lag]) / T
            var += 2 * w * gamma
        sigma2[k] = max(var, 1e-30)

    return sigma2


def spa_test(
    loss_benchmark: FloatArray,
    loss_models: FloatArray,
    n_bootstrap: int = 999,
    avg_block_len: float | None = None,
    bandwidth: int | None = None,
    rng: np.random.Generator | None = None,
) -> SPAResult:
    """
    Hansen (2005) Superior Predictive Ability test.

    Parameters
    ----------
    loss_benchmark : (T,) loss series for the benchmark model (lower = better)
    loss_models    : (T, M) loss series for M alternative models
    n_bootstrap    : stationary bootstrap replications (default 999)
    avg_block_len  : average block length for stationary bootstrap;
                     if None, uses T^{1/3}
    bandwidth      : Newey-West bandwidth for long-run variance;
                     if None, uses 1.2 * T^{1/3}
    rng            : numpy Generator; if None uses default_rng()

    Returns
    -------
    SPAResult
        Report .p_value_consistent for the standard SPA p-value.
        Report .p_value_upper for White's Reality Check p-value.

    Notes
    -----
    Loss convention: LOWER is BETTER (e.g. MSE, MAE, negative log-lik).
    Loss differential d_{k,t} = L_benchmark_t - L_model_k_t.
    Positive d_bar_k means model k beats benchmark on average.
    """
    lb = np.asarray(loss_benchmark, dtype=np.float64)
    lm = np.asarray(loss_models, dtype=np.float64)
    if lm.ndim == 1:
        lm = lm[:, None]

    T, M = lm.shape
    if rng is None:
        rng = np.random.default_rng()
    if avg_block_len is None:
        avg_block_len = max(2.0, float(T ** (1 / 3)))

    # Loss differentials: d_{k,t} = L_bench_t - L_model_k_t
    # d_bar_k > 0 means model k is better than benchmark
    d = lb[:, None] - lm   # (T, M)

    d_bar = d.mean(axis=0)  # (M,)

    # Long-run variance
    sigma2 = _long_run_variance(d, bandwidth=bandwidth)
    sigma = np.sqrt(sigma2)

    # Studentized statistics: t_k = sqrt(T) * d_bar_k / sigma_k
    t_stats = np.sqrt(T) * d_bar / sigma   # (M,)
    T_spa = float(np.max(t_stats))

    # Stationary bootstrap to get null distribution of max t_k
    idx = _stationary_bootstrap_indices(T, n_bootstrap, avg_block_len, rng)

    # Three null variants of Hansen (2005)
    # "consistent": zero out models with strongly negative d_bar (irrelevant)
    # "upper":      keep all models (White RC)
    # "lower":      only models with d_bar > 0 (conservative)

    # Threshold for "consistent": c_k = max(d_bar_k, -sqrt(sigma2_k * log(log(T)) / T))
    c_consistent = np.maximum(d_bar, -np.sqrt(sigma2 * np.log(np.log(T)) / T))
    c_upper = d_bar.copy()          # White Reality Check: mean-center on d_bar
    c_lower = np.maximum(d_bar, 0.0)

    boot_max_consistent = np.empty(n_bootstrap, dtype=np.float64)
    boot_max_upper = np.empty(n_bootstrap, dtype=np.float64)
    boot_max_lower = np.empty(n_bootstrap, dtype=np.float64)

    d_centered_consistent = d - c_consistent[None, :]
    d_centered_upper      = d - c_upper[None, :]
    d_centered_lower      = d - c_lower[None, :]

    for b in range(n_bootstrap):
        d_boot_c = d_centered_consistent[idx[b]].mean(axis=0)
        d_boot_u = d_centered_upper[idx[b]].mean(axis=0)
        d_boot_l = d_centered_lower[idx[b]].mean(axis=0)

        t_boot_c = np.sqrt(T) * d_boot_c / sigma
        t_boot_u = np.sqrt(T) * d_boot_u / sigma
        t_boot_l = np.sqrt(T) * d_boot_l / sigma

        boot_max_consistent[b] = float(np.max(t_boot_c))
        boot_max_upper[b]      = float(np.max(t_boot_u))
        boot_max_lower[b]      = float(np.max(t_boot_l))

    pval_consistent = float(np.mean(boot_max_consistent >= T_spa))
    pval_upper      = float(np.mean(boot_max_upper >= T_spa))
    pval_lower      = float(np.mean(boot_max_lower >= T_spa))

    return SPAResult(
        statistic=T_spa,
        p_value_consistent=pval_consistent,
        p_value_upper=pval_upper,
        p_value_lower=pval_lower,
        d_bar=d_bar,
        t_stats=t_stats,
        n_models=M,
        n_obs=T,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=boot_max_consistent,
    )
