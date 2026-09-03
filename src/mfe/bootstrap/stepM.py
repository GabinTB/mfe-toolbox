"""
StepM: Stepdown Multiple Hypothesis Testing with FWER control.

Romano, J.P. & Wolf, M. (2005): "Stepwise Multiple Testing as Formalized Data
Snooping", Econometrica, 73(4), 1237-1282.

Setup
-----
M null hypotheses H_k: mu_k <= 0 for k = 1..M, where mu_k = E[d_{k,t}] is the
mean performance differential of model k vs. the benchmark.

The algorithm controls the familywise error rate (FWER):
    FWER = P(reject at least one true H_k) <= alpha

This is more powerful than Bonferroni and more interpretable than SPA:
it returns which models are significantly better, not just whether any is.

Algorithm (Algorithm 4.1 of Romano & Wolf 2005)
------------------------------------------------
1. Start with all M models.
2. Compute test statistics t_k = sqrt(T) * d_bar_k / sigma_k.
3. Use the stationary bootstrap to get the joint null distribution of
   max_k t_k (over the remaining models).
4. Reject the model with the largest t_k if it exceeds the bootstrap
   critical value at level alpha.
5. Remove rejected models from the set and repeat.
6. Stop when no more rejections occur.

The result is a set of models significantly better than the benchmark.

This matches the MFE MATLAB implementation under bootstrap/stepm.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mfe.utils.typing import FloatArray


@dataclass
class StepMResult:
    """StepM multiple hypothesis testing result."""
    rejected: list[int]          # indices (0-based) of models that significantly beat benchmark
    accepted: list[int]          # indices that could not be rejected (H0 not rejected)
    t_stats: FloatArray          # (M,) studentized statistics for all models
    p_values_raw: FloatArray     # (M,) unadjusted p-values
    p_values_adjusted: FloatArray  # (M,) FWER-adjusted p-values (stepdown)
    n_models: int
    n_obs: int
    n_bootstrap: int
    alpha: float


def _stationary_bootstrap(
    d: FloatArray,
    n_boot: int,
    avg_block_len: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_boot, T, M) bootstrap resamples of the centered loss differentials."""
    T, M = d.shape
    p = 1.0 / avg_block_len
    out = np.empty((n_boot, T, M), dtype=np.float64)
    for b in range(n_boot):
        i = rng.integers(0, T)
        for t in range(T):
            out[b, t] = d[i]
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T
    return out


def _long_run_std(d: FloatArray, bandwidth: int) -> FloatArray:
    """(M,) Newey-West long-run standard deviations."""
    T, M = d.shape
    sigma = np.empty(M, dtype=np.float64)
    for k in range(M):
        dk = d[:, k] - d[:, k].mean()
        var = float(dk @ dk) / T
        for lag in range(1, bandwidth + 1):
            w = 1.0 - lag / (bandwidth + 1)
            var += 2 * w * float(dk[lag:] @ dk[:T - lag]) / T
        sigma[k] = max(var, 1e-30) ** 0.5
    return sigma


def step_m(
    loss_benchmark: FloatArray,
    loss_models: FloatArray,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    avg_block_len: float | None = None,
    bandwidth: int | None = None,
    rng: np.random.Generator | None = None,
) -> StepMResult:
    """
    Romano-Wolf StepM stepdown multiple hypothesis test.

    Tests H_k: E[L_bench - L_model_k] <= 0  for k = 1..M.
    Rejects H_k (model k beats benchmark) for k in result.rejected.
    Controls FWER <= alpha across all M tests.

    Parameters
    ----------
    loss_benchmark : (T,) benchmark loss series (lower = better)
    loss_models    : (T, M) alternative model loss series
    alpha          : familywise error rate (default 0.05)
    n_bootstrap    : stationary bootstrap replications
    avg_block_len  : average block length; if None uses T^{1/3}
    bandwidth      : Newey-West bandwidth; if None uses 1.2 * T^{1/3}
    rng            : random generator

    Returns
    -------
    StepMResult
        .rejected  — 0-based indices of models significantly beating benchmark
        .accepted  — the rest
    """
    lb = np.asarray(loss_benchmark, dtype=np.float64)
    lm = np.asarray(loss_models, dtype=np.float64)
    if lm.ndim == 1:
        lm = lm[:, None]

    T, M = lm.shape
    if rng is None:
        rng = np.random.default_rng()
    if avg_block_len is None:
        avg_block_len = max(2.0, T ** (1 / 3))
    if bandwidth is None:
        bandwidth = max(1, int(1.2 * T ** (1 / 3)))

    # Loss differentials: d_{k,t} = L_bench_t - L_model_k_t
    d = lb[:, None] - lm                   # (T, M)
    d_bar = d.mean(axis=0)                 # (M,)
    sigma = _long_run_std(d, bandwidth)    # (M,)
    t_stats = np.sqrt(T) * d_bar / sigma  # (M,)

    # Unadjusted p-values (individual, no FWER control)
    # Use bootstrap max distribution over all M models
    boot = _stationary_bootstrap(d - d_bar[None, :], n_bootstrap, avg_block_len, rng)
    # boot: (n_boot, T, M) — resampled centered loss diffs

    p_raw = np.empty(M, dtype=np.float64)
    boot_max_all = (np.sqrt(T) * boot.mean(axis=1) / sigma[None, :]).max(axis=1)
    for k in range(M):
        boot_k = np.sqrt(T) * boot[:, :, k].mean(axis=1) / sigma[k]
        p_raw[k] = float(np.mean(boot_k >= t_stats[k]))

    # Stepdown procedure
    remaining = list(range(M))
    rejected = []
    p_adjusted = np.ones(M, dtype=np.float64)

    step = 0
    while remaining:
        # Bootstrap max over remaining models
        boot_max = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            t_boot_remaining = np.sqrt(T) * boot[b, :, :][:, remaining].mean(axis=0) / sigma[remaining]
            boot_max[b] = float(np.max(t_boot_remaining))

        # Critical value at level alpha
        cv = float(np.quantile(boot_max, 1 - alpha))

        # Find the model with max t_stat among remaining
        t_remaining = t_stats[remaining]
        max_idx_in_remaining = int(np.argmax(t_remaining))
        max_k = remaining[max_idx_in_remaining]
        max_t = float(t_stats[max_k])

        if max_t > cv:
            # Reject this model
            p_adjusted[max_k] = float(np.mean(boot_max >= max_t))
            rejected.append(max_k)
            remaining.remove(max_k)
            step += 1
        else:
            # No more rejections possible
            break

    # Adjusted p-values for accepted: use the last step's distribution
    # (conservative: bound by the p-value from the last step)
    for k in remaining:
        p_adjusted[k] = float(np.mean(boot_max_all >= t_stats[k]))

    # Monotonise: stepdown p-values must be non-decreasing when sorted by t_stat descending
    order = np.argsort(-t_stats)
    p_mono = p_adjusted[order].copy()
    for i in range(1, M):
        p_mono[i] = max(p_mono[i], p_mono[i - 1])
    p_adjusted[order] = p_mono

    accepted = [k for k in range(M) if k not in rejected]

    return StepMResult(
        rejected=sorted(rejected),
        accepted=sorted(accepted),
        t_stats=t_stats,
        p_values_raw=p_raw,
        p_values_adjusted=p_adjusted,
        n_models=M,
        n_obs=T,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
