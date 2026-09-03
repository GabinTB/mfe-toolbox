"""
Performance benchmarks for mfe.realized on HFT-scale data.

Run with:
    cd repo_root
    PYTHONPATH=src python dev/benchmarks/bench_realized.py

Or with pytest-benchmark:
    PYTHONPATH=src pytest dev/benchmarks/bench_realized.py --benchmark-only

Target: all operations on T=1_000_000 ticks should complete in < 5s
on a modern CPU without Cython (pure numpy). With Cython, kernel inner
loop should be ~10x faster.
"""

import time

import numpy as np

from mfe.realized import (
    realized_variance,
    realized_bipower_variation,
    realized_med_variance,
    realized_kernel,
    bns_jump_test,
)
from mfe.realized.covariance import realized_hayashi_yoshida


def _make_returns(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) * 0.001


def _make_tick_prices(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    price = np.exp(np.cumsum(rng.standard_normal(T) * 0.001))
    time_ = np.sort(rng.uniform(0, 23400, T))
    return price, time_


def bench(label: str, fn, *args, n_runs: int = 3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    mean_ms = np.mean(times) * 1000
    print(f"  {label:<40s} {mean_ms:8.1f} ms  (n={n_runs})")


if __name__ == "__main__":
    print("=" * 60)
    print("mfe.realized benchmarks")
    print("=" * 60)

    for T in [10_000, 100_000, 500_000]:
        print(f"\nT = {T:,} returns")
        r = _make_returns(T)

        bench("realized_variance", realized_variance, r)
        bench("realized_bipower_variation (skip=0)", realized_bipower_variation, r, 0)
        bench("realized_bipower_variation (skip=1)", realized_bipower_variation, r, 1)
        bench("realized_med_variance", realized_med_variance, r)
        bench("realized_kernel (auto bandwidth)", realized_kernel, r)
        bench("bns_jump_test", bns_jump_test, r)

    print(f"\nHayashi-Yoshida bivariate (non-synchronous)")
    for T in [1_000, 5_000, 10_000]:
        p1, t1 = _make_tick_prices(T, seed=0)
        p2, t2 = _make_tick_prices(T, seed=1)
        # HY is O(N1 * N2) in worst case with the current numpy impl
        bench(f"  HY T={T:,}", realized_hayashi_yoshida, [p1, p2], [t1, t2], n_runs=2)

    print()
    print("NOTE: Cython uses event-sweep with swap-remove open-list.")
    print("      Complexity: O((N1+N2)*log(N1+N2)) sort + O((N1+N2)*k) sweep")
    print("      where k = avg simultaneous open intervals (typically O(1) for HFT data).")
