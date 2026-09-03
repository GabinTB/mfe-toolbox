# mfe — Financial Econometrics for Python

**mfe** is a Python port of Kevin Sheppard's Oxford MFE Toolbox, optimised for HFT data and quantitative research. It complements the [`arch`](https://arch.readthedocs.io) package — covering everything `arch` is missing.

---

## What's inside

=== "Realized volatility"
    Full HFT realized-measure library with Cython-accelerated hot paths:

    - Realized variance, BPV, MedRV, MinRV, pre-averaged RV, semivariance
    - Realized kernel (6 weight functions, auto-bandwidth)
    - Two-scale RV (TSRV) and multi-scale RV (MSRV) — noise-robust
    - Realized quantile variance — jump-robust
    - Hayashi-Yoshida covariance — O((N₁+N₂) log N) sweep-line in Cython
    - Multivariate realized kernel — PSD-guaranteed (K,K) covariance matrix
    - BNS jump test, realized range, microstructure noise estimation

=== "Multivariate GARCH"
    Every model missing from `arch`:

    - **DCC** (Engle 2002), cDCC (Aielli 2013)
    - **CCC** (Bollerslev 1990)
    - **BEKK** scalar + diagonal (Engle & Kroner 1995)
    - **O-GARCH** (Alexander 2001)
    - **GO-GARCH** — ICA or moments rotation (van der Weide 2002)
    - **RCC** (Noureldin, Shephard & Sheppard 2014) — rotated conditional correlation

=== "Univariate models"
    Models not in `arch`:

    - **HAR-RV** — standard, matrix-interval, MODIFIED spec, jump-augmented (HAR-RV-J)
    - **HEAVY** — joint model of returns + realized variance (Shephard & Sheppard 2010)

=== "Time series"
    The gap vs. statsmodels VAR:

    - **VAR** with 4 VCV options: homo/het × corr/uncorr
    - **Granger causality** — LR, LM, Wald tests with robust VCV
    - **Impulse response functions** with delta-method standard errors
    - **Beveridge-Nelson decomposition** — permanent/transitory components

=== "Bootstrap & tests"
    - **Wild bootstrap** for realized volatility statistics
    - **SPA test** — Hansen (2005) Superior Predictive Ability
    - **StepM** — Romano & Wolf (2005) stepdown FWER control
    - **ARCH-LM**, Ljung-Box Q, HAC-robust LM serial correlation
    - **Diebold-Mariano** (MSE/MAE/QLIKE), Mincer-Zarnowitz

=== "Cross-section"
    - **Fama-MacBeth** with Shanken correction
    - **OLS / OLSNW** — White and Newey-West standard errors
    - **PCA** with factor interpretation and reconstruction

---

## Quick example

```python
from mfe.realized import (
    price_filter, returns_from_prices,
    realized_variance, realized_kernel, bns_jump_test,
)
from mfe.realized.sampling import SamplingType

# Filter raw ticks to 5-minute grid
prices_5m, times_5m = price_filter(
    tick_prices, tick_times,
    sampling_type=SamplingType.CALENDAR_TIME,
    sampling_interval=300,
)
r = returns_from_prices(prices_5m)

# Realized variance and kernel
rv  = realized_variance(r)
rk  = realized_kernel(r)           # noise-robust
jmp = bns_jump_test(r)             # jump detection

print(f"RV  = {rv.value:.6f}")
print(f"RK  = {rk.rk_adjusted:.6f}  (H = {rk.bandwidth})")
print(f"Jump significant: {jmp.significant}  (p = {jmp.p_value:.3f})")
```

```python
from mfe.multivariate import DCC, RCC, GOGARCH
from mfe.univariate import HEAVY

# DCC-GARCH on daily returns
dcc = DCC().fit(returns)          # (T, K) return matrix
print(dcc.conditional_covariances.shape)   # (T, K, K)

# Rotated Conditional Correlation — covariance-targeting by construction
rcc = RCC().fit(returns)
print(f"RCC: a={rcc.a:.3f}  b={rcc.b:.3f}")

# HEAVY — use realised variance to sharpen daily vol forecasts
heavy = HEAVY().fit(daily_returns, realized_variances)
h_r, h_rm = heavy.h_returns, heavy.h_realized
```

---

## Installation

```bash
pip install mfe
```

Cython extensions (optional, ~10-800x speedup on hot paths):

```bash
pip install mfe[cython]
# or build from source:
python setup_cython.py build_ext --inplace
```

---

## Design principles

- **Results are dataclasses** — immutable, no state mutation bugs.
- **`ConvergenceWarning`** always fires on non-converged optimisers — never silent.
- **Cython extensions with pure-Python fallbacks** — wheels ship as binaries; pure installs work without a C compiler.
- **No duplication of `arch`** — univariate GARCH estimation stays in `arch`.
- **HFT-first** — all realized estimators operate on raw tick arrays, not DataFrames.

---

## Relationship to the MATLAB MFE Toolbox

This package ports the MATLAB [MFE Toolbox](https://github.com/bashtage/mfe-toolbox) (Kevin Sheppard, Oxford) to Python, fixing known bugs:

| MATLAB issue | Python fix |
|---|---|
| `gogarch.m` closes over `volData` in loop → memory leak | No closures; all state explicit |
| Convergence warning fires then parameters silently used | `ConvergenceWarning` raised |
| `realized_kernel.m` mixes validation into hot path | Separated: `realized_kernel()` vs `realized_kernel_validated()` |
| `dcc.m` recomputes `Q_bar` inside likelihood loop | Pre-computed, passed as constant |
| No sandwich VCV for multivariate estimators | Sandwich estimator standard |
| `realized_hayashi_yoshida.m` has TODO for K > 2 | Implemented for general K |
