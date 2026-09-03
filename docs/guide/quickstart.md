# Quick Start

## Realized volatility from tick data

```python
import numpy as np
from mfe.realized import (
    price_filter, returns_from_prices,
    realized_variance, realized_bipower_variation,
    realized_kernel, realized_med_variance,
    bns_jump_test, estimate_noise_variance,
)
from mfe.realized._types import SamplingType

# --- Your tick data ---
# tick_prices : (N,) array of mid-prices
# tick_times  : (N,) array of timestamps in seconds since session open

# 1. Filter to 5-minute calendar-time grid
prices_5m, times_5m = price_filter(
    tick_prices, tick_times,
    sampling_type=SamplingType.CALENDAR_TIME,
    sampling_interval=300,
)

# 2. Log-returns
r = returns_from_prices(prices_5m, log=True)

# 3. Standard estimators
rv   = realized_variance(r)
bpv  = realized_bipower_variation(r)
medrv = realized_med_variance(r)       # jump-robust
rk   = realized_kernel(r)             # noise-robust (Parzen kernel, auto bandwidth)
noise = estimate_noise_variance(r)    # microstructure noise σ²

# 4. Jump test
jmp = bns_jump_test(r)
print(f"RV  = {rv.value:.2e}")
print(f"BPV = {bpv.value:.2e}  (continuous variation)")
print(f"RK  = {rk.rk_adjusted:.2e}  (H={rk.bandwidth})")
print(f"Jump: {jmp.significant}  (Z={jmp.statistic:.2f}, p={jmp.p_value:.3f})")
```

## Multivariate realized covariance

```python
from mfe.realized import (
    refresh_time, realized_covariance,
    realized_hayashi_yoshida, realized_multivariate_kernel,
)

# Non-synchronous: Hayashi-Yoshida for each asset pair
prices = [price_asset1, price_asset2, price_asset3]
times  = [times_asset1, times_asset2, times_asset3]

cov_hy = realized_hayashi_yoshida(prices, times)
print(cov_hy.cov)   # (3, 3) covariance matrix

# Synchronised: refresh-time then realized kernel (PSD guaranteed)
sync_prices, sync_times = refresh_time(prices, times)
sync_returns = np.column_stack([np.diff(np.log(p)) for p in sync_prices])
mk = realized_multivariate_kernel(sync_returns)
print(mk.rk_adjusted)   # (3, 3) PSD covariance
```

## HAR-RV model

```python
from mfe.univariate import har_rv, har_rv_j, har_forecast
from mfe.realized import realized_variance, realized_bipower_variation

# Compute daily RV and BPV from intraday returns
rv_series  = np.array([realized_variance(r_day).value for r_day in intraday_returns])
bpv_series = np.array([realized_bipower_variation(r_day).value for r_day in intraday_returns])
jump_series = np.maximum(rv_series - bpv_series, 0.0)   # daily jump contribution

# Standard HAR
har = har_rv(rv_series, p=[1, 5, 22])
print(har.params)           # [const, β_d, β_w, β_m]
print(har.r_squared)

# HAR with matrix intervals (non-overlapping)
har_mod = har_rv(rv_series, p=[1, 5, 22], spec="modified")
# intervals: [1,1], [2,5], [6,22] — same fit, cleaner interpretation

# Jump-augmented HAR
har_j = har_rv_j(rv_series, jump_series)
print(har_j.param_names)    # [..., 'Jump_lag1']

# 5-day forecast
fc = har_forecast(har, rv_series[-30:], horizon=5)
```

## HEAVY model

```python
from mfe.univariate import HEAVY

# Daily returns + daily realized variance
heavy = HEAVY().fit(daily_returns, realized_variance_series)
print(f"ω_r={heavy.params[0]:.4f}  α_r={heavy.params[1]:.4f}  β_r={heavy.params[2]:.4f}")
print(f"ω_m={heavy.params[3]:.4f}  α_m={heavy.params[4]:.4f}  β_m={heavy.params[5]:.4f}")

# 10-day forecast
h_r_fc, h_rm_fc = HEAVY().forecast(heavy, horizon=10)
```

## Multivariate GARCH

```python
from mfe.multivariate import DCC, RCC, BEKK, GOGARCH

# DCC (two-step QML)
dcc = DCC().fit(returns)   # (T, K)
sigma_t = dcc.conditional_covariances  # (T, K, K)

# RCC — covariance targeting by construction, same 2 params
rcc = RCC().fit(returns)
print(f"a={rcc.a:.4f}  b={rcc.b:.4f}")

# BEKK scalar (numerically robust for K <= 10)
bekk = BEKK("scalar").fit(returns)

# GO-GARCH — independent factors via ICA rotation
gg = GOGARCH(rotation="ica").fit(returns)
print(f"U orthogonality: {np.max(np.abs(gg.rotation_matrix @ gg.rotation_matrix.T - np.eye(K))):.2e}")
```

## Bootstrap model comparison

```python
from mfe.bootstrap import spa_test, step_m

# SPA: does any model beat the benchmark?
res_spa = spa_test(loss_benchmark, loss_models, n_bootstrap=999)
print(f"SPA p-value (consistent): {res_spa.p_value_consistent:.3f}")

# StepM: which models beat the benchmark? (FWER-controlled)
res_step = step_m(loss_benchmark, loss_models, alpha=0.05, n_bootstrap=999)
print(f"Models that beat benchmark: {res_step.rejected}")
```

## Beveridge-Nelson decomposition

```python
from mfe.timeseries import beveridge_nelson

# Decompose log GDP into permanent (trend) and transitory (cycle)
res = beveridge_nelson(log_gdp, ic="bic")
print(f"AR order selected: {res.ar_order}")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(log_gdp, label="Original", alpha=0.7)
axes[0].plot(res.trend, label="BN Trend")
axes[0].legend()
axes[1].plot(res.cycle, label="BN Cycle")
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].legend()
plt.tight_layout()
```
