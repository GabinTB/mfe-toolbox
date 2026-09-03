# Realized Volatility

`mfe.realized` contains the full realized-measure library for HFT data.

## Sampling

```python
from mfe.realized import price_filter, returns_from_prices, refresh_time
from mfe.realized._types import SamplingType

# Calendar-time (5-min)
prices_5m, times_5m = price_filter(
    prices, times,
    sampling_type=SamplingType.CALENDAR_TIME,
    sampling_interval=300,
)
r = returns_from_prices(prices_5m)

# Synchronise K asynchronous series
sync_prices, sync_times = refresh_time(prices_list, times_list)
```

## Variance estimators

| Function | Description | Jump-robust |
|---|---|---|
| `realized_variance` | Sum of squared returns | No |
| `realized_bipower_variation` | Skip-k BPV | Partially |
| `realized_med_variance` | Median of triplets | Yes |
| `realized_min_variance` | Min of pairs | Yes |
| `realized_preaveraged_variance` | Jacod et al. (2009) | Yes + noise |
| `realized_semivariance` | Positive/negative decomposition | No |
| `realized_quantile_variance` | Quantile-truncated (τ=0.5) | Yes |
| `tsrv` | Two-Scale RV (Zhang et al. 2005) | No, noise-robust |
| `msrv` | Multi-Scale RV (Zhang 2006) | No, noise-robust |

## Realized kernel

```python
from mfe.realized import realized_kernel, select_bandwidth
from mfe.realized._types import KernelType

rk = realized_kernel(r, kernel_type=KernelType.PARZEN)
# Auto bandwidth, Parzen kernel, end-point jitter correction
```

Available kernels: `PARZEN`, `BARTLETT`, `TUKEY_HANNING`, `CUBIC`, `EPANECHNIKOV`, `FLAT_TOP`.

## Covariance

```python
from mfe.realized import (
    realized_covariance,           # synchronous returns
    realized_hayashi_yoshida,      # non-synchronous (K assets)
    realized_covariance_refresh_time,
    realized_multivariate_kernel,  # PSD-guaranteed (K,K)
)
```

## Jump detection

```python
from mfe.realized import bns_jump_test

jmp = bns_jump_test(r, alpha=0.05)
print(jmp.statistic, jmp.p_value, jmp.significant)
print(jmp.jump_variation, jmp.continuous_variation)
```

## Microstructure noise

```python
from mfe.realized import estimate_noise_variance
omega2 = estimate_noise_variance(r, method="bandi-russell")
```
