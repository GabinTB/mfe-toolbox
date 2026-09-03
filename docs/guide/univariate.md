# Univariate Models

## HAR-RV

```python
from mfe.univariate import har_rv, har_rv_j, har_forecast

# Standard HAR (Corsi 2009)
res = har_rv(rv_daily, p=[1, 5, 22])

# Non-overlapping intervals (MODIFIED spec)
res_mod = har_rv(rv_daily, p=[1, 5, 22], spec="modified")
# intervals become [1,1], [2,5], [6,22]

# Matrix notation — explicit intervals
res_mat = har_rv(rv_daily, p=[[1,1],[2,5],[6,22]])

# Jump-augmented
jump = np.maximum(rv - bpv, 0)
res_j = har_rv_j(rv_daily, jump)

# Forecast
fc = har_forecast(res, rv_daily[-30:], horizon=5)
```

## HEAVY

```python
from mfe.univariate import HEAVY

heavy = HEAVY().fit(daily_returns, realized_variance)
# Two equations:
#   h_{r,t}  = ω_r  + α_r  * RM_{t-1} + β_r  * h_{r,t-1}
#   h_{RM,t} = ω_RM + α_RM * RM_{t-1} + β_RM * h_{RM,t-1}

# Multi-step forecast
h_r_fc, h_rm_fc = HEAVY().forecast(heavy, horizon=10)
```
