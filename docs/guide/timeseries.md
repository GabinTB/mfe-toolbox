# Time Series

## VAR

```python
from mfe.timeseries import vectorar, grangercause, impulse_response

# VAR(2) with heteroskedasticity-robust VCV
res = vectorar(data, lags=2, het=True, uncorr=False)
print(res.params)        # list of (K,K) Phi_p matrices
print(res.sigma)         # (K,K) residual covariance
print(res.aic, res.bic)

# Granger causality
gc = grangercause(data, lags=2, method="lr")
# gc.statistics[i,j]: stat for "y_j does not GC y_i"
# gc.p_values[i,j]: corresponding p-value

# IRF with std errors
irf = impulse_response(data, lags=2, horizon=12, decomp="cholesky")
# irf.responses: (K, K, 13)  — responses[i,j,h] = y_i to shock in y_j at h
# irf.std_errors: same shape, delta-method std errors
```

## Beveridge-Nelson decomposition

```python
from mfe.timeseries import beveridge_nelson

# I(1) series (e.g. log price, log GDP)
res = beveridge_nelson(y, ic="bic")   # auto AR order selection
# or
res = beveridge_nelson(y, ar_order=4, method="state_space")

print(f"AR order: {res.ar_order}")
print(f"Drift: {res.drift:.4f}")
trend = res.trend   # random walk (permanent)
cycle = res.cycle   # stationary (transitory)
# trend + cycle == y exactly
```
