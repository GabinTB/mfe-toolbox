# mfe vs. arch vs. statsmodels

## Decision guide

| Task | Use |
|---|---|
| GARCH / EGARCH / TARCH / APARCH estimation | [`arch`](https://arch.readthedocs.io) |
| FIGARCH, HARCH, MIDAS-GARCH | `arch` |
| Unit root tests (ADF, PP, KPSS, DFGLS) | `arch.unitroot` |
| Cointegration (Johansen, Engle-Granger) | `statsmodels` |
| ARMA/ARMAX estimation | `statsmodels` |
| VAR with robust VCV options | **`mfe`** |
| Granger causality with heteroskedastic VCV | **`mfe`** |
| Impulse response functions under heteroskedasticity | **`mfe`** |
| Realized variance / BPV / MedRV | **`mfe`** |
| Realized kernel (noise-robust) | **`mfe`** |
| TSRV / MSRV (two-scale noise correction) | **`mfe`** |
| Hayashi-Yoshida non-synchronous covariance | **`mfe`** |
| Multivariate realized kernel (PSD guaranteed) | **`mfe`** |
| DCC-GARCH | **`mfe`** |
| BEKK-GARCH | **`mfe`** |
| CCC-GARCH | **`mfe`** |
| GO-GARCH / O-GARCH | **`mfe`** |
| RCC (Rotated Conditional Correlation) | **`mfe`** |
| HAR-RV model | **`mfe`** |
| HEAVY model (realized variance in mean equation) | **`mfe`** |
| Beveridge-Nelson decomposition | **`mfe`** |
| SPA test / StepM FWER | **`mfe`** |
| Wild bootstrap for realized volatility | **`mfe`** |
| Fama-MacBeth regression | **`mfe`** |
| OLS with White / Newey-West SEs | **`mfe`** (thin wrapper) or `statsmodels` |
| PCA with financial conventions | **`mfe`** |
| Hansen Skew-t / GED distributions (standalone) | **`mfe`** |

## Key API differences

### arch (univariate, correct home for GARCH)

```python
from arch import arch_model
am = arch_model(returns, vol="Garch", p=1, q=1)
res = am.fit()
```

### mfe (multivariate, realized, everything arch doesn't cover)

```python
from mfe.multivariate import DCC
from mfe.realized import realized_kernel

dcc = DCC().fit(returns)          # (T, K) → (T, K, K) sigma_t
rk = realized_kernel(r)           # noise-robust RV
```

### statsmodels (ARIMA, VAR basic, cointegration)

```python
from statsmodels.tsa.api import VAR
mod = VAR(data)
res = mod.fit(maxlags=2)
# → no robust VCV, no GC test with het-robust options
```

```python
from mfe.timeseries import vectorar, grangercause
res = vectorar(data, lags=2, het=True)          # White-robust VCV
gc  = grangercause(data, lags=2, method="wald") # Wald test with robust VCV
```

## What mfe does NOT replace

- `arch` for any univariate GARCH estimation — we use `arch` internally for
  the first step of DCC/RCC/CCC.
- `statsmodels` for ARIMA, SARIMA, SARIMAX estimation.
- `scipy.signal` for filtering.
- `sklearn.decomposition.PCA` for general-purpose PCA (our PCA is tailored to
  financial returns with covariance convention and factor interpretation tools).
