# Cross-Section

## OLS and OLSNW

```python
from mfe.crosssection import ols, olsnw

# Y = alpha + X beta + eps
res = ols(y, X)                      # White-robust SEs
res_nw = olsnw(y, X, nw_lags=6)     # Newey-West HAC SEs

print(res.params, res.std_errors, res.r_squared)
```

## Fama-MacBeth

```python
from mfe.crosssection import fama_macbeth, rolling_betas

# Pass 1: rolling betas
betas = rolling_betas(returns, factors, window=60)   # (N, K)

# Pass 2: FM
fm = fama_macbeth(returns, betas, include_intercept=True, shanken_correction=True)
print(fm.lambda_mean)        # (K+1,) mean risk premia
print(fm.t_stats_shanken)    # Shanken-corrected t-stats
print(fm.r_squared_mean)     # mean cross-sectional R²
```

## PCA

```python
from mfe.crosssection import pca

res = pca(returns, n_components=3)
print(res.explained_variance)          # proportion per component
print(res.cumulative_variance)         # cumulative
factors = res.factors                  # (T, 3) principal components
loadings = res.loadings                # (K, 3) factor loadings
recon = res.reconstruct(k_c=3)        # (T, K) reconstruction from 3 PCs
```
