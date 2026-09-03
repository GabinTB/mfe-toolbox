# Distributions

## Skew-t (Hansen 1994)

```python
from mfe.distributions import skewt_logpdf, skewt_ppf, skewt_score

# Log-PDF for GARCH residuals
ll = skewt_logpdf(residuals, nu=8.0, lam=0.1)

# Quantile (for VaR)
var_95 = skewt_ppf(np.array([0.05]), nu=8.0, lam=0.1)

# Analytic score w.r.t. (nu, lam) — used in MLE
score_nu, score_lam = skewt_score(residuals, nu=8.0, lam=0.1)
```

## GED (Generalized Error Distribution)

```python
from mfe.distributions import ged_logpdf, ged_ppf

ll = ged_logpdf(residuals, nu=1.5)   # nu=1: Laplace, nu=2: Normal
var_95 = ged_ppf(np.array([0.05]), nu=1.5)
```

## Multivariate normal

```python
from mfe.distributions import mvnorm_loglik, mahalanobis, standardize_mvn

# Time-varying Sigma_t
ll = mvnorm_loglik(returns, sigma_t)    # (T, K) returns, (T, K, K) sigma_t

# Mahalanobis distances — diagnostic
d = mahalanobis(returns, sigma_t)       # (T,)

# Standardised residuals — should be iid N(0, I_K)
z = standardize_mvn(returns, sigma_t)   # (T, K)
```
