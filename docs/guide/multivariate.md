# Multivariate GARCH

All models in `mfe.multivariate` are missing from the `arch` package.

## Model summary

| Model | Params | PSD guaranteed | Notes |
|---|---|---|---|
| CCC | 3K (GARCH) | Yes | Constant correlation |
| DCC | 3K + 2 | Yes | Dynamic correlation |
| BEKK scalar | K(K+1)/2 + 2 | Yes | Covariance targeting |
| BEKK diagonal | K(K+1)/2 + 2K | Yes | Per-asset persistence |
| OGARCH | 3K (GARCH) | Yes | PCA factors |
| GOGARCH | 3K + K(K-1)/2 | Yes | Independent factors |
| RCC | 3K (GARCH) + 2 | Yes | Rotation + targeting |

## DCC

```python
from mfe.multivariate import DCC

dcc = DCC(variant="dcc").fit(returns)  # (T, K)
sigma_t = dcc.conditional_covariances   # (T, K, K)
print(f"a={dcc.diagnostics['a']:.4f}  b={dcc.diagnostics['b']:.4f}")
```

## RCC

```python
from mfe.multivariate import RCC

rcc = RCC(rotation="symmetric").fit(returns)
# G_t recursion in rotated space: G_t = (1-a-b)I + a u_{t-1}u_{t-1}' + b G_{t-1}
# u_t = P^{-1/2} r_t;  Sigma_t = P^{1/2} G_t P^{1/2}
print(f"a={rcc.a:.4f}  b={rcc.b:.4f}")
corr_t = rcc.conditional_correlations()  # (T, K, K) — diagonal = 1
```

## BEKK

```python
from mfe.multivariate import BEKK

bekk = BEKK("scalar").fit(returns)
# H_t = C'C + a² eps_{t-1}eps_{t-1}' + b² H_{t-1}
print(bekk.diagnostics["a"], bekk.diagnostics["b"])
```

## GO-GARCH

```python
from mfe.multivariate import GOGARCH

gg = GOGARCH(rotation="ica", n_components=None).fit(returns)
# W = W_pca @ U.T;  Sigma_t = W diag(h_{k,t}) W'
print(f"Rotation U orthogonality: {np.max(np.abs(gg.rotation_matrix @ gg.rotation_matrix.T - np.eye(K))):.2e}")
```
