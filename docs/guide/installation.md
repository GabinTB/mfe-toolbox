# Installation

## Requirements

- Python 3.11+
- numpy >= 1.26
- scipy >= 1.11
- pandas >= 2.1
- arch >= 6.2  (univariate GARCH base)
- statsmodels >= 0.14  (VAR, cointegration)

## PyPI (pure Python)

```bash
pip install mfe
```

## With Cython extensions (recommended for production)

```bash
pip install mfe[cython]
```

This compiles the `realized._core` and `multivariate._core` Cython extensions.
Binary wheels are provided for Linux x86-64, macOS, and Windows (Python 3.11-3.13).

## From source

```bash
git clone https://github.com/your-org/mfe-toolbox
cd mfe-toolbox
uv sync --group dev
uv run python setup_cython.py build_ext --inplace   # optional Cython
```

## With uv (recommended)

```bash
uv add mfe
# or for the latest development version:
uv add git+https://github.com/your-org/mfe-toolbox
```

## Verifying Cython compilation

```python
from mfe.realized._core import _autocovariance_sum
print("Cython compiled: OK")
```

If this raises `ImportError`, the pure-Python fallback is used automatically.
All functions work correctly without Cython — it is a performance optimisation only.

## Performance without Cython

| Operation | numpy | Cython |
|---|---|---|
| Realized kernel (T=500K) | 34 ms | 7 ms |
| MedRV (T=500K) | 62 ms | 1 ms |
| Hayashi-Yoshida (N=10K) | 730 ms | < 1 ms |
| DCC Q-recursion (T=2000, K=5) | 8.7 ms | 0.06 ms |
| BEKK scalar (T=2000, K=2) | 32 ms | 0.04 ms |
