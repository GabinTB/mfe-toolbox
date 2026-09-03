# mfe — Financial Econometrics for Python

[![PyPI version](https://img.shields.io/pypi/v/mfe-toolbox.svg)](https://pypi.org/project/mfe-toolbox/)
[![Python](https://img.shields.io/pypi/pyversions/mfe-toolbox.svg)](https://pypi.org/project/mfe-toolbox/)
[![Downloads](https://static.pepy.tech/badge/mfe-toolbox)](https://pepy.tech/project/mfe-toolbox)
[![Downloads/month](https://static.pepy.tech/badge/mfe-toolbox/month)](https://pepy.tech/project/mfe-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://gabintb.github.io/mfe-toolbox/)

Python port of Kevin Sheppard's Oxford MFE Toolbox, optimised for HFT data.
Complements [`arch`](https://arch.readthedocs.io) — covering everything `arch` is missing.

> **Upstream**: ported from [`bashtage/mfe-toolbox`](https://github.com/bashtage/mfe-toolbox) —
> see the [project homepage](https://www.kevinsheppard.com/code/matlab/mfe-toolbox/) and
> [MATLAB Central entry](https://ch.mathworks.com/matlabcentral/fileexchange/170381-mfe-toolbox-kevin-sheppard)
> for the original MATLAB implementation by Kevin Sheppard (Oxford MFE).

## Documentation

Full documentation at **[gabintb.github.io/mfe-toolbox](https://gabintb.github.io/mfe-toolbox/)**, including:

- [Installation & Cython build](https://gabintb.github.io/mfe-toolbox/guide/installation/)
- [Quick start](https://gabintb.github.io/mfe-toolbox/guide/quickstart/)
- [mfe vs. arch vs. statsmodels](https://gabintb.github.io/mfe-toolbox/guide/comparison/)
- [API reference](https://gabintb.github.io/mfe-toolbox/api/realized/)

## What's inside

| Module | Contents |
|--------|----------|
| `mfe.realized` | RV, BPV, MedRV, realized kernel, TSRV, MSRV, Hayashi-Yoshida, multivariate kernel, BNS jump test |
| `mfe.multivariate` | DCC, CCC, BEKK, O-GARCH, GO-GARCH, RCC |
| `mfe.univariate` | HAR-RV (standard / MODIFIED / matrix intervals / HAR-J), HEAVY |
| `mfe.timeseries` | VAR (4 VCV options), Granger causality, IRF, Beveridge-Nelson |
| `mfe.bootstrap` | Wild bootstrap, SPA test (Hansen 2005), StepM (Romano-Wolf 2005) |
| `mfe.crosssection` | Fama-MacBeth, OLS/OLSNW, PCA |
| `mfe.tests_stat` | ARCH-LM, Ljung-Box, HAC LM, Diebold-Mariano, Mincer-Zarnowitz |
| `mfe.distributions` | Skew-t (analytic score), GED, multivariate normal log-likelihood |

## Installation

```bash
pip install mfe-toolbox
```

## Cython extensions (optional, recommended for production)

Cython compilation gives 10–800× speedups on hot paths (realized kernel inner
loop, Hayashi-Yoshida sweep, DCC/BEKK recursions):

```bash
uv run python setup_cython.py build_ext --inplace
```

## Quick start

```python
from mfe.realized import price_filter, returns_from_prices, realized_kernel, bns_jump_test
from mfe.realized._types import SamplingType

prices_5m, times_5m = price_filter(
    tick_prices, tick_times,
    sampling_type=SamplingType.CALENDAR_TIME,
    sampling_interval=300,
)
r = returns_from_prices(prices_5m)
rk  = realized_kernel(r)
jmp = bns_jump_test(r)

from mfe.multivariate import DCC, RCC
dcc = DCC().fit(returns)          # (T, K) → (T, K, K) sigma_t
rcc = RCC().fit(returns)          # covariance targeting by construction

from mfe.univariate import HEAVY
heavy = HEAVY().fit(daily_returns, realized_variances)
```

## Upstream

This package ports the [Oxford MFE Toolbox](https://github.com/bashtage/mfe-toolbox)
by Kevin Sheppard to Python, fixing several bugs present in the MATLAB source
(memory leaks, silent non-convergence, O(N²) algorithms replaced with
O(N log N) Cython implementations). See [`UPSTREAM.md`](UPSTREAM.md) for the
full list of files reviewed, intentional corrections, and deferred items.

## Development

```bash
uv sync --all-groups
PYTHONPATH=src pytest tests/     # 246 tests
mkdocs serve                     # documentation
```

## License

MIT