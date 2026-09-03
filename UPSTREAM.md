# Upstream snapshot

This Python port was prepared against:

## Primary source

- **Repository**: `bashtage/mfe-toolbox`
- **GitHub**: https://github.com/bashtage/mfe-toolbox
- **Branch**: `main`
- **Author**: Kevin Sheppard (Oxford MFE)
- **Homepage**: https://www.kevinsheppard.com/code/matlab/mfe-toolbox/
- **MATLAB Central**: https://ch.mathworks.com/matlabcentral/fileexchange/170381-mfe-toolbox-kevin-sheppard
- **Last upstream update**: June 7, 2013 (stable since; GitHub repo is the living version)

## Files and modules reviewed for the port

### Realized measures (`realized/`)
- `realized/realized_variance.m`
- `realized/realized_bipower_variation.m`
- `realized/realized_kernel.m`
- `realized/realized_multivariate_kernel.m`
- `realized/realized_quantile_variance.m`
- `realized/realized_range.m`
- `realized/realized_covariance.m`
- `realized/realized_hayashi_yoshida.m`
- `realized/realized_noise_variance.m`
- `realized/realized_preaveraged_variance.m`
- `realized/realized_refresh_time.m`
- `realized/tsrv.m`
- `realized/msrv.m`
- `realized/realized_med_variance.m`, `realized/realized_min_variance.m`
- `realized/realized_quarticity.m`, `realized/realized_tripower_quarticity.m`
- `realized/jump_test_bns.m`

### Multivariate GARCH (`multivariate/`)
- `multivariate/dcc.m`, `multivariate/scalar_dcc.m`
- `multivariate/bekk.m` (scalar, diagonal, full)
- `multivariate/ccc.m`
- `multivariate/ogarch.m`, `multivariate/gogarch.m`
- `multivariate/rcc.m` (Noureldin, Shephard & Sheppard 2014)
- `multivariate/heavy.m` (multivariate extension)

### Univariate (`univariate/`, `timeseries/`)
- `timeseries/heterogeneousar.m` (HAR-RV)
- `timeseries/heavy.m`
- `timeseries/vectorar.m`, `timeseries/grangercause.m`, `timeseries/impulseresponse.m`
- `timeseries/beveridgenelson.m`

### Bootstrap (`bootstrap/`)
- `bootstrap/stationary_bootstrap.m`
- `bootstrap/block_bootstrap.m`
- `bootstrap/spa.m` (Hansen 2005 SPA / White 2000 Reality Check)
- `bootstrap/stepm.m` (Romano & Wolf 2005)

### Cross-section and utilities
- `crosssection/ols.m`, `crosssection/olsnw.m`
- `crosssection/fama_macbeth.m`
- `utility/newey_west.m`
- `utility/vcv.m` (sandwich estimator)

### Tests (`tests/`)
- `tests/ljungbox.m`
- `tests/lmtest1.m`
- `tests/arch_test.m`
- `tests/dieboldmariano.m`
- `tests/mincerzarnowitz.m`

## Coverage notes

The following upstream functions were **intentionally not ported** — covered by existing Python packages:

| Upstream function | Reason skipped | Python alternative |
|---|---|---|
| GARCH/EGARCH/TARCH/APARCH estimation | Fully covered | `arch` package |
| FIGARCH, HARCH, MIDAS | Fully covered | `arch` package |
| Unit root tests (ADF, PP, KPSS, DFGLS) | Fully covered | `arch.unitroot`, `statsmodels` |
| Cointegration (Johansen, Engle-Granger) | Fully covered | `statsmodels` |
| ARMA/ARMAX estimation | Fully covered | `statsmodels` |
| Kernel density estimation | Fully covered | `scipy.stats` |
| Jarque-Bera, KS, Berkowitz tests | Fully covered | `scipy.stats`, `statsmodels` |
| Baxter-King / HP filter | Macro focus, fully covered | `statsmodels.tsa.filters` |
| GUI components | Not applicable to Python | — |
| MEX/DLL files | Replaced by Cython extensions | `mfe.realized._core`, `mfe.multivariate._core` |

The following upstream functions are **on the roadmap** (not yet ported):

| Upstream function | Status |
|---|---|
| `realized/jump_test_abd.m` | Planned |
| `realized/jump_test_asj.m` | Planned |
| `realized/jump_test_jo.m` | Planned |
| `realized/multipower_variation.m` | Planned |
| `realized/truncated_multipower_variation.m` | Planned |
| `realized/modulated_multipower_variation.m` | Planned |
| `realized/qmle_realized_variance.m` | Planned |
| `bootstrap/mcs.m` (Model Confidence Set) | Planned |
| `multivariate/adcc.m` (Asymmetric DCC) | Planned |

## Intentional corrections and Python-specific choices

1. **`gogarch.m` memory leak**: closes over `volData` inside an `fmincon` anonymous function in a loop. Python port uses no closures; all optimisation inputs are explicit.
2. **Silent non-convergence**: all MATLAB estimators fire `MFEToolbox:Convergence` then continue with non-converged parameters. Python port raises `ConvergenceWarning` instead.
3. **`realized_kernel.m` hot path**: parameter validation was mixed into the inner loop. Separated in the Python port.
4. **`realized_bipower_variation.m` `skip` default**: documented as 0 in help text but implemented inconsistently. Standardised to 0.
5. **`dcc.m` `Q_bar`**: recomputed on every likelihood call. Pre-computed once in the Python port.
6. **No sandwich VCV**: MATLAB multivariate estimators return only inverse-Hessian VCV. Python port computes sandwich (robust) VCV as standard.
7. **`realized_hayashi_yoshida.m` TODO**: K > 2 assets not implemented in MATLAB. Python port implements the general K-asset case via O((N₁+N₂) log(N₁+N₂)) Cython sweep-line.
8. **Cython sort key precision**: initial implementation used `time * 4 + type` as a float sort key, losing bits for large nanosecond timestamps. Fixed with `np.lexsort((type_arr, time_arr))`.
9. **Analytic gradients**: MATLAB Skew-t and GED use finite-difference gradients. Python port provides analytic score functions.