# Changelog

## 0.1.0 (2026-09)

Initial release — Python port of the Oxford MFE Toolbox.

### mfe.realized
- `realized_variance`, `realized_bipower_variation` (skip-k), `realized_med_variance`, `realized_min_variance`, `realized_preaveraged_variance`, `realized_semivariance`, `realized_quantile_variance`
- `realized_kernel` (Parzen/Bartlett/Tukey-Hanning/Cubic/Epanechnikov/FlatTop, auto-bandwidth)
- `tsrv`, `msrv` — Two-Scale and Multi-Scale Realized Variance
- `realized_covariance`, `realized_correlation`, `realized_hayashi_yoshida`, `realized_covariance_refresh_time`
- `realized_multivariate_kernel` — PSD-guaranteed (K,K) multivariate realized kernel
- `realized_range`, `realized_range_from_ticks`
- `bns_jump_test`, `estimate_noise_variance`, `realized_quarticity`, `realized_tripower_quarticity`
- `price_filter`, `returns_from_prices`, `refresh_time`
- Cython extensions: `_autocovariance_sum`, `_bpv_sum`, `_medvar_triplets`, `_hy_sweep`, `_refresh_time_indices`

### mfe.multivariate
- `DCC` (Engle 2002) with cDCC and DECO variants
- `CCC` (Bollerslev 1990)
- `BEKK` scalar and diagonal (Engle & Kroner 1995)
- `OGARCH` (Alexander 2001)
- `GOGARCH` ICA and moments rotation (van der Weide 2002)
- `RCC` symmetric and Cholesky rotation (Noureldin, Shephard & Sheppard 2014)
- Cython extensions: `_dcc_q_recursion`, `_dcc_corr_loglik`, `_bekk_scalar_recursion`, `_bekk_diagonal_recursion`

### mfe.univariate
- `har_rv` — standard, MODIFIED spec, matrix intervals
- `har_rv_j` — jump-augmented HAR
- `har_forecast` — multi-step forecasting
- `HEAVY` — joint model of returns + realized variance (Shephard & Sheppard 2010)

### mfe.timeseries
- `vectorar` — VAR(P) with 4 VCV options (homo/het × corr/uncorr)
- `grangercause` — Granger causality LR/LM/Wald with robust VCV
- `impulse_response` — IRF with delta-method standard errors
- `beveridge_nelson` — AR and state-space methods, auto order selection

### mfe.bootstrap
- `wild_bootstrap_rv`, `wild_bootstrap_test` — Rademacher/Mammen/Normal multipliers
- `spa_test` — Hansen (2005) SPA: consistent, upper (Reality Check), lower p-values
- `step_m` — Romano & Wolf (2005) stepdown FWER control

### mfe.crosssection
- `ols`, `olsnw` — OLS with White / Newey-West SEs
- `fama_macbeth` — Fama-MacBeth with Shanken correction
- `rolling_betas`, `pca`

### mfe.tests_stat
- `ljung_box` — Ljung-Box Q statistic
- `lm_test` — HAC-robust LM serial correlation test (MFE lmtest1.m)
- `arch_lm` — Engle (1982) ARCH-LM test (LM form + F form)
- `mincer_zarnowitz` — MZ regression forecast evaluation
- `diebold_mariano` — DM test (MSE/MAE/QLIKE loss)

### mfe.distributions
- `skewt_logpdf`, `skewt_ppf`, `skewt_score` — Hansen (1994) Skew-t with analytic gradient
- `ged_logpdf`, `ged_ppf`, `ged_score` — Generalized Error Distribution
- `mvnorm_loglik`, `mvnorm_loglik_t`, `mahalanobis`, `standardize_mvn`

### mfe.utils
- `lag_matrix`, `har_lag_matrix`
- `sandwich`, `newey_west`
- Type aliases: `FloatArray`, `IntArray`, etc.

### Bugs fixed vs. MATLAB source
- `gogarch.m`: memory leak via closure over `volData` inside `fmincon` loop → eliminated
- All univariate estimators: `MFEToolbox:Convergence` silently used bad params → `ConvergenceWarning`
- `realized_kernel.m`: parameter validation mixed into hot path → separated
- `realized_bipower_variation.m`: inconsistent `skip` default → standardised to 0
- `dcc.m`: `Q_bar` recomputed inside likelihood → pre-computed
- No sandwich VCV in MATLAB multivariate → standard in all `mfe` estimators
- `realized_hayashi_yoshida.m` TODO for K>2 → implemented for general K
- HY Cython: `sort_key = time * 4 + type` loses bits for large timestamps → `np.lexsort`
