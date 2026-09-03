# Bootstrap & Multiple Testing

## Wild bootstrap (realized volatility)

```python
from mfe.bootstrap import wild_bootstrap_rv, wild_bootstrap_test

# 95% CI for a realized volatility statistic
result = wild_bootstrap_rv(r, n_replications=999, multiplier="rademacher")
print(f"Statistic: {result.statistic:.6f}")
print(f"95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
```

## SPA test (Superior Predictive Ability)

```python
from mfe.bootstrap import spa_test

# loss_benchmark: (T,) loss for benchmark model
# loss_models:    (T, M) losses for M alternatives  (lower = better)
res = spa_test(loss_benchmark, loss_models, n_bootstrap=999)
print(f"SPA p-value (consistent): {res.p_value_consistent:.3f}")
print(f"Reality Check p-value:    {res.p_value_upper:.3f}")
```

## StepM (Romano-Wolf FWER control)

```python
from mfe.bootstrap import step_m

res = step_m(loss_benchmark, loss_models, alpha=0.05, n_bootstrap=999)
print(f"Models beating benchmark: {res.rejected}")  # indices, 0-based
print(f"FWER-adjusted p-values:   {res.p_values_adjusted.round(3)}")
```
