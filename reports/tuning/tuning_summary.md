# P4 Hyperparameter Tuning Summary

- Trials: 50 (10 parallel)
- Total tuning time: 2610.9s (43.5 min)
- Default (P3, default hyperparams) CV mean PR-AUC: 0.5642
- Tuned CV mean PR-AUC: 0.6167
- Delta vs default: +0.0525
- Val PR-AUC (best config, single split, same as P2): 0.6317
- Final model fit time (full train split): 353.6s

## Best params

```json
{
  "n_estimators": 1000,
  "max_depth": 8,
  "learning_rate": 0.06872119535632425,
  "subsample": 0.824660681692089,
  "colsample_bytree": 0.7292519555709781,
  "min_child_weight": 1,
  "gamma": 0.26805605321682324,
  "scale_pos_weight": 53.916316364868706
}
```
