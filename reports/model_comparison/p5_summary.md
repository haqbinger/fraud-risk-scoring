P5 Model Comparison Summary

Comparison table

| Model | CV mean PR-AUC | CV std | Val PR-AUC | Avg fit time per fold |
|---|---|---|---|---|
| Tuned XGBoost | 0.6167 | 0.0232 | 0.6317 | 92.4s |
| Tuned LightGBM | 0.6316 | 0.0203 | 0.6741 | 93.6s |

Winner: LightGBM (higher CV mean PR-AUC)

Details

- XGBoost CV mean sanity check vs P4: 0.6167 vs 0.6167 (delta +0.0000)
- LightGBM tuning: 50 trials, 10 parallel, 5764.6s (96.1 min)
- LightGBM final model fit time (full train split): 307.5s

LightGBM best params

```json
{
  "n_estimators": 1750,
  "max_depth": 10,
  "learning_rate": 0.04155210381881139,
  "num_leaves": 85,
  "subsample": 0.7574988675745632,
  "colsample_bytree": 0.8097431291671384,
  "min_child_samples": 67,
  "scale_pos_weight": 20.413893170178223
}
```
