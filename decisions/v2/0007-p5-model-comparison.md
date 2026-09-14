ADR V2-0007: P5 Model Comparison

Status

Accepted — supersedes ADR V2-0004 (XGBoost as incumbent)

Method

Tuned XGBoost (best_params.json from P4) vs Tuned LightGBM (50-trial Optuna).
Same 182-feature set, same 5-fold temporal CV objective as P3/P4.

Results

| Model          | CV mean PR-AUC | CV std | Val PR-AUC | Avg fit time/fold |
| -------------- | -------------- | ------ | ---------- | ----------------- |
| Tuned XGBoost  | 0.6167         | 0.0232 | 0.6317     | 92.4s             |
| Tuned LightGBM | 0.6316         | 0.0203 | 0.6741     | 93.6s             |

XGBoost sanity check: re-evaluating best_params.json reproduced P4's 0.6167 exactly.

Best LightGBM Config

| Param             | Value |
| ----------------- | ----- |
| n_estimators      | 1750  |
| max_depth         | 10    |
| learning_rate     | 0.042 |
| num_leaves        | 85    |
| subsample         | 0.757 |
| colsample_bytree  | 0.810 |
| min_child_samples | 67    |
| scale_pos_weight  | 20.4  |

Key Finding

LightGBM's scale_pos_weight converged to 20.4 (close to natural 1/fraud_rate ≈ 28.6) vs XGBoost's 53.9 (nearly 2x overcorrection). LightGBM's leaf-wise growth with num_leaves control handles class imbalance more natively — XGBoost needed heavier explicit correction to compensate.

LightGBM also shows lower fold variance (std 0.0203 vs 0.0232) — more stable across the temporal drift pattern observed in P3.

Decision

LightGBM is the final model. XGBoost retired as incumbent. Proceed to P6 error analysis on LightGBM predictions.

Artifacts

- reports/model_comparison/p5_results.csv
- reports/model_comparison/lgbm_best_params.json
- reports/model_comparison/p5_summary.md
