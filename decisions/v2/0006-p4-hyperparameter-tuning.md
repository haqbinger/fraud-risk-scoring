ADR V2-0006: P4 Hyperparameter Tuning

Status

Accepted

Method

Optuna, 50 trials, objective = mean PR-AUC across P3's exact 5 temporal CV folds.
Search space: n_estimators 200-1000, max_depth 3-8, learning_rate 0.01-0.3 (log), subsample 0.6-1.0, colsample_bytree 0.6-1.0, min_child_weight 1-10, gamma 0-5, scale_pos_weight 20-60.

Results

| Metric                                 | Value                             |
| -------------------------------------- | --------------------------------- |
| Default CV mean PR-AUC (P3)            | 0.5642                            |
| Tuned CV mean PR-AUC                   | 0.6167 (+0.0525)                  |
| Val PR-AUC (best config, single split) | 0.6317                            |
| Total tuning time                      | 43.5 min (50 trials, 10 parallel) |
| Best trial                             | #34                               |

Best Params

| Param            | Value |
| ---------------- | ----- |
| n_estimators     | 1000  |
| max_depth        | 8     |
| learning_rate    | 0.069 |
| subsample        | 0.825 |
| colsample_bytree | 0.729 |
| min_child_weight | 1     |
| gamma            | 0.268 |
| scale_pos_weight | 53.9  |

Key Finding — Binding Constraints

Top trials consistently converged toward n_estimators=1000 (upper bound), max_depth 7-8 (upper bound), scale_pos_weight 45-55 (near 60 cap). The textbook imbalance correction is 1/fraud_rate ≈ 28.6 — optimizer converged to 53.9, nearly 2x higher. When the optimum sits at the search space boundary rather than the interior, the true optimum likely lies outside the current bounds. A wider search pass (n_estimators to 2000, max_depth to 10, scale_pos_weight to 80) could yield further gains if needed.

Decision

Proceed to P5 with tuned XGBoost as the incumbent. Add LightGBM to the comparison using the same CV objective. If LightGBM beats tuned XGBoost on CV mean PR-AUC, LightGBM becomes the final model.

Artifacts

- reports/tuning/optuna_trials.csv
- reports/tuning/best_params.json
- reports/tuning/tuning_summary.md
