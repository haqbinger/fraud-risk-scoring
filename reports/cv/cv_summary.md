P3 Temporal Cross-Validation Summary

- Folds: 5 (rolling window, train/val block size ~83659 rows each)
- Mean val PR-AUC: 0.5642
- Std val PR-AUC: 0.0287
- Min val PR-AUC: 0.5406
- Max val PR-AUC: 0.5979
- P2 single-split val PR-AUC: 0.5888
- |mean - single-split|: 0.0246

Verdict

Single-split estimate (0.5888) was RELIABLE -- CV mean is within 0.03 (0.0246) and std (0.0287) is below 0.05.

Per-fold results

| fold | train_size | val_size | val_pr_auc | fit_time_s |
|---|---|---|---|---|
| 1 | 83659 | 83659 | 0.5426 | 5.9 |
| 2 | 83659 | 83659 | 0.5979 | 3.5 |
| 3 | 83659 | 83659 | 0.5932 | 2.9 |
| 4 | 83659 | 83659 | 0.5467 | 3.1 |
| 5 | 83659 | 83664 | 0.5406 | 3.1 |
