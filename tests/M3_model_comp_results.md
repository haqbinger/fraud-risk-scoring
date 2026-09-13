This time we use XGBoost's native NaN-handling in comparison to prev models :

Loaded 590540 rows, fraud rate = 3.4990%
Temporal split — train: 413378 (2017-12-01 00:00:00 to 2018-03-30 19:26:36) | val: 88581 | test: 88581

--- LogisticRegression -- VAL ---
ROC-AUC: 0.7463
PR-AUC: 0.1593 (primary metric — see decisions/0003)
Precision: 0.0934 Recall: 0.5901 F1: 0.1612
Brier score: 0.2019
Confusion matrix: {'tn': 68107, 'fp': 17432, 'fn': 1247, 'tp': 1795}

--- RandomForest -- VAL ---
ROC-AUC: 0.7842
PR-AUC: 0.2081 (primary metric — see decisions/0003)
Precision: 0.1425 Recall: 0.4921 F1: 0.2211
Brier score: 0.1323
Confusion matrix: {'tn': 76534, 'fp': 9005, 'fn': 1545, 'tp': 1497}

--- XGBoost -- VAL ---
ROC-AUC: 0.7961
PR-AUC: 0.2312 (primary metric — see decisions/0003)
Precision: 0.1193 Recall: 0.5759 F1: 0.1977
Brier score: 0.1357
Confusion matrix: {'tn': 72609, 'fp': 12930, 'fn': 1290, 'tp': 1752}

======================================================================
MODEL SELECTION (VAL PR-AUC) -- test set not touched yet
======================================================================
Model PR-AUC ROC-AUC Brier Fit(s)
LogisticRegression 0.1593 0.7463 0.2019 0.81
RandomForest 0.2081 0.7842 0.1323 64.49
XGBoost 0.2312 0.7961 0.1357 8.29

Highest VAL PR-AUC: XGBoost

--- XGBoost -- TEST (final, unbiased, touched once) ---
ROC-AUC: 0.7785
PR-AUC: 0.2189 (primary metric — see decisions/0003)
Precision: 0.1127 Recall: 0.5605 F1: 0.1876
Brier score: 0.1386
Confusion matrix: {'tn': 71887, 'fp': 13611, 'fn': 1355, 'tp': 1728}

This is your M3 'done when' evidence: justify the choice in one paragraph using these actual numbers -- PR-AUC gap over the next-best model, fit-time cost, and interpretability tradeoffs (LogReg coefficients vs. RF/XGBoost feature importances + SHAP support coming in M5). Write it into decisions/0004.
