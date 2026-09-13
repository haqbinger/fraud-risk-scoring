1st result without from sklearn.preprocessing import StandardScalar of Train_baseline.py:
Loading joined features + raw columns from Postgres ...
Loaded 590540 rows, fraud rate = 3.4990%
Temporal split — train: 413378 (2017-12-01 00:00:00 to 2018-03-30 19:26:36) | val: 88581 | test: 88581
C:\Users\yashv\OneDrive\Desktop\fraud-risk-scoring\venv\Lib\site-packages\sklearn\linear_model_logistic.py:599: ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = \_check_optimize_result(

--- LogisticRegression — VAL (model selection) ---
ROC-AUC: 0.6993
PR-AUC: 0.0713 (primary metric — see decisions/0003)
Precision: 0.0696 Recall: 0.6489 F1: 0.1258
Brier score: 0.2373
Confusion matrix: {'tn': 59161, 'fp': 26378, 'fn': 1068, 'tp': 1974}

--- LogisticRegression — TEST (final, unbiased) ---
ROC-AUC: 0.6901
PR-AUC: 0.0641 (primary metric — see decisions/0003)
Precision: 0.0653 Recall: 0.6831 F1: 0.1193
Brier score: 0.2586
Confusion matrix: {'tn': 55372, 'fp': 30126, 'fn': 977, 'tp': 2106}
Precision @ 80% recall (test): 0.0493

--- RandomForest — VAL (model selection) ---
ROC-AUC: 0.7871
PR-AUC: 0.2158 (primary metric — see decisions/0003)
Precision: 0.1113 Recall: 0.5773 F1: 0.1866
Brier score: 0.1566
Confusion matrix: {'tn': 71512, 'fp': 14027, 'fn': 1286, 'tp': 1756}

--- RandomForest — TEST (final, unbiased) ---
ROC-AUC: 0.7804
PR-AUC: 0.2106 (primary metric — see decisions/0003)
Precision: 0.1090 Recall: 0.5897 F1: 0.1840
Brier score: 0.1607
Confusion matrix: {'tn': 70635, 'fp': 14863, 'fn': 1265, 'tp': 1818}
Precision @ 80% recall (test): 0.0620

============================================================
SUMMARY — model selection uses VAL PR-AUC; TEST is reported, never used to pick
============================================================
LogisticRegression VAL PR-AUC: 0.0713 TEST PR-AUC: 0.0641
RandomForest VAL PR-AUC: 0.2158 TEST PR-AUC: 0.2106

2nd test with the Standarlearn :

Loaded 590540 rows, fraud rate = 3.4990%
Temporal split — train: 413378 (2017-12-01 00:00:00 to 2018-03-30 19:26:36) | val: 88581 | test: 88581

--- LogisticRegression — VAL (model selection) ---
ROC-AUC: 0.7463
PR-AUC: 0.1593 (primary metric — see decisions/0003)
Precision: 0.0934 Recall: 0.5901 F1: 0.1612
Brier score: 0.2019
Confusion matrix: {'tn': 68107, 'fp': 17432, 'fn': 1247, 'tp': 1795}

--- LogisticRegression — TEST (final, unbiased) ---
ROC-AUC: 0.7452
PR-AUC: 0.1086 (primary metric — see decisions/0003)
Precision: 0.0867 Recall: 0.6361 F1: 0.1526
Brier score: 0.2247
Confusion matrix: {'tn': 64847, 'fp': 20651, 'fn': 1122, 'tp': 1961}
Precision @ 80% recall (test): 0.0578

--- RandomForest — VAL (model selection) ---
ROC-AUC: 0.7878
PR-AUC: 0.2113 (primary metric — see decisions/0003)
Precision: 0.1129 Recall: 0.5822 F1: 0.1892
Brier score: 0.1567
Confusion matrix: {'tn': 71630, 'fp': 13909, 'fn': 1271, 'tp': 1771}

--- RandomForest — TEST (final, unbiased) ---
ROC-AUC: 0.7811
PR-AUC: 0.2078 (primary metric — see decisions/0003)
Precision: 0.1090 Recall: 0.5851 F1: 0.1838
Brier score: 0.1604
Confusion matrix: {'tn': 70754, 'fp': 14744, 'fn': 1279, 'tp': 1804}
Precision @ 80% recall (test): 0.0629

============================================================
SUMMARY — model selection uses VAL PR-AUC; TEST is reported, never used to pick
============================================================
LogisticRegression VAL PR-AUC: 0.1593 TEST PR-AUC: 0.1086
RandomForest VAL PR-AUC: 0.2113 TEST PR-AUC: 0.2078

RESULTS FROM RANDOM_VS TEMPORAL::

1ST RESULT :

Loading joined features + raw columns from Postgres ...
Loaded 590540 rows, fraud rate = 3.4990%

### RANDOM SPLIT (the naive/standard-practice approach)

Random split — train: 413378 | val: 88580 | test: 88582
C:\Users\yashv\OneDrive\Desktop\fraud-risk-scoring\venv\Lib\site-packages\sklearn\linear_model_logistic.py:599: ConvergenceWarning: lbfgs failed to converge after 5 iteration(s) (status=2):
ABNORMAL:

You might also want to scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = \_check_optimize_result(

--- Random split ---
ROC-AUC: 0.5594
PR-AUC: 0.0461 (primary metric — see decisions/0003)
Precision: 0.0482 Recall: 0.2072 F1: 0.0782
Brier score: 0.2497
Confusion matrix: {'tn': 72812, 'fp': 12669, 'fn': 2457, 'tp': 642}

### TEMPORAL SPLIT (the correct approach for this data)

Temporal split — train: 413378 (2017-12-01 00:00:00 to 2018-03-30 19:26:36) | val: 88581 | test: 88581
C:\Users\yashv\OneDrive\Desktop\fraud-risk-scoring\venv\Lib\site-packages\sklearn\linear_model_logistic.py:599: ConvergenceWarning: lbfgs failed to converge after 90 iteration(s) (status=2):
ABNORMAL:

You might also want to scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = \_check_optimize_result(

--- Temporal split ---
ROC-AUC: 0.5413
PR-AUC: 0.0507 (primary metric — see decisions/0003)
Precision: 0.0334 Recall: 0.5457 F1: 0.0630
Brier score: 0.2466
Confusion matrix: {'tn': 37513, 'fp': 48026, 'fn': 1382, 'tp': 1660}

============================================================
THE HEADLINE NUMBER
============================================================
Random split PR-AUC: 0.0461
Temporal split PR-AUC: 0.0507
Gap: -0.0046 PR-AUC points

If gap > 0: random splitting made the model look better than it would actually perform in production. Copy this number into decisions/0001-temporal-split-not-random.md's Consequencessection.

AGAIN THIS WAS WITHOUT STANDARDLEARN SPLIT STRATEGY

NOW WITH IT :

Loading joined features + raw columns from Postgres ...
Loaded 590540 rows, fraud rate = 3.4990%

### RANDOM SPLIT (the naive/standard-practice approach)

Random split — train: 413378 | val: 88580 | test: 88582

--- Random split ---
ROC-AUC: 0.7487
PR-AUC: 0.1876 (primary metric — see decisions/0003)
Precision: 0.0966 Recall: 0.5979 F1: 0.1663
Brier score: 0.1897
Confusion matrix: {'tn': 68142, 'fp': 17339, 'fn': 1246, 'tp': 1853}

### TEMPORAL SPLIT (the correct approach for this data)

Temporal split — train: 413378 (2017-12-01 00:00:00 to 2018-03-30 19:26:36) | val: 88581 | test: 88581

--- Temporal split ---
ROC-AUC: 0.7463
PR-AUC: 0.1593 (primary metric — see decisions/0003)
Precision: 0.0934 Recall: 0.5901 F1: 0.1612
Brier score: 0.2019
Confusion matrix: {'tn': 68107, 'fp': 17432, 'fn': 1247, 'tp': 1795}

============================================================
THE HEADLINE NUMBER
============================================================
Random split PR-AUC: 0.1876
Temporal split PR-AUC: 0.1593
Gap: 0.0283 PR-AUC points

If gap > 0: random splitting made the model look better than it would actually perform in production. Copy this number into decisions/0001-temporal-split-not-random.md's Consequencessection.
