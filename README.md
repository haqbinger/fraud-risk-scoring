Fraud Risk Scoring

End-to-end fraud detection pipeline on the IEEE-CIS transaction dataset (590,540 transactions, 3.5% fraud rate), covering feature engineering, feature selection, temporal cross-validation, hyperparameter tuning, model comparison, calibration, cost-sensitive thresholding, and SHAP explainability. V2 replaces the V1 baseline (20 of 394 available features, XGBoost) with a 182-feature LightGBM model, taking test PR-AUC from 0.2189 to 0.5941; test ROC-AUC is 0.9168, though PR-AUC remains the primary metric given the ~3.5% fraud prevalence. The calibrated model is served via FastAPI with live Postgres feature computation, per-request SHAP explanations, MLflow experiment tracking, and a prediction-log table for drift monitoring, deployed on Render.

V1 vs V2

| Metric                  | V1                              | V2         |
| ----------------------- | ------------------------------- | ---------- |
| Features used           | 20 / 394                        | 182 / 394  |
| Test PR-AUC             | 0.2189                          | 0.5941     |
| Test ROC-AUC            | n/a (V1 artifacts not retained) | 0.9168     |
| Test Brier score        | 0.0306                          | 0.0209     |
| Operational threshold   | 0.141                           | 0.036      |
| Total cost at threshold | $4,633,800                      | $2,572,450 |

Threshold and total cost are both under the same cost model (FN=$2,000, FP=$150, max review rate=5%). V1's total cost is derived from decisions/0006-cost-sensitive-threshold.md's naive-0.5 baseline minus V1's reported test-set savings; V2's is measured directly. V2's ROC-AUC is recomputed independently from the saved models/lgbm_calibrated.pkl pipeline in scripts/compute_roc_auc.py; V1's underlying model artifacts were not retained, so no comparable figure exists for it. Full derivation and the head-to-head reasoning (why this replaces a "savings vs naive 0.5" comparison) are in reports/calibration/p7_summary.md.

Architecture

Data layer — raw_transaction/raw_identity (Postgres) hold the unmodified IEEE-CIS CSVs. sql/features_m1.sql builds features_m1: leakage-safe window-function features (velocity, historical amount stats, expanding-window target encoding for card1/addr1/p_emaildomain, missingness flags per V-column block) joined against transactions. fraud.data.load_data() is the single read path every training script uses.

Model layer — LightGBM, tuned with Optuna against 5-fold rolling temporal cross-validation, selected over XGBoost after a head-to-head comparison on identical features and folds. Platt scaling on top of the raw model output, selected over isotonic regression by out-of-fold Brier score. Decision threshold set by minimizing expected cost under an operational review-rate cap, not a fixed 0.5 cutoff.

Serving layer — FastAPI (/predict, /health, /model), computing card1/device history features live against Postgres at request time using the same leakage-safe temporal gating as the offline SQL. Per-request SHAP explanations via TreeExplainer on the raw model output. Predictions logged to a prediction_logs table for drift monitoring. Runs tracked in MLflow (experiment fraud-risk-scoring). Containerized single-stage Docker build, deployed on Render.

V2 Roadmap

- P0 — EDA on raw_transaction: V-column missingness block structure, categorical cardinality audit, univariate signal ranking.
- P1 — Full feature engineering: rebuilt transactions (397 cols) and features_m1 (410 cols) with leakage-safe target/frequency encoding; load_data() at 415 columns.
- P2 — Feature selection: kitchen-sink XGBoost ceiling (0.5725 val PR-AUC, 411 features) reduced to 182 features via gain/permutation disagreement + correlation pruning (0.5691 val PR-AUC, within tolerance).
- P1 patch — Piped card2-6/addr2/dist1/dist2 into the feature matrix; ceiling moved to 0.5796, reselected to 182 features at 0.5888 val PR-AUC.
- P3 — Temporal CV: 5 rolling folds, mean PR-AUC 0.5642 ± 0.0287, confirming the single-split estimate was reliable.
- P4 — Optuna hyperparameter tuning (50 trials) against the P3 folds: CV mean PR-AUC 0.5642 → 0.6167.
- P5 — LightGBM vs. tuned XGBoost, same features and folds: LightGBM wins (CV mean 0.6316 vs. 0.6167), selected as the final model.
- P6 — Error analysis on FN/FP blind spots (product_cd=W volume dominance, card4=discover false-alarm rate); dedicated features tested and reverted (val PR-AUC delta -0.0068).
- P7 — Platt calibration selected by out-of-fold Brier; cost-sensitive threshold (0.036) under a 5% review-rate cap; final test PR-AUC 0.5941.
- P8 — SHAP: card1_te_fraud_rate dominates global importance (2x the runner-up); false positives traced to target-encoding "guilt by association" on cards with a prior confirmed fraud.

Key engineering decisions

- Temporal split discipline — chronological train/val/test split, never random, because card-level velocity/history features leak future information into the past under random splitting; the measured gap is 0.0283 PR-AUC (decisions/0001-temporal-split-not-random.md).
- Leakage-safe target encoding — every expanding-window fraud-rate feature (card1, addr1, p_emaildomain) uses ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING, ordered by txn_ts, transaction_id, replicated exactly from the one pattern already leakage-tested in V1 (decisions/v2/0002-p1-feature-engineering.md).
- Permutation over gain importance — 297 of 411 features disagreed by more than 50 rank positions between gain and permutation importance; 81 were dropped for negative permutation importance despite high gain, since gain reflects training-set memorization and permutation reflects held-out generalization (decisions/v2/0003-p2-feature-selection.md).
- Cost-model threshold, not 0.5 — the operating threshold is the minimizer of expected cost (FN=$2,000, FP=$150) subject to a 5% manual-review-rate cap, not a default classification cutoff (decisions/0006-cost-sensitive-threshold.md, decisions/v2/0010-p7-calibration-method-and-operational-threshold.md).
- Absolute cost over naive-baseline savings — "savings vs. naive 0.5" was dropped as a V1/V2 comparison because the naive baseline itself moves with each model's calibration quality; comparing total cost at each model's own optimal threshold is the number that survives scrutiny (reports/calibration/p7_summary.md).

Stack

Python 3.11/3.12, PostgreSQL 16, LightGBM, XGBoost, FastAPI, MLflow, SHAP, Optuna, Docker, Render.

Running locally
docker compose up -d
.\venv\Scripts\Activate.ps1
python -c "from fraud.config import DB_URL; print(DB_URL)"
python -c "from fraud.data import load_data; df = load_data(); print(df.shape)"
