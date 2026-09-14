# Fraud Risk Scoring — Project Handoff Document

# V1 COMPLETE — Starting V2 in fresh session

---

## V1 STATUS

Tag: v1.0-baseline
GitHub: https://github.com/haqbinger/fraud-risk-scoring
Render: https://fraud-risk-scoring-api-b12m.onrender.com
Completed: 2026-09-13
M0-M13 complete. Model PR-AUC: 0.22 (20/394 features — V2 fixes this).

---

## V2 PRIORITY — DO THIS FIRST, BEFORE ANYTHING ELSE

The model uses 20 features out of 394 available. V/C/D/M columns sit
in raw_transaction in Postgres, unused. Expected PR-AUC after full
feature engineering: 0.55-0.70. Follow Roadmap V2 (P0-P8) below.

---

## Environment

- Windows 11, PowerShell
- Python 3.11 locally, Python 3.12 in Docker (xgboost 3.4.1 requires >=3.12)
- venv at project root: .\venv\Scripts\Activate.ps1
- Postgres 16 in Docker, port 5433, container: fraud_risk_pg
  Connection: postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db
- Docker Compose project name: fraud-risk-scoring
- pip install -e . already run — fraud package importable
- calibration importable as top-level package (sits at src/calibration/)

Resume session:
docker compose up -d
.\venv\Scripts\Activate.ps1
python -c "from fraud.config import DB_URL; print(DB_URL)"
python -c "from fraud.data import load_data; df = load_data(); print(df.shape)"

---

## Actual file structure (verified 2026-09-13)

fraud-risk-scoring/
├── .env.example
├── .gitignore
├── Dockerfile # python:3.12-slim, COPY src/ sql/
├── docker-compose.yml # postgres (5433) + api (8000) services
├── pyproject.toml # src layout, finds fraud + calibration
├── render.yaml # Render deployment config
├── requirements.txt # full dev deps (pinned versions below)
├── requirements-serve.txt # slim serving deps (no mlflow/matplotlib)
├── leakage_test.py # standalone CLI leakage test (no package needed)
├── mlflow.db # local MLflow tracking store
│
├── data/
│ └── raw/ # train_transaction.csv etc (gitignored)
│
├── decisions/ # ADRs 0001-0010
│ ├── 0001-temporal-split-not-random.md
│ ├── 0002-card1-device-proxy-and-null-coldstart.md
│ ├── 0003-pr-auc-primary-metric.md
│ ├── 0004-final-model-selection.md
│ ├── 0005-calibration-method.md
│ ├── 0006-cost-sensitive-threshold.md
│ ├── 0007-explainability-policy.md
│ ├── 0008-serving-architecture.md
│ ├── 0009-prediction-logging-and-test-architecture.md
│ └── 0010-containerization-architecture.md
│
├── docs/
│ ├── PROJECT_HANDOFF.md # this file
│ ├── m4_summary.md
│ ├── m5_explainability.md
│ ├── m7_summary.md
│ ├── m8_summary.md
│ ├── m9_summary.md
│ └── m10_render_deployment.md
│
├── models/ # empty, gitkeep only
│
├── reports/
│ └── figures/ # calibration_curves.png, threshold_cost_curve.png
│
├── REVIEW_QUESTIONS/
│ └── layer_1_review.md
│
├── scripts/
│ ├── load_data.py # CSV -> Postgres via COPY (not to_sql)
│ ├── train_baseline.py # M2: LR + RF baselines
│ ├── random_vs_temporal.py # M2: chronological vs random gap experiment
│ ├── model_comparison.py # M3: LR/RF/XGBoost comparison
│ └── save_model_artifacts.py # M7: saves platt_calibrator + model_metadata
│
├── sql/
│ ├── features_m1.sql # M1: leakage-safe window function features
│ └── prediction_logs.sql # M8: prediction audit log DDL
│
├── src/
│ ├── calibration/ # NOT inside fraud package — import as calibration.X
│ │ ├── calibrate.py # load_winning_model(), Platt/isotonic comparison
│ │ ├── cost_model.py # expected_cost(), sweep_thresholds(), sensitivity
│ │ └── threshold_optimization.py
│ │
│ ├── fraud/ # main package (pip install -e .)
│ │ ├── **init**.py
│ │ ├── config.py # DB_URL, RANDOM_STATE, costs, MAX_REVIEW_RATE
│ │ ├── data.py # load_data() — features_m1 JOIN transactions
│ │ ├── features.py # build_feature_matrix(), fit_median_impute()
│ │ ├── api/
│ │ │ ├── **init**.py
│ │ │ ├── main.py # FastAPI: /predict /health /model
│ │ │ ├── schemas.py # Pydantic v2 request/response models
│ │ │ └── serving.py # build_live_feature_vector() — live Postgres queries
│ │ ├── evaluations/
│ │ │ ├── splits.py # temporal_split(), random_split()
│ │ │ └── metrics.py # compute_metrics(), precision_at_recall()
│ │ ├── explainability/
│ │ │ ├── **init**.py
│ │ │ └── explain.py # SHAP global + local (TP/FP/borderline)
│ │ └── tracking/
│ │ ├── **init**.py
│ │ └── track_run.py # MLflow logging
│ │
│ └── reports/
│ └── figures/ # SHAP plots (note: in src/, not reports/)
│
└── tests/
├── conftest.py # engine fixture (skip if DB down), fake_model_state
├── test_api.py # 14 tests, fully hermetic (no DB/model needed)
├── test_leakage.py # 5 tests, skips if DB down
├── test_model_output.py # 8 tests, skips if artifacts missing
├── test_schema.py # 6 tests, skips if DB down
├── first_baseline_results.md
└── M3_model_comp_results.md

---

## Requirements (pinned, verified working)

requirements.txt (full dev):
pandas==3.0.5, sqlalchemy==2.0.52, psycopg2-binary==2.9.13,
scikit-learn==1.9.1, xgboost==3.4.1, matplotlib==3.11.2,
shap==0.52.0, mlflow==3.16.0, joblib==1.6.0, fastapi==0.141.1,
uvicorn[standard]==0.52.4, httpx==0.28.1, boto3==1.43.93,
pytest==7.4.2

requirements-serve.txt (Docker/Render, no mlflow/matplotlib/boto3):
pandas, sqlalchemy, psycopg2-binary, scikit-learn, xgboost,
joblib, fastapi, uvicorn[standard], httpx

---

## V1 results (real numbers, verified on 590,540 rows)

Fraud rate: 3.499%
Temporal split: train 413,378 / val 88,581 / test 88,581

M2 baselines (test PR-AUC):
LogisticRegression: 0.1086
RandomForest: 0.2078
Random vs chronological gap: 0.0283 PR-AUC (~17.8% relative overestimation)

M3 model selection (val PR-AUC → selection, test PR-AUC → honest number):
LogisticRegression: val 0.1593, test 0.1086
RandomForest: val 0.2081, test 0.2078
XGBoost: val 0.2312, test 0.2189 ← selected
XGBoost fit time: 8.29s vs RandomForest 64.49s (~7.8x faster)

M4 calibration:
Uncalibrated Brier: 0.1378 → Platt Brier: 0.0306 (4.5x improvement)
Operational threshold: 0.141 (under 5% review cap)
Cost savings vs naive 0.5: $1,532,200 on test set
COST_FALSE_NEGATIVE=2000, COST_FALSE_POSITIVE=150, MAX_REVIEW_RATE=0.05

M5 SHAP: global summary + TP/FP/borderline waterfall plots
M6 MLflow: experiment "fraud-risk-scoring", model "fraud-xgboost" v1
M7 FastAPI: /predict /health /model, live Postgres feature computation
M8 Tests: 33 passing (14 hermetic API, 8 model output, 6 schema, 5 leakage)
M9 Docker: docker compose up --build → fully working system
M10 Render: /health and /model live at render URL above

---

## Key architectural decisions (ADRs 0001-0010)

ADR 0001: Temporal split. Gap: 0.0283 PR-AUC.
ADR 0002: card1=customer proxy, device_info=device proxy. NULL cold-start.
ADR 0003: PR-AUC primary metric (not ROC-AUC, not accuracy).
ADR 0004: XGBoost selected. SUPERSEDED in V2 at P5.
ADR 0005: Platt calibration over isotonic (PR-AUC preservation).
ADR 0006: Threshold=0.141 under 5% review cap. SUPERSEDED in V2 at P7.
ADR 0007: SHAP on raw model output, not calibrated probabilities.
ADR 0008: Live Postgres feature computation per request (serving).
ADR 0009: Prediction logging DDL in sql/, skip-not-fail for DB tests.
ADR 0010: Single-stage Docker build, pinned requirements, sql/ in image.

---

## Known tech debt (fix in V2)

1. src/reports/figures/ vs reports/figures/ — SHAP plots in wrong location
2. calibration/ not inside fraud package — inconsistent with rest of src/fraud/
3. transactions table has ~12 columns — V/C/D/M columns unused (THE main issue)
4. model.get_params() round-trip issue — XGBoost hyperparams hardcoded in track_run.py
5. mlflow.db in project root — should be in .gitignore
6. Render /predict returns 500 — Render Postgres has no transaction data
7. Messy M10 git history (temp commits, force-adds) — cosmetic only

---

## V2 Roadmap (P0-P8) — model first, infrastructure last

FIRST: verify transactions table column count:
docker exec fraud_risk_pg psql -U fraud -d fraud_db -c \
 "SELECT COUNT(\*) FROM information_schema.columns WHERE table_name='transactions';"
Expected: ~12. This confirms V-columns are missing and P1 is needed.

P0 — Full EDA on raw_transaction
Missingness map (V-columns missing in correlated BLOCKS by payment processor)
Cardinality audit on all categoricals
Univariate signal: fraud vs non-fraud distribution per column
Deliverable: written list of informative/redundant/structured-missing columns
Script: standalone, no fraud package needed, output to reports/eda/

P1 — Full feature engineering
Rebuild transactions table with ALL columns from raw_transaction
Pass V/C/D/M columns through features_m1.sql
High-cardinality categoricals: frequency encoding first; target encoding
ONLY with expanding-window leakage discipline (same as device_hist_fraud_rate)
Missingness: indicator flag per V-column BLOCK (not per column)
Deliverable: unified feature matrix ~200-400 features, all leakage risks documented

P2 — Feature selection
Kitchen sink XGBoost as ceiling check
Rank by gain AND permutation importance (they disagree — disagreement is informative)
Drop near-zero/redundant; confirm PR-AUC holds on val
Deliverable: smaller clean feature set, explicitly justified

P3 — Temporal cross-validation
Rolling/expanding window CV (4-5 folds, val strictly after train)
Deliverable: mean +/- std PR-AUC across folds (not a single point estimate)

P4 — Hyperparameter tuning
Optuna or manual random search, optimized against P3 CV
Deliverable: tuned config vs M3 config with explicit PR-AUC delta

P5 — Model comparison at full scale
Re-run LR/RF/XGBoost + add LightGBM
Rewrite ADR 0004 with current-generation numbers

P6 — Error analysis
Pull actual FNs and FPs, find patterns by amount/time/merchant category
Feed back into P1 as new features or blind-spot flags
Deliverable: at least one concrete "found X, did Y about it"

P7 — Calibration + threshold on final model
Re-run src/calibration/ logic on P5 model
Rewrite ADRs 0005/0006 with real numbers

P8 — SHAP on final model
Re-run src/fraud/explainability/explain.py against new model/feature set

Then infrastructure (M7-M13 already built, just re-run against better model):
save_model_artifacts.py → retrain → redeploy to Render

---

## What to tell Claude at the start of V2 session

Paste this entire document, then say:

"We are starting V2 of this fraud detection project. V1 (M0-M13) is
complete and tagged as v1.0-baseline on GitHub. The model PR-AUC is
0.22 because we only used 20 of 394 available features. We are now
following Roadmap V2 starting at P0.

First task: write a standalone EDA script (no fraud package imports,
just sqlalchemy + pandas + matplotlib) that runs against raw_transaction
in Postgres and produces: (1) a missingness heatmap for V1-V339 columns
showing which columns are missing together (block structure), (2) a
cardinality table for all categorical columns, (3) a top-20 feature
signal chart comparing fraud vs non-fraud distributions. Script saves
outputs to reports/eda/ and uses SQL aggregation before pulling into
pandas where possible to avoid loading 590k x 400 columns into memory."
