# ADR V2-0010: P7 Calibration Method Selection

## Status

Accepted

## Context

Final LightGBM model (P5 winner, tuned params from P4/P5). Raw predicted
probabilities on val were compared against Platt scaling and isotonic
regression, both fit on val via 5-fold `cross_val_predict` (StratifiedKFold)
so the reported Brier scores are honest out-of-fold estimates, not the
calibrator scoring the same data it was fit on.

V1 (ADR 0005) picked Platt over isotonic because isotonic's near-identical
Brier came with a PR-AUC hit. V2 uses a simpler, explicit tie rule instead:
if Platt and isotonic are within 0.002 Brier of each other, prefer Platt
(same underlying reasoning — Platt is a smooth monotonic transform, less
prone to overfitting the calibration curve than isotonic's step function,
especially with a ~3.5% base rate where isotonic bins can get sparse).

## Brier Scores (val, out-of-fold)

| Method       | Brier      |
| ------------ | ---------- |
| Uncalibrated | 0.0177     |
| Platt        | 0.0181     |
| Isotonic     | **0.0174** |

## Decision

Selected **Platt scaling**.

Isotonic's Brier (0.0174) is lower than Platt's (0.0181), but the gap
(0.0007) is well inside the 0.002 tie threshold, so the tie-break rule
applies: Platt is preferred.

## Consequences

- The deployed pipeline (`models/lgbm_calibrated.pkl`) applies a
  `LogisticRegression` fit on `(raw_score) -> is_fraud` as the calibrator —
  a smooth, monotonic, two-parameter transform. Ranking (PR-AUC) is
  unchanged from the raw model by construction.
- Threshold optimization (cost sweep, review-rate cap) runs on Platt-
  calibrated probabilities, not raw LightGBM scores.
- If isotonic's calibration edge ever needs revisiting (e.g. a use case
  that's more calibration-sensitive than cost-threshold-sensitive), it's a
  one-line swap in `calibration_p7.py` — the isotonic calibrator is fit and
  evaluated in the same run, just not selected.

## Artifacts

- reports/calibration/reliability_diagram.png
- reports/calibration/threshold_cost_curve.png
- reports/calibration/p7_summary.md
- models/lgbm_calibrated.pkl
- models/model_metadata.json
