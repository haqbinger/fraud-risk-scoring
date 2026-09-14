# ADR V2-0003: P3 Temporal Cross-Validation

## Status

Accepted

## Method

5-fold rolling temporal CV on 182-feature set (selected_features.txt).
Each fold's val window strictly after train window, split by transaction count.
XGBoost default hyperparams (same as P2 kitchen-sink for fair comparison).

## Results

| Fold | Train window  | Val window    | Val PR-AUC |
| ---- | ------------- | ------------- | ---------- |
| 1    | Dec 1–20      | Dec 20–Jan 10 | 0.5426     |
| 2    | Dec 20–Jan 10 | Jan 10–Feb 7  | 0.5979     |
| 3    | Jan 10–Feb 7  | Feb 7–Mar 5   | 0.5932     |
| 4    | Feb 7–Mar 5   | Mar 5–Apr 1   | 0.5467     |
| 5    | Mar 5–Apr 1   | Apr 1–May 1   | 0.5406     |

Mean PR-AUC: 0.5642 ± 0.0287 (min 0.5406, max 0.5979)

## Verdict

RELIABLE. P2 single-split val PR-AUC 0.5888 is within 0.0246 of CV mean (threshold 0.03). Std 0.0287 under 0.05 tolerance. Single-split was not a lucky draw.

## Key Finding — Seasonal
