# ADR V2-0008: P6 Error Analysis

## Status

Accepted

## Context

LightGBM final model (P5), threshold 0.5 (unoptimized — P7 fixes this).
Evaluated on val set only. Test set untouched.

## Headline Numbers

| Metric                          | Value                        |
| ------------------------------- | ---------------------------- |
| False Negatives (fraud missed)  | 1,393 (45.8% miss rate)      |
| False Positives (legit flagged) | 468 (0.55% false-alarm rate) |

Note: 45.8% FN rate at threshold 0.5 is expected — model is conservative at this threshold. Val PR-AUC 0.6741 measures quality across all thresholds. P7 threshold optimization will collapse FN rate significantly.

## FN Blind Spots

| Finding                          | Detail                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------ |
| product_cd=W dominates by volume | 981/1393 FNs (70.4%), miss rate 66% (1.44x base — just under 1.5x flag)        |
| Temporal drift                   | FN rate 29% in March → 47% in April — consistent with P3 fold underperformance |

## FP Blind Spots

| Bucket            | FP Rate vs Base             |
| ----------------- | --------------------------- |
| card4=discover    | 5.22x base false-alarm rate |
| product_cd=C      | 5.13x                       |
| comcast.net email | 4.14x                       |
| card6=credit      | 2.78x                       |

## Feature Candidates

- FN fix: product_cd-conditioned expanding-window fraud rate (per-product_cd historical fraud rate, same leakage-safe pattern as device_hist_fraud_rate)
- FP fix: card4-conditioned expanding-window non-fraud rate (per-card-network historical legitimacy signal to reduce Discover false alarms)

## Decision

Add both feature candidates as a P6 patch before P7. Same leakage-safe expanding-window pattern already in features_m1.sql — replicate for product_cd and card4.

## Artifacts

- reports/error_analysis/fn_analysis.csv
- reports/error_analysis/fp_analysis.csv
- reports/error_analysis/blind_spots.md
- reports/error_analysis/p6_summary.md
