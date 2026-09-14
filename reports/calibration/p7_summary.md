# P7 Calibration + Threshold Optimization Summary

## Calibration

- Val Brier: uncalibrated=0.0177, Platt (5-fold CV)=0.0181, Isotonic (5-fold CV)=0.0174
- Selected calibrator: **platt** (tied within 0.002 Brier -- Platt preferred (V1 ADR 0005 reasoning))
- See decisions/v2/0010-p7-calibration-method.md for the full ADR

## Threshold optimization

- Cost model: FN=$2,000, FP=$150, max review rate=5%
- Unconstrained cost-optimal threshold: 0.027 (review_rate=5.9%)
- Operational threshold (review_rate <= 5%): **0.036**
- Val: precision=0.4935, recall=0.7068, review_rate=4.9%
- Val cost savings vs naive 0.5: $936,450

## Final test set results (one-time, honest)

- PR-AUC: 0.5941
- ROC-AUC: 0.9168
- Brier: 0.0209
- Precision: 0.4405
- Recall: 0.6442
- F1: 0.5232
- Confusion matrix: {'tn': 82975, 'fp': 2523, 'fn': 1097, 'tp': 1986}
- Cost @ threshold 0.036: $2,572,450
- Cost @ naive 0.5: $3,406,300
- Cost savings vs naive 0.5: $833,850

## V1 vs V2 (test set)

| Metric | V1 | V2 | Delta |
|---|---|---|---|
| PR-AUC (test) | 0.2189 | 0.5941 | +0.3752 |
| Brier (test) | 0.0306 | 0.0209 | -0.0097 |
| Threshold | 0.1410 | 0.0360 | -0.1050 |
| Cost savings vs naive 0.5 | $1,532,200 | $833,850 | $-698,350 |

## Artifacts

- reports/calibration/threshold_cost_curve.png
- reports/calibration/reliability_diagram.png
- models/lgbm_calibrated.pkl
- models/model_metadata.json
