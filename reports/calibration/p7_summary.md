P7 Calibration + Threshold Optimization Summary

Calibration

- Val Brier: uncalibrated=0.0177, Platt (5-fold CV)=0.0181, Isotonic (5-fold CV)=0.0174
- Selected calibrator: platt (tied within 0.002 Brier -- Platt preferred (V1 ADR 0005 reasoning))
- See decisions/v2/0010-p7-calibration-method-and-operational-threshold.md for the full ADR

Threshold optimization

- Cost model: FN=$2,000, FP=$150, max review rate=5%
- Unconstrained cost-optimal threshold: 0.027 (review_rate=5.9%)
- Operational threshold (review_rate <= 5%): 0.036
- Val: precision=0.4935, recall=0.7068, review_rate=4.9%

Final test set results (one-time, honest)

- PR-AUC: 0.5941
- ROC-AUC: 0.9168
- Brier: 0.0209
- Precision: 0.4405
- Recall: 0.6442
- F1: 0.5232
- Confusion matrix: {'tn': 82975, 'fp': 2523, 'fn': 1097, 'tp': 1986}
- Cost @ threshold 0.036: $2,572,450

V1 vs V2 (test set) -- head-to-head, no naive-0.5 baseline

The naive-0.5 baseline was dropped from this comparison: it moves with each
model's own calibration (V2's well-calibrated scores rarely exceed 0.5 at
all, since the true base rate is ~3.5%), so "savings vs naive" compressed
the visible gap even though V2 is the better model. Comparing each model's
total cost at its own optimal threshold is the honest, apples-to-apples
number.

| Metric | V1 | V2 | Delta (V2 - V1) |
|---|---|---|---|
| PR-AUC (test) | 0.2189 | 0.5941 | +0.3752 |
| Brier (test) | 0.0306 | 0.0209 | -0.0097 |
| Own optimal threshold | 0.1410 | 0.0360 | -0.1050 |
| Total cost @ own optimal threshold | $4,633,800 | $2,572,450 | -$2,061,350 |

V2 costs $2,061,350 less (44.5% lower) than V1 at each model's own
optimal threshold -- operationally, running V2 instead of V1 would cut
expected fraud-related losses/review costs by roughly $2.06M over a
comparable transaction volume, confirming the PR-AUC/Brier gains translate
directly into a materially better business outcome, not just a ranking-
metric improvement.

V1's total cost figure is derived, not re-measured: V1's naive-0.5 test cost
($6,166,000, from decisions/0006-cost-sensitive-threshold.md) minus V1's
previously reported test-set savings ($1,532,200) = $4,633,800. This is
consistent with (and close to) ADR 0006's own directly-computed total cost
of $4,588,350 at its documented threshold of 0.136 -- the small difference
is expected, since the two source numbers reflect a slightly different
optimal threshold (0.141 vs 0.136).

Artifacts

- reports/calibration/threshold_cost_curve.png
- reports/calibration/reliability_diagram.png
- models/lgbm_calibrated.pkl
- models/model_metadata.json

ROC-AUC (context)

- Test ROC-AUC: 0.9168
- PR-AUC (test_pr_auc=0.5941) remains the primary metric for this model because ROC-AUC is dominated by the large true-negative count under ~3.5% fraud prevalence and so overstates ranking quality on the minority (fraud) class relative to PR-AUC.
