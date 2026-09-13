# M4 Summary: Calibration, thresholding, and cost-model pivots

## What went wrong at first

We encountered a series of issues that were not all model-related:

- Repository/layout problems: the wrong script invocation and package assumptions
- sklearn compatibility issue: `CalibratedClassifierCV(..., cv="prefit")` failed in the installed version
- filesystem bug: the script tried to save a figure before the output directory existed
- missing artifact bug: the winner model was never persisted, so downstream calibration could not load it

These were real engineering problems, but they were not the core problem with the fraud model itself.

## What we fixed

We corrected the pipeline by:

- using validation-based Platt and isotonic calibration instead of `cv="prefit"`
- ensuring output folders exist before saving figures
- persisting the selected winner model to `data/processed/winning_model.json`
- enforcing review-cap and cost-aware threshold optimization

## Calibration finding

The key calibration result was that the PR-AUC stayed essentially unchanged across methods, and that is expected.

Why:

- calibration changes the probability scale, not the ranking order
- PR-AUC measures ranking quality
- a monotonic probability transform should not materially reorder predictions

This means the important comparison is:

- calibration quality (Brier score)
- ranking preservation (PR-AUC)

The final numbers were:

- Uncalibrated: Brier = 0.1378, PR-AUC = 0.2205
- Platt: Brier = 0.0306, PR-AUC = 0.2205
- Isotonic: Brier = 0.0303, PR-AUC = 0.2123

This showed that:

- calibration materially improved probability quality
- Platt was preferred because it preserved PR-AUC while achieving near-identical Brier performance to isotonic

## Cost-model pivot

The early thresholding results were mathematically consistent but operationally absurd. That was a sign that the cost model assumptions were unrealistic, not that the optimization logic was completely broken.

The original assumptions effectively implied:

- false positives were almost free
- reviewing many legitimate transactions was cheap
- the optimizer therefore pushed the threshold to the lower bound

This is why we moved to a more realistic fraud-operation setup:

- `COST_FALSE_NEGATIVE = 2000.0`
- `COST_FALSE_POSITIVE = 150.0`
- `MAX_REVIEW_RATE = 0.05`

These values reflect:

- a missed fraud is very expensive
- a false positive is still costly, but substantially less so
- a manual review policy cannot realistically exceed 5% of traffic

## Final decision

We selected the Platt-calibrated model and a cost-aware threshold under a 5% review cap.

Final operational results:

- selected calibration: Platt
- optimal threshold under cap: 0.136
- review rate: 5.0%
- FN: 2043
- FP: 3349
- total cost: $4,588,350.00

Compared with naive threshold 0.5:

- naive threshold total cost: $6,166,000.00
- cost-aware threshold savings: $1,577,650.00

## Why the plots matter

### Cost-sensitive threshold sweep

This plot shows expected total cost as a function of threshold. It demonstrates:

- the threshold is not a property of the model alone
- the threshold is a function of the cost model and review constraints
- a naive threshold like 0.5 is often not the business-optimal choice

### Reliability diagram

This plot shows whether the predicted probabilities match actual fraud rates.

The important story is:

- the uncalibrated model is visibly miscalibrated
- after Platt or isotonic calibration, the curve moves much closer to the diagonal
- this is the visual evidence that the probability outputs are much more trustworthy for downstream thresholding

## Final takeaway

This project changed from a “train a model” problem into a “calibrate probabilities and optimize under business constraints” problem.

That was the crucial pivot:

- engineering issues were fixed
- the cost model was corrected
- the review-rate constraint was enforced
- the final threshold is realistic, explainable, and operationally defensible

The result is a reliable production-style fraud-risk decision process, not just a technically valid model.
