# ADR 0005: Calibration method selection

- Status: Accepted
- Date: 2026-09-13

## Context

The winning model from the model comparison step is XGBoost. The calibration pipeline compares three probability post-processing strategies:

- uncalibrated XGBoost probabilities
- Platt scaling
- isotonic regression

The objective was not to maximize raw ranking performance again, but to improve probability calibration while preserving ranking quality. This matters because cost-sensitive thresholding depends on probability values being approximately truthful.

The calibration script prints the key rule explicitly:

> PR-AUC/ROC-AUC should barely move across methods -- calibration reshapes probabilities, it doesn't change ranking.

This is expected behavior. Calibration is a monotonic transformation of the score scale. It changes the probability values but should not materially reorder the examples.

## Decision

We selected Platt scaling.

## Why

On the validation/test split used in the project:

- Uncalibrated: Brier = 0.1378, PR-AUC = 0.2205
- Platt: Brier = 0.0306, PR-AUC = 0.2205
- Isotonic: Brier = 0.0303, PR-AUC = 0.2123

This is the important comparison:

- Both Platt and isotonic materially improve calibration relative to the uncalibrated probabilities.
- Platt and isotonic have nearly identical Brier scores, but isotonic slightly degrades PR-AUC.
- The project explicitly preserves ranking quality as a guardrail before accepting a lower Brier score.
- Therefore, the calibration method selected is Platt, because it improves calibration without hurting ranking performance.

## Consequences

- We keep the ranking structure of the model unchanged.
- The probability outputs are materially more trustworthy for expected-cost optimization.
- Thresholds chosen downstream are based on calibrated probabilities rather than poorly calibrated raw XGBoost scores.

## Notes

This is a case where calibration quality and ranking quality are intentionally treated separately. A better-calibrated model is only acceptable if it does not materially degrade the ordering signal that the fraud model is using for prioritization.
