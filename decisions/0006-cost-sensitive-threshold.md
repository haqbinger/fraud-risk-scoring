# ADR 0006: Cost-sensitive thresholding under operational review constraints

- Status: Accepted
- Date: 2026-09-13

## Context

The calibrated model produces probabilities that are useful for expected-cost optimization. However, the threshold is not a property of the model alone; it is a property of the business cost model and operational capacity.

The project uses:

- COST_FALSE_NEGATIVE = 2000.0
- COST_FALSE_POSITIVE = 150.0
- MAX_REVIEW_RATE = 0.05

These values represent a realistic fraud-operation policy:

- missing a fraud event is very expensive due to chargeback, investigation, and reputational risk
- reviewing a legitimate transaction is still costly, but materially lower than a missed fraud
- manual-review capacity is capped at 5% of transactions

This policy matters because a raw cost minimization without a review cap would push the threshold toward the lower bound and effectively flag too much traffic.

## Decision

We select the cost-aware threshold by minimizing expected cost under the constraint that review rate does not exceed 5%.

## Result

Using the final cost assumptions and operational cap:

- optimal threshold under cap: 0.136
- review rate: 5.0%
- FN: 2043
- FP: 3349
- total cost: $4,588,350.00

Compared with the naive threshold of 0.5:

- naive threshold total cost: $6,166,000.00
- cost-aware threshold savings: $1,577,650.00

This is both operationally feasible and materially better than the naive decision rule.

## Why this is the right threshold

The model itself is not “bad” because it pushes the threshold lower under a different cost function. The threshold is a consequence of the commercial objective:

- if false negatives are valued much more highly than false positives, then the optimal threshold drops
- if review capacity is limited, the threshold must be constrained to stay operationally feasible

This is not a bug in the optimization pipeline; it is a policy decision encoded in the cost model.

## Consequences

- We do not use a fixed threshold like 0.5 as a default business rule.
- We choose thresholds based on expected cost and review capacity.
- The threshold can change meaningfully if the business rebalances false-positive vs false-negative cost assumptions.

## Sensitivity summary

From the generated sensitivity table:

| cost_fn | cost_fp | optimal_threshold | total_cost |   fn |   fp |
| ------- | ------: | ----------------: | ---------: | ---: | ---: |
| 50      |   150.0 |          0.375621 |   152000.0 | 2989 |   17 |
| 100     |   150.0 |          0.375621 |   301450.0 | 2989 |   17 |
| 250     |   150.0 |          0.340657 |   734700.0 | 2859 |  133 |
| 500     |   150.0 |          0.255742 |  1406950.0 | 2540 |  913 |
| 1000    |   150.0 |          0.155843 |  2542600.0 | 2137 | 2704 |
| 2000    |   150.0 |          0.080919 |  4452850.0 | 1706 | 6939 |

This is the core M4 finding: the selected threshold is not a model property; it is a cost-model property. That is exactly why the project documents both the cost assumptions and the review-rate constraint alongside the threshold selection.
