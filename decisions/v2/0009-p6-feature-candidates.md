# ADR V2-0009: P6 Feature Candidates — Feed-Back Into P1

## Status

Accepted

## Context

P6 error analysis produced two concrete feature candidates worth adding before P7 calibration and threshold optimization.

## Candidates

### 1. product_cd fraud rate (FN fix)

- Pattern: product_cd=W accounts for 70.4% of all FNs by volume
- Feature: expanding-window fraud rate partitioned by product_cd, ordered by txn_ts, transaction_id
- Same ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING pattern as device_hist_fraud_rate
- Expected impact: direct signal for dominant FN segment

### 2. card4 legitimacy rate (FP fix)

- Pattern: card4=discover false-alarm rate is 5.22x base
- Feature: expanding-window non-fraud rate partitioned by card4
- Gives model explicit signal that Discover transactions are overwhelmingly legitimate historically
- Expected impact: reduce FP rate on uncommon-but-legitimate card networks

## Implementation

Add both to sql/features_m1.sql. Rebuild features_m1 view. Rerun load_data() — f.\* absorbs automatically. Rerun P2 feature selection to confirm both survive into selected set. If they do, retrain LightGBM and check val PR-AUC delta before proceeding to P7.

## Risk

Low — both follow identical leakage-safe pattern already proven in production. No new SQL primitives needed.
