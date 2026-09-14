ADR V2-0002: P1 Full Feature Engineering

Status

Accepted

Context

transactions had 12 curated columns. raw_transaction has 394. features_m1.sql produced ~20 engineered features. Goal: expand to full feature matrix without introducing leakage.

Decisions

1. transactions table rebuild

Rejected literal SELECT \* FROM raw_transaction — would have replaced snake_case columns (transaction_id, is_fraud, txn_ts, device_type/device_info) with original camelCase, breaking features_m1.sql, leakage_test.py, and all training scripts. Instead: kept existing curated/renamed columns, dynamically passed through all remaining raw_transaction columns not already covered. Added --rebuild-transactions-only flag to load_data.py. Result: 397 columns.

2. features_m1.sql extension (410 output columns)

Generated programmatically, committed as static SQL.

Missingness flags: 9 flags, one per V-col block (V1-11, V12-34, V35-52, V53-94, V95-137, V138-166, V167-278, V279-321, V322-339) — derived from actual null-rate structure, not one flag per column (would add 339 noisy indicators).

Frequency encoding: P_emaildomain and R_emaildomain only — the two categoricals with meaningful cardinality (59/60 values). All others (≤5 values) left as-is; will be one-hot encoded at model training time.

Target encoding: card1, addr1, p_emaildomain — expanding-window fraud rate using ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING ordered by txn_ts, transaction_id. Replicates existing device_hist_fraud_rate pattern exactly.

Raw pass-through: All V/C/D/M columns passed unchanged.

3. load_data() update

Replaced hardcoded column list with f.\* plus the few raw fields not in features_m1 (transaction_amt, product_cd, addr1, p_emaildomain, device_type). Result: 415 columns.

Verification

- features_m1: 590,540 rows / 410 columns
- leakage_test.py: all checks passed (velocity, amount z-score, device fraud rate matched independent pandas recomputation)
- load_data(): 415 columns, correct fraud rate confirmed

Expected impact

V1 PR-AUC: 0.22 (20 features). Expected V2 PR-AUC after full pipeline: 0.55-0.70.
