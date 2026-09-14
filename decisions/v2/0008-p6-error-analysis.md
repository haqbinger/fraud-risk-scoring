ADR V2-0008: P6 Error Analysis

Status

Closed — tested, no lift, skipped (see Patch Test Result below)

Context

LightGBM final model (P5), threshold 0.5 (unoptimized — P7 fixes this).
Evaluated on val set only. Test set untouched.

Headline Numbers

| Metric                          | Value                        |
| ------------------------------- | ---------------------------- |
| False Negatives (fraud missed)  | 1,393 (45.8% miss rate)      |
| False Positives (legit flagged) | 468 (0.55% false-alarm rate) |

Note: 45.8% FN rate at threshold 0.5 is expected — model is conservative at this threshold. Val PR-AUC 0.6741 measures quality across all thresholds. P7 threshold optimization will collapse FN rate significantly.

FN Blind Spots

| Finding                          | Detail                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------ |
| product_cd=W dominates by volume | 981/1393 FNs (70.4%), miss rate 66% (1.44x base — just under 1.5x flag)        |
| Temporal drift                   | FN rate 29% in March → 47% in April — consistent with P3 fold underperformance |

FP Blind Spots

| Bucket            | FP Rate vs Base             |
| ----------------- | --------------------------- |
| card4=discover    | 5.22x base false-alarm rate |
| product_cd=C      | 5.13x                       |
| comcast.net email | 4.14x                       |
| card6=credit      | 2.78x                       |

Feature Candidates

- FN fix: product_cd-conditioned expanding-window fraud rate (per-product_cd historical fraud rate, same leakage-safe pattern as device_hist_fraud_rate)
- FP fix: card4-conditioned expanding-window non-fraud rate (per-card-network historical legitimacy signal to reduce Discover false alarms)

Decision (original, at time of P6)

Add both feature candidates as a P6 patch before P7. Same leakage-safe expanding-window pattern already in features_m1.sql — replicate for product_cd and card4.

Patch Test Result (post-P7)

Tested after P7, not before it (P7 proceeded directly per explicit direction).
Both features were added to features_m1.sql using the exact device_hist_fraud_rate
expanding-window pattern (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`,
ordered by txn_ts, transaction_id):

- `product_cd_te_fraud_rate` / `product_cd_te_txn_count`
- `card4_hist_legit_rate` / `card4_hist_txn_count`

Kitchen-sink LightGBM retrained on the 182 P2-selected features + these 4 new
columns (186 total), using the existing tuned `lgbm_best_params.json` (no new
hyperparameter search), evaluated on val:

| | Val PR-AUC |
| --- | --- |
| Ceiling (182 features) | 0.6741 |
| Patch test (186 features) | 0.6673 |
| Delta | -0.0068 |

The patch made val PR-AUC worse, not just flat — below the 0.005 absorb
threshold in the wrong direction. Most likely explanation: `product_cd` and
`card4` are low-cardinality (5 and 4 distinct values respectively) and are
already present as raw categorical features in the selected set, so their
expanding-window fraud/legitimacy rates are highly collinear with signal the
model already had access to via `product_cd`/`card4` splits directly — the
new columns added noise (small-sample expanding-window rates early in the
train window) without adding new information.

Verdict: reverted. `sql/features_m1.sql` restored to its pre-patch state
via `git checkout`; `features_m1` rebuilt to match. The FN (product_cd=W
volume dominance) and FP (card4=discover blind spot) findings from P6 remain
valid observations about the current model's error modes — they just don't
translate into a useful feature this way. Worth revisiting in P8 error
analysis with a different formulation (e.g. amount-conditioned or
interaction features) if the same blind spots persist.

Artifacts

- reports/error_analysis/fn_analysis.csv
- reports/error_analysis/fp_analysis.csv
- reports/error_analysis/blind_spots.md
- reports/error_analysis/p6_summary.md
- reports/error_analysis/p6_patch_test_result.json
