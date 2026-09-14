P6 Error Analysis Summary

- Model: LightGBM (P5 winner), params from reports/model_comparison/lgbm_best_params.json
- Threshold: 0.5 (unoptimized -- P7 will tune this)
- Val set: 88581 rows, 3042 fraud, 85539 legit
- FN: 1393 (miss rate 0.4579)
- FP: 468 (false-alarm rate 0.0055)

FN rate by month

| month | population | FN | FN rate |
|---|---|---|---|
| 2018-03 | 183 | 53 | 0.2896 |
| 2018-04 | 2828 | 1330 | 0.4703 |
| 2018-05 | 31 | 10 | 0.3226 |

Feature candidates

- FN -- found: No bucket cleared the 1.5x flag (overall miss rate is already high at 0.46), but product_cd=W accounts for 981/1393 (70.4%) of all FNs at a 0.66 miss rate (1.44x base) -- the dominant FN segment by volume even though its rate ratio falls just under the flag line. Closest rate-based near-miss: p_emaildomain=icloud.com (1.49x).; did: Add a product_cd-conditioned fraud-rate/velocity feature (e.g. expanding-window fraud rate partitioned by product_cd) so the model has a direct signal for this segment instead of relying on it correlating with existing features.
- FP -- found: Top FP blind spot: card4=discover (5.22x base false-alarm rate); did: Add a card4-conditioned historical legitimacy feature (e.g. per-card4 expanding-window non-fraud rate or txn count) so common-but-unusual-looking legitimate patterns in this segment stop triggering false alarms.

Feed back into P1

If the blind spots above point at a categorical or velocity feature not yet in features_m1 (e.g. a per-product_cd or per-card4 expanding-window fraud rate), it belongs in P1's target/frequency encoding pass, then P2-P5 should be re-run on the expanded feature set to confirm it survives selection and moves CV PR-AUC.
