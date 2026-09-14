# P8 SHAP Explainability Summary

- Model: LightGBM (P5/P7 final), raw (uncalibrated) output used for SHAP per V1 ADR 0007
- Explainer: TreeExplainer, 3012 rows (3000 random + 12 local examples)
- Operational threshold: 0.036

## Top 10 features by mean |SHAP|

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `card1_te_fraud_rate` | 0.84599 | Expanding-window historical fraud rate for card1 -- direct target-encoded risk signal. |
| 2 | `C13` | 0.39419 | Anonymized count feature (e.g. addresses/emails associated with this card). |
| 3 | `addr1_te_txn_count` | 0.33679 | Historical transaction count for this entity -- a maturity/volume proxy (new vs. established). |
| 4 | `transaction_amt` | 0.30072 | Raw transaction amount. |
| 5 | `p_emaildomain_te_txn_count` | 0.28672 | Historical transaction count for this entity -- a maturity/volume proxy (new vs. established). |
| 6 | `p_emaildomain` | 0.26411 | Raw categorical field (card network/type, product, email domain, or device). |
| 7 | `D2` | 0.25970 | Anonymized time-delta feature (days since some reference event). |
| 8 | `card1_hist_txn_count` | 0.21178 | Historical transaction count for this entity -- a maturity/volume proxy (new vs. established). |
| 9 | `dist1` | 0.20851 | Raw numeric identifier/distance field from the payment processor. |
| 10 | `C5` | 0.20282 | Anonymized count feature (e.g. addresses/emails associated with this card). |

## Local explanation patterns

### True positives

- Example 1: val_row=1710, calibrated_prob=0.9562, actual=1 -- top drivers: card1_te_fraud_rate (+2.2149), device_hist_fraud_rate (+2.0563), C1 (+1.9946)
- Example 2: val_row=14088, calibrated_prob=0.9562, actual=1 -- top drivers: device_hist_fraud_rate (+2.6967), C1 (+1.6965), card1_te_fraud_rate (+1.3528)
- Example 3: val_row=1715, calibrated_prob=0.9562, actual=1 -- top drivers: device_hist_fraud_rate (+2.0430), C1 (+2.0120), card1_te_fraud_rate (+1.8393)
- Recurring top-3 drivers across these 3 examples: card1_te_fraud_rate (x3), device_hist_fraud_rate (x3), C1 (x3)

### False positives

- Example 1: val_row=69236, calibrated_prob=0.9562, actual=0 -- top drivers: C1 (+1.9102), card1_te_fraud_rate (+1.8310), V258 (+0.8748)
- Example 2: val_row=69234, calibrated_prob=0.9562, actual=0 -- top drivers: C1 (+2.0433), card1_te_fraud_rate (+1.8108), device_hist_fraud_rate (+0.7348)
- Example 3: val_row=60354, calibrated_prob=0.9562, actual=0 -- top drivers: card1_te_fraud_rate (+2.2660), C1 (+1.9623), V258 (+1.3691)
- Recurring top-3 drivers across these 3 examples: C1 (x3), card1_te_fraud_rate (x3), V258 (x2), device_hist_fraud_rate (x1)

### False negatives

- Example 1: val_row=61479, calibrated_prob=0.0115, actual=1 -- top drivers: V313 (-1.6055), D13 (-1.5994), card1_te_fraud_rate (+1.3906)
- Example 2: val_row=5142, calibrated_prob=0.0115, actual=1 -- top drivers: D8 (-2.0688), p_emaildomain (-1.2012), device_hist_txn_count (-0.5688)
- Example 3: val_row=42498, calibrated_prob=0.0115, actual=1 -- top drivers: V313 (-1.5598), card1_te_fraud_rate (+1.2522), device_hist_txn_count (-0.8398)
- Recurring top-3 drivers across these 3 examples: V313 (x2), card1_te_fraud_rate (x2), device_hist_txn_count (x2), D13 (x1), D8 (x1)

### Borderline (0.45-0.55)

- Example 1: val_row=3362, calibrated_prob=0.4997, actual=0 -- top drivers: C14 (+1.3134), card1_te_fraud_rate (+1.1701), C13 (+1.0156)
- Example 2: val_row=48143, calibrated_prob=0.4997, actual=0 -- top drivers: card1_te_fraud_rate (+2.0437), C13 (+0.4347), card1 (+0.4341)
- Example 3: val_row=46911, calibrated_prob=0.5007, actual=1 -- top drivers: card1_te_fraud_rate (+1.5680), C13 (+0.7357), C1 (+0.6004)
- Recurring top-3 drivers across these 3 examples: card1_te_fraud_rate (x3), C13 (x3), C14 (x1), card1 (x1), C1 (x1)

## Notes

- FN waterfalls show the model's top drivers pushing *toward* legitimate for these missed frauds -- worth checking whether the recurring drivers above overlap with known FN blind spots from P6 (product_cd=W dominance, April temporal drift) even though the dedicated product_cd/card4 features tested worse in the P6 patch test.
- FP waterfalls show which features most strongly (and wrongly) pushed toward fraud for these false alarms -- compare against the P6 card4=discover / product_cd=C blind spots.
