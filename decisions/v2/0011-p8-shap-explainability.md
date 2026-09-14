ADR V2-0011: P8 SHAP Explainability

Status

Accepted

Method

TreeExplainer on uncalibrated LightGBM model (SHAP operates on raw model output, not calibrated probabilities — per V1 ADR 0007 policy, preserved in V2).
Val set only. 15 outputs: global beeswarm, importance bar chart, 12 waterfall plots (3x TP/FP/FN/borderline).

Top 10 Features by Mean |SHAP|

| Rank | Feature                    | Mean abs SHAP |
| ---- | --------------------------- | ------------- |
| 1    | card1_te_fraud_rate        | 0.846         |
| 2    | C13                        | 0.394         |
| 3    | addr1_te_txn_count         | 0.337         |
| 4    | transaction_amt            | 0.301         |
| 5    | p_emaildomain_te_txn_count | 0.287         |
| 6    | p_emaildomain              | 0.264         |
| 7    | D2                         | 0.260         |
| 8    | card1_hist_txn_count       | 0.212         |
| 9    | dist1                      | 0.209         |
| 10   | C5                         | 0.203         |

card1_te_fraud_rate dominates at 2x the runner-up — the P1 leakage-safe target encoding decision accounts for the majority of model signal.

Key Findings

False Positive Root Cause

Val rows 69234/69236 (FP) and 1710/1715 (TP) are near-adjacent indices, both driven by card1_te_fraud_rate + C1. Classic expanding-window target encoding failure mode: once a card has confirmed fraud, its historical fraud-rate spikes for all subsequent transactions on that card — model flags legitimate follow-up activity on the same card ("guilt by association"). Quantitative verification recommended if FP analysis is revisited.

False Negative Root Cause

Missed frauds show negative SHAP contributions from V313, D13, D8 (anonymized Vesta/time-delta features actively pushing toward "legitimate"), overriding card1_te_fraud_rate's fraud signal. Model loses when anonymized behavioral features disagree with card-level risk history.

Decision

P8 complete. All findings cross-referenced in p8_shap_summary.md against P6 blind-spot analysis. V2 model pipeline finalized — proceed to infrastructure update (save artifacts, redeploy to Render).

Artifacts

reports/figures/:

- p8_shap_global_summary.png
- p8_shap_importance.png
- p8waterfall_tp{1,2,3}.png
- p8waterfall_fp{1,2,3}.png
- p8waterfall_fn{1,2,3}.png
- p8waterfall_borderline{1,2,3}.png
- p8_shap_summary.md
