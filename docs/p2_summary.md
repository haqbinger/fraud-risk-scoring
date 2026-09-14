# P2 Feature Selection — Summary

## Result

V1 PR-AUC: 0.219 (20 features) → V2 ceiling: 0.5725 (411 features) → Clean set: 0.5691 (182 features)

The V1 model was starved of signal, not architecturally wrong. Full feature engineering accounts for the entire gap.

## What was dropped and why

| Reason                          | Count   |
| ------------------------------- | ------- |
| Near-zero gain (< 0.0001)       | 95      |
| Negative permutation importance | 81      |
| Spearman redundancy (> 0.95)    | 53      |
| **Total dropped**               | **229** |

## Most important finding

297/411 features disagreed by >50 positions between gain and permutation rank. Gain importance reflects what the tree used during training; permutation importance reflects what actually generalizes. Where they disagree badly, permutation wins. Several high-gain V-columns were memorizing training structure with no val signal — all removed.

## Missing columns (feed back into P1)

card2-6, addr2, dist1, dist2, R_emaildomain never reached the feature matrix. dist1/dist2 are billing-to-shipping distance features — strong fraud signal candidates. Quick P1 patch recommended before P3.

## Next steps

1. P1 patch: pipe missing columns through features_m1.sql and load_data(), rerun kitchen-sink
2. If ceiling moves: rerun P2 on expanded set
3. If ceiling doesn't move: proceed to P3 temporal CV
