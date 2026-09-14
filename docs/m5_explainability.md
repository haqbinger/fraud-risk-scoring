M5: Explainability

Scope

This module explains the raw XGBoost model output, not the Platt-calibrated probabilities from M4. The two are complementary:

- SHAP answers: “what did the model rely on for this score?”
- Calibration answers: “how confident should we be in that score?”

These are different questions and should not be confused.

Correlation caveat

Every SHAP explanation is phrased as:

- “the model relied heavily on X for this prediction”

We do not say:

- “X caused fraud”
- “X indicates fraud”

SHAP shows what the model weighted. It does not establish causation or real-world fraud mechanisms.

Global behavior

Across the sampled test set, the model relied most heavily on:

- transaction amount
- product code
- device fraud history
- card1 historical transaction statistics
- time since previous transaction
- email domain
- velocity features

These are exactly the features that align with fraud-risk logic: unusual spend, suspicious historical behavior, and device/email anomalies.

True positive example

The true positive case was a high-scoring fraud example with a probability of 0.891. The model relied heavily on `device_hist_fraud_rate`, `product_cd_C`, and `p_emaildomain_gmail.com` — device-level fraud history and product/email context drove the high score.

False positive example

The false positive case was a high-scoring legitimate transaction with a probability of 0.850. The model relied heavily on `transaction_amt`, `device_hist_fraud_rate`, and `p_emaildomain_gmail.com` — high-value and suspicious-looking patterns that pushed the score up on a transaction that turned out to be benign. The model was not "wrong" in a causal sense; it weighted features that look risky in aggregate.

Borderline example

The borderline case had a fraud probability of 0.415 and was near the decision threshold. The model relied on a mix of signals: `product_cd_C` was a negative contributor, while `card1_hist_std_amt` and `amount_zscore_vs_card1_history` pushed the score upward — competing signals rather than a single dominant risk flag.

Interpretation

The key point is that SHAP is explaining model weighting, not fraud causality. The model is not telling us that a given feature “caused fraud”; it is telling us that the feature was one of the strongest drivers of the model’s score in that context.
