ADR 0007: Explainability policy for SHAP-backed fraud scoring

- Status: Accepted
- Date: 2026-09-13

Context

M5 adds SHAP-based explainability on top of the fraud model. This is important for model governance, stakeholder communication, and interviews, but it introduces a common misunderstanding:

- SHAP explains what the model weighted
- SHAP does not explain causal fraud mechanisms
- SHAP does not imply that a feature caused the event

This distinction matters because fraud models are often reviewed by non-technical stakeholders who are tempted to read SHAP results as cause-and-effect statements.

Decision

We will explain SHAP output using the following language:

- “the model relied heavily on X for this prediction”
- “X was a strong contributor to the model score”
- “X was one of the top features influencing the prediction”

We will not use language such as:

- “X caused this fraud”
- “X indicates fraud”
- “X is the reason the transaction was fraudulent”

Why

This project is about fraud-risk scoring, not causal attribution. A feature may be strongly predictive of fraud without being a direct causal mechanism in the real world. For example, a high transaction amount or suspicious device history may raise the model score even when the transaction is ultimately legitimate.

SHAP is therefore a model-interpretation tool, not a fraud-investigation or causal-analysis tool.

Consequences

- All documentation and write-ups must phrase results as model reliance, not causality.
- The false-positive example is explicitly treated as a model-weighting case, not as proof of fraud behavior.
- This keeps the model explainable without overstating what the model actually knows.

Operational interpretation

In the final explainability report:

- global SHAP rankings describe which features the model relies on overall
- local SHAP values explain how the model weighed specific transactions
- the model should be described as “relying heavily on X” rather than “X caused the fraud”

This preserves scientific honesty and makes the model easier to defend in interviews and governance reviews.

Summary

The policy is simple:

- SHAP explains model behavior
- not reality
- not causality
- not fraud mechanisms

The model’s explanation is therefore phrased as:
- “the model relied heavily on X for this prediction”

and never:
- “X caused this fraud”