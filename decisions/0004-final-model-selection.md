# 0004 — Final model selection: XGBoost

**Status:** Accepted
**Milestone:** M3

## Context

Compared Logistic Regression, Random Forest, and XGBoost on the same
temporal split, same feature matrix, same train-median imputation
(`scripts/model_comparison.py`).

VAL PR-AUC (used for selection):

- LogisticRegression: 0.1593
- RandomForest: 0.2081
- XGBoost: 0.2312

Fit time: LogisticRegression 0.81s, RandomForest 64.49s, XGBoost 8.29s.

TEST PR-AUC (XGBoost only, evaluated once, after selection was already
locked in on val): 0.2189.

## Decision

XGBoost. It beats RandomForest by 0.0231 PR-AUC points (~11.1% relative)
on val, and beats LogisticRegression by 0.0719 points (~45.1% relative).
The test PR-AUC (0.2189) is close to val (0.2312) — a normal, modest
generalization gap, not a red flag for overfitting.

This isn't a tradeoff decision — XGBoost wins on PR-AUC _and_ is
~7.8x faster to train than RandomForest (8.29s vs 64.49s), so there's no
"better but slower" tension to argue through. The only real cost is
XGBoost's larger hyperparameter surface (n_estimators, max_depth,
learning_rate, subsample, colsample_bytree all set here, none tuned yet)
— more knobs to justify if asked "why these settings" in an interview.

## Alternatives considered

- **Logistic Regression** — most interpretable (coefficients directly
  readable), fastest to train, but clearly the weakest PR-AUC — real
  signal, underfits the nonlinear interactions in the velocity/history
  features.
- **Random Forest** — solid PR-AUC, no scaling needed, reasonably
  interpretable via feature importances — but ~8x slower to train than
  XGBoost for a worse PR-AUC. No scenario here where RandomForest is the
  better choice.

## Consequences

XGBoost is the model served in M7 and explained in M5. SHAP's
`TreeExplainer` has fast, first-class support for XGBoost specifically —
a practical reason on top of the accuracy/speed reason to prefer it,
worth stating explicitly if asked.
