0003 — PR-AUC as primary ranking metric, not ROC-AUC or accuracy

Status: Accepted
Milestone: M2

Context

Fraud is a small minority class in this dataset. Accuracy is meaningless
here — a model predicting "not fraud" for everything scores extremely
high accuracy while catching zero fraud. ROC-AUC is commonly used
instead, but it's evaluated partly against the true-negative rate, which
is trivially easy to get right when negatives dominate — it can look
deceptively strong even for a mediocre fraud-ranking model.

Decision

Use PR-AUC (average precision) as the primary metric for model
selection and comparison across M2-M3. ROC-AUC, precision, recall, F1,
and Brier score are all reported alongside it — no single number is
ever the whole story — but PR-AUC is what determines which model wins.

Alternatives considered

- ROC-AUC alone — rejected: overstates performance under severe class
  imbalance, a well-known critique of ROC-AUC on rare-event problems.
- Accuracy — rejected outright, not reported as a headline number at all.
- F1 — kept as a secondary metric, rejected as primary: F1 weights
  precision and recall equally, which contradicts the actual business
  cost model (M4), where false negatives and false positives have
  different real dollar costs (`COST_FALSE_NEGATIVE` vs
  `COST_FALSE_POSITIVE` in `config.py`).

Consequences

Model comparison in M3 and threshold selection in M4 are both anchored
to PR-AUC plus the explicit cost model — never to a single
accuracy-looking number. This is the answer ready for "why not just use
accuracy or ROC-AUC?"
