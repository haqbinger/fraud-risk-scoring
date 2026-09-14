# 0001 — Chronological split, not random split

**Status:** Accepted
**Milestone:** M2

## Context

Random train/test splitting is standard practice for most ML problems,
but this dataset has a time dimension that matters: a card's velocity
and history features at transaction N depend on transactions 1..N-1 in
time order. Randomly shuffling rows across train/test breaks that
structure — a transaction that's chronologically "in the future" of the
model's real deployment scenario can end up in the training set.

## Decision

Use a chronological split (first 70% by `txn_ts` = train, next 15% =
val, last 15% = test) for every reported model result from M2 onward.
Random split is kept in the codebase (`evaluation/split.py`,
`scripts/random_vs_temporal.py`) ONLY to run the comparison experiment
below — never to produce a number that gets reported as this project's
actual performance.

## The experiment

`scripts/random_vs_temporal.py` trains the identical model on the
identical features, varying only the split strategy, and reports the
PR-AUC gap between them.

Result: random split PR-AUC = 0.1876, temporal split PR-AUC = 0.1593.
Gap = 0.0283 PR-AUC points (~17.8% relative overestimation from random
splitting on this dataset, with a LogisticRegression baseline).

Model Test PR-AUC vs. random-guess baseline (~0.035)
LogisticRegression ---> 0.1086 ~3.1x better
RandomForest ---> 0.2078 ~5.9x better

## Why the gap exists specifically here

Card-level velocity/history features (`velocity_5min`, `card1_hist_avg_amt`,
etc.) are the highest-value features in this project, and they're also
the ones most sensitive to split leakage: under random splitting, a
card's transactions from adjacent points in time can land on both sides
of the split, so the model implicitly "sees" a card's near-future
behavior via features computed from its past — except that past isn't
actually past relative to the random test set the way it would be in
production.

## Consequences

Every model comparison in M3, threshold selection in M4, and the served
API in M7 are evaluated exclusively on chronologically-held-out data.
This is the answer ready for "why not just use `train_test_split`?"
