ADR 0008 — Serving architecture for leakage-safe fraud inference

Status: Accepted
Milestone: M7

Context

Milestone 7 completed the production-facing serving layer for the fraud model. The system is built around a FastAPI application that receives transaction requests, computes the leakage-safe features required by the model at inference time, and returns a calibrated fraud probability plus an explainability payload.

This project has already established a strict leakage rule in the SQL feature pipeline: only historical transaction and device behavior strictly before the current event timestamp may be used. The runtime serving path therefore had to follow the same rule. The primary design choice was whether to precompute features offline, use a feature store, or compute them live per request from Postgres. The project also needed to persist the trained model and calibration artifacts so the API could score requests without rerunning training logic.

M7 also surfaced a few real runtime problems that only appear when the model is exercised against live data, not when it is trained in notebooks. These included timezone mismatches between aware and naive Python datetimes, and a dtype issue when device history was missing and produced an object-typed feature frame.

The final serving contract is:

- POST /predict
- GET /health
- GET /model

With a threshold of 0.14, consistent with the cost-sensitive threshold decision in ADR 0006.

Decision

We will serve predictions using a live inference path that computes the required features in Postgres for each request, rather than relying on precomputed features or a dedicated feature store.

Specifically, the API queries:

- card1 velocity prior to the current transaction timestamp
- device fraud rate prior to the current transaction timestamp

These queries are performed strictly before `txn_ts`, matching the leakage-safe rule enforced in `sql/features_m1.sql`. The tradeoff is that each prediction requires two Postgres queries, which adds latency, but it preserves correctness and prevents leakage at serving time.

We will also persist the calibration and metadata artifacts separately from the winning model artifact. The serving layer will load all required runtime objects at startup:

- `winning_model.json`
- `platt_calibrator.joblib`
- `model_metadata.json`

The FastAPI service loads the threshold, category maps, medians, and feature columns from these at startup, without rerunning training code or reconstructing training state on demand.

For explainability, we will compute SHAP values per request using `TreeExplainer` on the raw XGBoost output, not on the Platt-calibrated probability. This follows ADR 0007: SHAP explains the model’s decision path, while calibration explains the confidence adjustment applied to the score.

Prediction logging will be handled via FastAPI `BackgroundTasks`, so the API can return the prediction response without waiting for database logging or network-side persistence work to complete, keeping the client-facing latency low.

Alternatives considered

1. Precomputed feature tables
   - Pros: lower latency at inference time.
   - Cons: harder to guarantee leakage-safe behavior unless the same timestamp gating logic is enforced faithfully in offline and online pipelines; also adds operational complexity around feature freshness and drift.

2. Feature store or streaming feature service
   - Pros: more scalable and production-friendly at larger traffic volumes.
   - Cons: not needed for this M7 milestone, and it would broaden the scope beyond the project’s current objective of validating the serving layer in a controlled environment.

3. Calibrating before SHAP or using calibrated probabilities for explanation
   - Pros: easy conceptual shortcut.
   - Cons: not aligned with the project’s model explanation policy; SHAP on calibrated probabilities mixes explanation of model decisions with confidence adjustment and does not answer the same question as the raw feature contribution analysis.

4. Synchronous logging inside the request path
   - Pros: simple implementation.
   - Cons: adds latency to the user-visible response and makes serving less robust under load or slow storage.

Consequences

- The API enforces the same leakage-safe logic at inference time as in the training SQL feature logic.
- The system is correct but not yet the lowest-latency production design; each prediction performs two live Postgres lookups.
- Model and calibration artifacts are persisted intentionally and loaded at startup, making deployment simpler and more deterministic.
- SHAP outputs remain interpretable as model reliance, while Platt calibration remains a probability-quality layer.
- Logging is decoupled from the response path, which improves client latency and operational robustness.
- Two runtime bugs were found during live testing and fixed in production code:
  1. timezone-aware/naive datetime collision from Postgres storing timestamps without timezone while Python used `datetime.now(timezone.utc)`; fixed by normalizing to naive UTC before querying Postgres.
  2. `None` device history producing an object-dtype column; fixed by casting the numeric feature frame to `float` before XGBoost scoring in the serving layer.

These are the main tradeoffs of M7: correctness and reliability were prioritized, with a clear path to optimize latency later via feature storage or caching if production scale requires it.
