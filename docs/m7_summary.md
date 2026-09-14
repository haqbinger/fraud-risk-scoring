M7 Summary: Serving layer, leakage-safe inference, and runtime hardening

What M7 actually builds

This milestone did not simply produce a REST API. It built a production-style fraud scoring service that does the hard parts correctly under live conditions.

The system receives a transaction request, computes the required historical features from Postgres on demand, and scores it using the selected XGBoost model with a Platt-calibrated probability. The live feature computation is intentionally leakage-safe:

- card1 velocity is queried only from prior events before `txn_ts`
- device fraud rate is also computed strictly before `txn_ts`
- this mirrors the same rule enforced in `sql/features_m1.sql`

This means the API does not just call a model; it enforces the same temporal logic used during offline training at inference time. The tradeoff is straightforward: each prediction performs two Postgres queries, which is correct but not free.

M7 also fixed the artifact-loading path for the model itself. The API loads the winning model and the calibration metadata at startup instead of rerunning training logic or reconstructing state on the fly. That includes:

- `winning_model.json`
- `platt_calibrator.joblib`
- `model_metadata.json`

Those objects supply the model, the calibrated probability transform, the threshold, the feature columns, and the category maps that the scoring service needs to operate in production. The result is a serving layer that is deterministic and deployable without training-time assumptions.

Finally, the service computes per-request SHAP explanations using `TreeExplainer` on the raw XGBoost output rather than the calibrated probability. That is consistent with the project’s explainability policy: SHAP explains the model’s decision contribution, while calibration explains confidence adjustment.

What surfaced during live testing

The most important part of M7 was not the FastAPI scaffolding. It was the fact that the code had to survive real operational failures.

Bug 1: timezone mismatch between naive and aware datetimes

The first runtime bug showed up when the request path compared Python timestamps with database timestamps. Postgres stores timestamps without timezone, while Python was generating `datetime.now(timezone.utc)`, which is timezone-aware. That led to a subtraction crash when the code tried to compute historical windows using a aware/naive datetime mix.

The fix was simple but critical:

- normalize the request timestamp to naive UTC before building the Postgres query
- ensure the comparison is against a consistent datetime type in both Python and SQL

This made the leakage-safe query logic operate correctly instead of failing on the boundary between application code and the database.

Bug 2: missing device history created an object-dtype feature column

The second bug appeared when a transaction had no device history. In that case, the feature frame could include a column with object dtype instead of numeric dtype. XGBoost rejects non-numeric dtypes during inference, so the model would fail even though the feature logic itself was conceptually valid.

The fix was to coerce the numeric feature frame to `float` before passing it to the model:

- explicitly cast the serving feature matrix to float
- ensure all feature columns are numeric before XGBoost scoring

This turned a silent feature-engineering edge case into a hard failure that was then eliminated at the source.

Why BackgroundTasks are the right logging mechanism

Prediction logging is implemented with FastAPI `BackgroundTasks` for a specific reason: logging must not affect the time the client waits for the response.

If the API waited for a database write or an external log sink before returning the prediction, the user-facing latency would be inflated. With `BackgroundTasks`, the response is returned immediately and the logging work is scheduled afterward. That keeps the scoring API responsive while still recording predictions for auditability and monitoring.

SHAP vs. calibration: different jobs, different outputs

This distinction matters a lot in a production ML system.

- SHAP explains how the raw model is weighting the features for a given prediction
- Platt calibration explains how the probability scale is adjusted to better represent confidence

These answer different questions:

- SHAP tells us what the model relied on
- calibration tells us whether the probability is trustworthy

The project therefore computes SHAP on the raw XGBoost output and leaves calibration as a separate probability transformation layer. Mixing them would blur the interpretation and would not faithfully explain either the model decision or the confidence calibration step.

Honest limitation

The live inference path is correct, but it is not the fastest possible architecture. Because each prediction asks Postgres for feature history, the system incurs additional latency per request. At real traffic scale, this would likely need a feature store, cache, or a more mature online feature serving layer. That is an honest tradeoff of M7: correctness and leakage safety first, latency optimization later.

Final takeaway

The model was never the bottleneck in M7. The engineering decisions — timezone handling, dtype safety, artifact persistence, leakage-safe live feature computation — were.
