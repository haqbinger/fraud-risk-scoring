ADR 0009 — Prediction logging and test architecture

Status: Accepted
Milestone: M8

Context

M8 operationalized the serving layer by making prediction logging and the test harness explicit, versioned, and robust. The project already had a live API that could score transactions and return explanations, but it still had a gap between “works in a notebook” and “works under production conditions.” The main operational concern was the prediction log table: it needed to be created reproducibly, versioned in SQL, and kept aligned with the API contract. This table also feeds later monitoring and drift work, so its schema matters beyond the request path.

The project also needed a testing strategy that separated hermetic checks from database-dependent validation. Different tests require different dependencies: some should run with no Postgres or model available, while others legitimately depend on local processed artifacts or a running database. Without this distinction, the test suite would either be brittle, fail on developer machines without Docker, or mask real errors in the production path.

During live testing, one real bug surfaced: the logging helper called `engine.connect()` unconditionally. In the fake or minimal test state, `engine` could be `None`, causing the background task to raise and fail the request itself. This made it clear that logging must be treated as infrastructure, not as part of the core prediction path.

The schema also needed to reflect the actual data model. A test initially asserted that `transaction_amt` lived in the feature SQL table, but the real runtime join path in `data.py` brings that field from the transaction table at query time. That was a test assumption, not a production bug, but it was still worth fixing because it revealed how easy it is to test the wrong layer.

Decision

We will define the prediction log table DDL in `sql/prediction_logs.sql` and have the application read and execute that file at startup. This is the single source of truth for the schema. Keeping the SQL in `sql/` makes it visible to version control, reviewable in pull requests, and consistent with the project’s broader principle that SQL belongs in the SQL folder rather than hidden inside Python strings.

The prediction log table will include:

- `id` as `BIGSERIAL`
- `transaction_id` as `TEXT`
- `probability_raw`
- `probability_calibrated` as `DOUBLE PRECISION NOT NULL`
- `decision` as `TEXT NOT NULL CHECK IN ('ALLOW', 'BLOCK')`
- `model_version` as `TEXT NOT NULL`
- `latency_ms`
- `created_at` as `TIMESTAMPTZ DEFAULT now()`
- indexes on `created_at` and `transaction_id`

This table is intentionally the direct input to later drift monitoring work in milestone 11, so its shape must remain stable and explicit.

We will treat prediction logging as a best-effort background side effect. `_log_prediction` will become a no-op when the engine is `None` and will be wrapped in `try/except` so logging failures cannot take down `/predict`. This mirrors the `BackgroundTask` pattern from M7: the request returns to the client without waiting on storage work, and any logging failure is isolated to the infrastructure layer.

We will split the test suite into four separate files with clear dependency boundaries:

- `test_api.py` — fully hermetic, with no DB and no real model; it uses `FakeModel`, `FakeCalibrator`, and `FakeExplainer` injected into `_state` via `conftest`
- `test_model_output.py` — depends on processed data and model artifacts; it skips if those files are missing
- `test_leakage.py` — requires a live Postgres instance; it skips if the database is unreachable
- `test_schema.py` — requires a live Postgres instance; it skips if the database is unreachable

Database-dependent tests will use skip semantics instead of fail-fast behavior when Postgres is unavailable. This keeps CI green on a machine without Docker and still allows the tests to catch real issues when the database is running.

Alternatives considered

1. Keep the prediction log schema inline in `main.py`
   - Pros: quick to implement and easy to read in one file.
   - Cons: not version-controlled as schema, not discoverable in `sql/`, and easy to drift from the actual database state.

2. Make logging synchronous and blocking
   - Pros: simpler mental model.
   - Cons: adds latency to `/predict`, makes the endpoint more fragile, and violates the design principle that prediction is the product while logging is support infrastructure.

3. Fail the request when database logging is down
   - Pros: strict audit completeness.
   - Cons: wrong tradeoff for a production API; a logging outage should not turn a valid fraud score into a failed customer request.

4. One monolithic integration test
   - Pros: simple at first glance.
   - Cons: does not isolate dependency boundaries and makes it harder to tell whether failures are due to API logic, missing artifacts, schema drift, or database availability.

Consequences

- The schema is now under source control and review, which makes changes to the logging table visible and intentional.
- Prediction logging is isolated from the request path and does not compromise latency or correctness.
- The API remains robust even when the database is unavailable or when the application is running in a minimal fake state.
- The testing suite distinguishes between hermetic validation and environment-dependent checks, which makes failures more interpretable.
- DB tests skip when Postgres is unavailable, which is a practical CI choice, but it also means a permanently down database could silently hide regressions unless the environment is monitored.
- The main production bug we corrected was:
  1. `_log_prediction` called `engine.connect()` unconditionally; with `engine=None`, the background task raised and propagated through Starlette's test client, failing the request. This was fixed by guarding `engine` and wrapping the logging call in `try/except`.
- We also corrected a test assumption in `test_schema.py`: `transaction_amt` belongs to the transaction table, not to the feature SQL table, because the join happens at query time in `data.py`. This was a useful reminder that tests can be wrong even when the application code is right.

The full suite result with the database running is: 33 passed, 0 failed, 0 skipped. That is the real milestone: not every check runs on every machine, but the checks that do run are meaningful and the project has a clear signal for future regressions.
