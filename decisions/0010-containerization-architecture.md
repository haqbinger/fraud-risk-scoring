# ADR 0010 — Containerization architecture for the fraud-serving stack

**Status:** Accepted
**Milestone:** M9

## Context

Milestone 9 moved the project from a local developer setup into a reproducible containerized runtime. The system is composed of several moving parts: the FastAPI service, the PostgreSQL database, model and calibration artifacts, SQL DDL files, and a few Python packages with tight version constraints. The goal was to make the app run consistently in Docker while preserving the project’s operational correctness.

This was not a generic “containerize it” exercise. Several real constraints emerged during runtime setup:

- the application needed to read SQL DDL files from `sql/` at startup
- the runtime needed access to persisted model and calibration artifacts
- the service needed to discover calibration code outside the `fraud` package
- Postgres had to be ready before the API started
- the Python version had to satisfy the XGBoost dependency floor
- Docker had to build with exactly the dependency versions that had been validated locally

The first container start surfaced a direct failure: the API tried to read `/app/sql/prediction_logs.sql` and failed because the SQL directory was not copied into the image. That revealed a more general principle: DDL belongs in `sql/`, and the container has to include it as part of the runtime bundle.

The project also had a dependency upgrade issue. `xgboost==3.4.1` requires Python 3.12 or later, so the Dockerfile had to move from Python 3.11 to Python 3.12. Separately, an unpinned `mlflow` dependency resolved to the dummy PyPI package `mlflow==0.0.1` in the Linux build environment, causing a `ResolutionImpossible` error. That was a real packaging bug, not a preference issue.

## Decision

We will use a single-stage Docker build based on `python:3.12-slim`.

This is the chosen architecture because image size is not a meaningful constraint for a portfolio project, and a single-stage build is easier to read, debug, and maintain. The project value is in correctness and operational clarity, not in shaving a few hundred megabytes off the image. The Python version is not arbitrary: `xgboost==3.4.1` requires Python 3.12+, so the Dockerfile must run on 3.12-slim.

We will pin every package in `requirements.txt` to exact versions. This includes packages such as:

- `pandas==3.0.5`
- `sqlalchemy==2.0.52`
- `xgboost==3.4.1`
- `mlflow==3.16.0`

This avoids accidental dependency drift during Docker builds. In this project, an unpinned `mlflow` package resolved to an invalid placeholder package, which caused the Linux build to fail before the app ever started. Exact pinning ensures that the container builds the same environment that was tested locally.

We will also copy the entire `sql/` directory into the container image with:

- `COPY sql/ ./sql/`

This is required because the API reads `sql/prediction_logs.sql` at startup to create the `prediction_logs` table. The correct pattern is to keep SQL DDL in `sql/` and let the app execute it from there, rather than inlining DDL inside Python.

For the model and calibration artifacts, we will mount the `models/` (or `data/processed/` equivalent) directory as a read-only bind mount. These artifacts are loaded at startup and should be refreshed by restarting the API rather than by rebuilding the image. This includes:

- `winning_model.json`
- `platt_calibrator.joblib`
- `model_metadata.json`

The same principle applies to `src/calibration/`, which will be mounted separately as a bind mount. This directory sits outside the installed `fraud` package and is not automatically discovered by `pip install -e .`, so it must be explicitly mounted in the container.

The Postgres dependency will be managed with a healthcheck and a startup dependency. We will use:

- `pg_isready` in the database healthcheck
- `depends_on: condition: service_healthy`

This ensures the API does not start before PostgreSQL has accepted connections. The container network will route the API to Postgres using the service name `postgres` on internal port `5432`. The host port mapping `5433` is only for connections from the developer machine, not for API-to-database communication inside the Docker network.

## Alternatives considered

1. Multi-stage Docker build
   - Pros: smaller final image, more production-like layering.
   - Cons: adds complexity without solving an actual project constraint; for a portfolio project, readability and debuggability matter more than image minimization.

2. Python 3.11 image
   - Pros: familiar default.
   - Cons: incompatible with `xgboost==3.4.1`, which requires Python 3.12+.

3. Unpinned dependency set
   - Pros: easier to maintain superficially.
   - Cons: can resolve to bogus or incompatible packages in Linux builds and create nondeterministic environment failures.

4. Keeping SQL DDL in code instead of `sql/`
   - Pros: simpler for a one-file prototype.
   - Cons: violates the project’s schema-management rule, makes schema changes hard to review, and breaks runtime assumptions when the container does not include the SQL files.

5. Starting the API before Postgres is ready
   - Pros: simpler orchestration.
   - Cons: race conditions and startup failures when the database is still coming up.

## Consequences

- The container build is reproducible and reflects the exact dependency environment that was validated locally.
- The Python runtime is compatible with the required XGBoost version, preventing a common version-compatibility failure.
- SQL DDL is included in the image and remains discoverable in the `sql/` folder, consistent with the project’s architecture.
- Model and calibration artifacts can be updated without rebuilding the image, making retraining and redeploys operationally simpler.
- The API starts reliably only after Postgres is healthy, reducing startup flakiness.
- Container networking follows the correct internal pattern: the application uses `postgres:5432` inside Docker, while host-level access remains on `localhost:5433` for local debugging.
- The system is now more deployment-ready, but it is also explicit about the real tradeoff: containerization improves reproducibility and startup safety, while live feature and model serving remain the more important engineering constraints.
