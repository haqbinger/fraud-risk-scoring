# M9 Summary: Containerization, dependency pinning, and startup safety

## What M9 actually delivers

M9 did not merely add a Dockerfile. It turned the fraud-serving stack into a reproducible runtime environment with the same behavior the project had already validated locally.

The key result was a containerized API stack that includes:

- a FastAPI app serving fraud predictions
- a PostgreSQL database for feature lookups and logging
- SQL schema files stored under `sql/` and executed at startup
- persisted model and calibration artifacts mounted from the host or workspace
- a Python runtime that matches the XGBoost compatibility requirement

This mattered because the service depends on more than just code. It depends on the model files, the calibration state, the SQL DDL, the database readiness, and the exact dependency versions used during validation. M9 made those dependencies explicit instead of implicit.

## Why the container decisions were not cosmetic

The first major decision was to use a single-stage Docker image based on `python:3.12-slim`.

This was not arbitrary. The project had already reached a real dependency constraint: `xgboost==3.4.1` requires Python 3.12 or newer. The Docker image therefore had to move from 3.11 to 3.12, and that is a valid engineering requirement rather than a personal preference. A single-stage build was then chosen because the project’s main objective was clarity and debugability, not optimized image slimming.

The second major decision was to pin every dependency in `requirements.txt` to exact versions. That was a necessary fix, not a style preference.

The concrete failure was an unpinned `mlflow` package resolving to a dummy PyPI package (`mlflow==0.0.1`) in the Linux build environment, which caused a `ResolutionImpossible` error. That is the exact sort of problem that only shows up in a clean container build. Pinning the dependency set prevents the environment from drifting away from the version set that was actually tested.

## The SQL DDL issue was a real architecture bug

One of the most important runtime bugs in M9 was not in the model. It was in the container layout.

The API expects to read `sql/prediction_logs.sql` at startup to create the `prediction_logs` table. The first container run failed with `FileNotFoundError` because the `sql/` directory was not copied into the image. That exposed the correct pattern for this project: SQL DDL belongs in `sql/`, not in Python strings embedded in the application code.

Once the directory was copied into the image, the startup path became deterministic and aligned with the project’s schema versioning principle. This is a core project decision: database schema files are part of the codebase and must be shipped with the service.

## Bind mounts were chosen to respect the runtime model contract

The serving app loads artifacts at startup from the filesystem rather than rerunning training logic. For that reason, the project mounts the model and metadata directories as read-only bind mounts:

- `models/` or `data/processed/` for the persisted runtime artifacts
- `src/calibration/` for the non-package calibration code

This is important because the app does not just read a model file; it reads the full inference contract: the winning model, the Platt calibrator, the feature metadata, and the calibration code needed by the service. The calibration directory is separated from the `fraud` package because it is outside the package install path and would otherwise be missed by an editable install.

This is a good example of the difference between “code runs locally” and “code runs in the real runtime.” The bind mounts make the runtime assumptions explicit instead of hiding them inside a developer environment.

## Health checks and networking made startup reliable

The database startup path was hardened with a Postgres healthcheck and a `depends_on: condition: service_healthy` dependency. That means the API does not start before the database accepts connections, which removes a classic race condition from the stack.

The networking rule is also explicit and correct:

- internal container traffic uses `postgres` on port `5432`
- host-side access uses `localhost:5433` for local developer connection

This is the right separation of concerns. The application does not talk to `localhost` inside Docker; it talks to the internal service name. The host mapping exists only for external access and is not the service-to-service network path.

## The real win in M9

Not a smaller image or a prettier Dockerfile, but a reproducible, robust runtime chain:

- Python version matches the dependency floor
- dependency pins remove silent drift
- SQL schema files are included with the service
- model artifacts are mounted and loaded correctly
- Postgres readiness is enforced before startup
- internal networking matches the container architecture

This is what turns a local prototype into something that can be reliably built, started, and debugged in a real environment.
