FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt pyproject.toml ./
COPY src/ ./src/
COPY sql/ ./sql/
COPY data/processed/ ./data/processed/

RUN pip install --no-cache-dir -r requirements-serve.txt && \
    pip install --no-cache-dir -e . && \
    pip cache purge

EXPOSE 8000

CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]