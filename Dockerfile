FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
COPY sql/ ./sql/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

RUN mkdir -p data/processed

EXPOSE 8000

CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]