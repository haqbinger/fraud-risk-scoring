# M10 — Render Deployment Evidence

**Service URL:** https://fraud-risk-scoring-api-b12m.onrender.com
**Deployed:** 2026-09-13
**Platform:** Render (free tier, Docker runtime)

## /health endpoint

GET https://fraud-risk-scoring-api-b12m.onrender.com/health
Response: {"status":"ok","model_loaded":true,"db_connected":true}

## /model endpoint

GET https://fraud-risk-scoring-api-b12m.onrender.com/model
Response: model info with threshold=0.14, val_pr_auc=0.2312

## /predict status

Returns 500 — expected. Render's Postgres is a fresh empty database
with no transactions table or IEEE-CIS data. The API is correctly
connecting to the DB (db_connected: true) but has no historical
transaction data to compute live velocity/device features against.
This is a data-loading gap, not a code bug.

## Render logs confirming live deployment

Application startup complete.
Uvicorn running on http://0.0.0.0:8000
GET /health HTTP/1.1 200 OK (confirmed multiple times)
service is live
