import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from fastapi import BackgroundTasks, FastAPI, HTTPException
from sqlalchemy import text

from fraud.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SHAPContribution,
    TransactionRequest,
)
from fraud.api.serving import build_live_feature_vector
from fraud.data import get_engine

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lgbm_calibrated.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
PREDICTION_LOGS_DDL_PATH = ROOT / "sql" / "prediction_logs.sql"

INSERT_PREDICTION_LOG_SQL = text(
    """
    INSERT INTO prediction_logs
        (transaction_id, probability_raw, probability_calibrated, decision, model_version, latency_ms)
    VALUES
        (:transaction_id, :probability_raw, :probability_calibrated, :decision, :model_version, :latency_ms)
    """
)

_state = {
    "model": None,  # CalibratedLGBMPipeline (LightGBM model + calibrator + threshold)
    "explainer": None,
    "engine": None,
    "metadata": None,
}

app = FastAPI(title="Fraud Risk Scoring API")


@app.on_event("startup")
def startup():
    _state["model"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        _state["metadata"] = json.load(f)
    _state["engine"] = get_engine()
    if SHAP_AVAILABLE:
        _state["explainer"] = shap.TreeExplainer(_state["model"].model)

    with _state["engine"].connect() as conn:
        conn.execute(text(PREDICTION_LOGS_DDL_PATH.read_text()))
        conn.commit()


def _log_prediction(transaction_id, probability_raw, probability_calibrated, decision, model_version, latency_ms):
    engine = _state["engine"]
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                INSERT_PREDICTION_LOG_SQL,
                {
                    "transaction_id": transaction_id,
                    "probability_raw": probability_raw,
                    "probability_calibrated": probability_calibrated,
                    "decision": decision,
                    "model_version": model_version,
                    "latency_ms": latency_ms,
                },
            )
            conn.commit()
    except Exception as exc:
        print(f"prediction logging failed: {exc}")


def _flatten_request(request: TransactionRequest, txn_ts) -> dict:
    raw_input = {
        "card1": request.card1,
        "transaction_amt": request.transaction_amt,
        "txn_ts": txn_ts,
        "product_cd": request.product_cd,
        "p_emaildomain": request.p_emaildomain,
        "r_emaildomain": request.r_emaildomain,
        "device_type": request.device_type,
        "device_info": request.device_info,
        "card2": request.card2,
        "card3": request.card3,
        "card4": request.card4,
        "card5": request.card5,
        "card6": request.card6,
        "addr1": request.addr1,
        "addr2": request.addr2,
        "dist1": request.dist1,
        "dist2": request.dist2,
    }
    raw_input.update(request.C)
    raw_input.update(request.D)
    raw_input.update(request.V)
    raw_input.update(request.M)
    return raw_input


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()

    if _state["model"] is None or _state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = _state["metadata"]
    pipeline = _state["model"]
    txn_ts = request.txn_ts or datetime.now(timezone.utc)
    if txn_ts.tzinfo is not None:
        txn_ts = txn_ts.astimezone(timezone.utc).replace(tzinfo=None)

    raw_input = _flatten_request(request, txn_ts)

    X = build_live_feature_vector(
        raw_input,
        engine=_state["engine"],
        selected_features=metadata["selected_features"],
        cat_cols=metadata["categorical_features"],
        categories_map=pipeline.categories_map,
    )

    prob_raw = float(pipeline.predict_proba_raw(X)[0])
    prob_calibrated = float(pipeline.predict_proba(X)[0])
    threshold = metadata["threshold"]
    decision = "BLOCK" if prob_calibrated >= threshold else "ALLOW"

    if SHAP_AVAILABLE and _state["explainer"] is not None:
        shap_values = _state["explainer"].shap_values(X)
        if isinstance(shap_values, list):
            row_shap = shap_values[-1][0]
        else:
            row_shap = shap_values[0]
        order = np.argsort(-np.abs(row_shap))[:5]
        cat_cols = metadata["categorical_features"]
        top_features = []
        for i in order:
            feature = X.columns[i]
            if feature in cat_cols:
                value = raw_input.get(feature)
                if value is None:
                    value = "missing"
            else:
                value = float(X.iloc[0, i])
            top_features.append(
                SHAPContribution(
                    feature=feature,
                    value=value,
                    shap_contribution=float(row_shap[i]),
                )
            )
    else:
        top_features = []

    latency_ms = (time.perf_counter() - start) * 1000

    background_tasks.add_task(
        _log_prediction,
        request.transaction_id,
        prob_raw,
        prob_calibrated,
        decision,
        metadata["model_type"],
        latency_ms,
    )

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability_raw=prob_raw,
        fraud_probability_calibrated=prob_calibrated,
        decision=decision,
        threshold=threshold,
        top_features=top_features,
        model_version=metadata["model_type"],
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = _state["model"] is not None
    db_connected = False
    if _state["engine"] is not None:
        try:
            with _state["engine"].connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
        except Exception:
            db_connected = False

    status = "ok" if model_loaded and db_connected else "degraded"
    return HealthResponse(status=status, model_loaded=model_loaded, db_connected=db_connected)


@app.get("/model", response_model=ModelInfoResponse)
def model_info():
    if _state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")

    metadata = _state["metadata"]
    return ModelInfoResponse(
        model_version=metadata["model_type"],
        model_type="lightgbm",
        val_pr_auc=metadata["val_pr_auc"],
        test_pr_auc=metadata["test_pr_auc"],
        calibration_method=metadata["calibrator_type"],
        decision_threshold=metadata["threshold"],
        cost_false_negative=metadata["cost_model"]["cost_false_negative"],
        cost_false_positive=metadata["cost_model"]["cost_false_positive"],
    )
