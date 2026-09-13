from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    card1: int
    transaction_amt: float = Field(gt=0)
    txn_ts: Optional[datetime] = None
    product_cd: Optional[str] = None
    p_emaildomain: Optional[str] = None
    device_type: Optional[str] = None
    device_info: Optional[str] = None


class SHAPContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability_raw: float
    fraud_probability_calibrated: float
    decision: str
    threshold: float
    top_features: list[SHAPContribution]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    trained_on_rows: int
    val_pr_auc: float
    calibration_method: str
    decision_threshold: float
    cost_false_negative: float
    cost_false_positive: float
