from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field

V_COLS = [f"V{i}" for i in range(1, 340)]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    card1: int
    transaction_amt: float = Field(gt=0)
    txn_ts: Optional[datetime] = None
    product_cd: Optional[str] = None
    p_emaildomain: Optional[str] = None
    r_emaildomain: Optional[str] = Field(default=None, alias="R_emaildomain")
    device_type: Optional[str] = None
    device_info: Optional[str] = None

    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    # Raw V/C/D/M columns straight from the IEEE-CIS schema, keyed by their
    # original names (e.g. "V1", "C14", "D15", "M3") -- there are too many
    # (339 V-columns alone) to spell out as individual fields.
    C: dict[str, Optional[float]] = Field(default_factory=dict)
    D: dict[str, Optional[float]] = Field(default_factory=dict)
    V: dict[str, Optional[float]] = Field(default_factory=dict)
    M: dict[str, Optional[str]] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def raw_column(self, name: str):
        if name in C_COLS:
            return self.C.get(name)
        if name in D_COLS:
            return self.D.get(name)
        if name in V_COLS:
            return self.V.get(name)
        if name in M_COLS:
            return self.M.get(name)
        raise KeyError(f"unknown raw column: {name}")


class SHAPContribution(BaseModel):
    feature: str
    value: Union[float, str]
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
    val_pr_auc: float
    test_pr_auc: float
    calibration_method: str
    decision_threshold: float
    cost_false_negative: float
    cost_false_positive: float
