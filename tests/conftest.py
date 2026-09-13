import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from fraud.data import get_engine


@pytest.fixture(scope="session")
def engine():
    eng = get_engine()
    try:
        with eng.connect():
            pass
    except OperationalError:
        pytest.skip("Postgres is unreachable")
    return eng


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])


class FakeCalibrator:
    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])


class FakeExplainer:
    def shap_values(self, X):
        return np.zeros((1, X.shape[1]))


FAKE_METADATA = {
    "threshold": 0.14,
    "model_version": "xgboost-v1",
    "calibration_method": "platt",
    "feature_columns": ["feat_a"],
    "category_maps": {},
    "medians": {},
    "cost_false_negative": 2000.0,
    "cost_false_positive": 150.0,
}


@pytest.fixture
def fake_model_state(monkeypatch):
    import fraud.api.main as main_module

    fake_state = {
        "model": FakeModel(),
        "calibrator": FakeCalibrator(),
        "explainer": FakeExplainer(),
        "engine": None,
        "metadata": dict(FAKE_METADATA),
    }
    monkeypatch.setattr(main_module, "_state", fake_state)
    return fake_state


@pytest.fixture
def test_client(monkeypatch, fake_model_state):
    import fraud.api.main as main_module

    def fake_build_live_feature_vector(raw_input, engine, category_maps, medians, feature_columns):
        return pd.DataFrame([{"feat_a": 1.0}])

    monkeypatch.setattr(main_module, "build_live_feature_vector", fake_build_live_feature_vector)

    return TestClient(main_module.app)
