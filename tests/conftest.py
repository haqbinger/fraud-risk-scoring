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
    """Stands in for CalibratedLGBMPipeline."""

    categories_map = {}

    def predict_proba_raw(self, X):
        return np.array([0.3])

    def predict_proba(self, X):
        return np.array([0.3])


class FakeExplainer:
    def shap_values(self, X):
        return np.zeros((1, X.shape[1]))


FAKE_METADATA = {
    "threshold": 0.14,
    "model_type": "LightGBM",
    "calibrator_type": "platt",
    "selected_features": ["feat_a"],
    "categorical_features": [],
    "val_pr_auc": 0.67,
    "test_pr_auc": 0.59,
    "cost_model": {
        "cost_false_negative": 2000.0,
        "cost_false_positive": 150.0,
    },
}


@pytest.fixture
def fake_model_state(monkeypatch):
    import fraud.api.main as main_module

    fake_state = {
        "model": FakeModel(),
        "explainer": FakeExplainer(),
        "engine": None,
        "metadata": dict(FAKE_METADATA),
    }
    monkeypatch.setattr(main_module, "_state", fake_state)
    return fake_state


@pytest.fixture
def test_client(monkeypatch, fake_model_state):
    import fraud.api.main as main_module

    def fake_build_live_feature_vector(raw_input, engine, selected_features, cat_cols, categories_map):
        return pd.DataFrame([{"feat_a": 1.0}])

    monkeypatch.setattr(main_module, "build_live_feature_vector", fake_build_live_feature_vector)

    return TestClient(main_module.app)
