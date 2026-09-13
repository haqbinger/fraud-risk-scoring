import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from calibration.calibrate import load_winning_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = DATA_DIR / "winning_model.json"
CALIBRATOR_PATH = DATA_DIR / "platt_calibrator.joblib"
METADATA_PATH = DATA_DIR / "model_metadata.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CALIBRATOR_PATH.exists() and METADATA_PATH.exists()),
    reason="data/processed/ model artifacts are missing",
)


@pytest.fixture(scope="module")
def model():
    return load_winning_model()


@pytest.fixture(scope="module")
def calibrator():
    return joblib.load(CALIBRATOR_PATH)


@pytest.fixture(scope="module")
def metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def synthetic_rows(metadata):
    columns = metadata["feature_columns"]
    rng = np.random.default_rng(42)
    data = rng.random((5, len(columns)))
    return pd.DataFrame(data, columns=columns)


def test_winning_model_loads_without_error(model):
    assert model is not None


def test_predict_proba_returns_valid_shape(model, synthetic_rows):
    proba = model.predict_proba(synthetic_rows)
    assert proba.shape == (5, 2)


def test_predict_proba_values_in_0_1_range(model, synthetic_rows):
    proba = model.predict_proba(synthetic_rows)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_platt_calibrator_loads_without_error(calibrator):
    assert calibrator is not None


def test_platt_calibrator_output_in_0_1_range(calibrator):
    raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    proba = calibrator.predict_proba(raw)[:, 1]
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_platt_calibrator_is_monotonic(calibrator):
    raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    proba = calibrator.predict_proba(raw)[:, 1]
    assert np.all(np.diff(proba) >= 0)


def test_model_metadata_has_required_keys(metadata):
    required = {"threshold", "feature_columns", "category_maps", "medians", "model_version"}
    assert required <= set(metadata.keys())


def test_threshold_in_valid_range(metadata):
    assert 0 < metadata["threshold"] < 1
