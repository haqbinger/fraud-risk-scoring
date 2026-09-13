VALID_PAYLOAD = {"card1": 13926, "transaction_amt": 250.0}


def test_health_returns_200(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded_true(test_client):
    response = test_client.get("/health")
    assert response.json()["model_loaded"] is True


def test_health_db_connected_false(test_client):
    response = test_client.get("/health")
    assert response.json()["db_connected"] is False


def test_model_info_returns_200(test_client):
    response = test_client.get("/model")
    assert response.status_code == 200


def test_model_info_has_expected_fields(test_client):
    body = test_client.get("/model").json()
    for field in ("model_version", "val_pr_auc", "decision_threshold"):
        assert field in body


def test_predict_valid_request_returns_200(test_client):
    response = test_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_decision_is_allow_or_block(test_client):
    body = test_client.post("/predict", json=VALID_PAYLOAD).json()
    assert body["decision"] in ("ALLOW", "BLOCK")


def test_predict_probability_in_0_1_range(test_client):
    body = test_client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["fraud_probability_raw"] <= 1.0
    assert 0.0 <= body["fraud_probability_calibrated"] <= 1.0


def test_predict_rejects_negative_amount(test_client):
    payload = {**VALID_PAYLOAD, "transaction_amt": -10}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_zero_amount(test_client):
    payload = {**VALID_PAYLOAD, "transaction_amt": 0}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_card1(test_client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "card1"}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type_for_card1(test_client):
    payload = {**VALID_PAYLOAD, "card1": "not_a_number"}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_503_when_model_not_loaded(test_client, fake_model_state):
    fake_model_state["model"] = None
    response = test_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_returns_top_features_list(test_client):
    body = test_client.post("/predict", json=VALID_PAYLOAD).json()
    assert "top_features" in body
    assert len(body["top_features"]) >= 0
