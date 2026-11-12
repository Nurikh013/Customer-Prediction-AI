
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app
from predict import predict_customer


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "customer_model.joblib"
)

client = TestClient(app)


def sample_customer():
    return {
        "age": 40,
        "job": "management",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1000,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "jan",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Saved model not available"
)
def test_real_model_prediction():
    customer = sample_customer()

    prediction, probability = predict_customer(customer)

    assert prediction in [0, 1]
    assert 0.0 <= probability <= 1.0

    assert prediction == int(probability >= 0.60)
    assert prediction == 0


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Saved model not available"
)
def test_real_api_prediction():
    customer = sample_customer()

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    prediction, probability = predict_customer(customer)

    assert data["prediction"] == prediction
    assert data["label"] == "no"
    assert abs(data["model_score"] - probability) < 0.0001
