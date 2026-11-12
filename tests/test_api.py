
from fastapi.testclient import TestClient
import api


client = TestClient(api.app)


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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint(monkeypatch):
    def fake_prediction(customer):
        return 0, 0.3748

    monkeypatch.setattr(
        api, "predict_customer", fake_prediction
    )

    response = client.post(
        "/predict",
        json=sample_customer()
    )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": 0,
        "label": "no",
        "model_score": 0.3748
    }


def test_invalid_age():
    customer = sample_customer()
    customer["age"] = 12

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_invalid_job():
    customer = sample_customer()

    customer["job"] = "astronaut"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_missing_model_returns_503(monkeypatch):
    def missing_model(customer):
        raise FileNotFoundError("Model file is missing")

    monkeypatch.setattr(
        api, "predict_customer", missing_model
    )

    response = client.post(
        "/predict",
        json=sample_customer()
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Prediction model is unavailable."
    }


