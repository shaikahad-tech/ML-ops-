"""Smoke test for the FastAPI app using TestClient (no live model needed)."""

from fastapi.testclient import TestClient


def test_health_endpoint():
    from mlops.serving import app as app_module

    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_input_validation():
    """Missing required fields should be rejected with 422."""
    from mlops.serving import app as app_module

    client = TestClient(app_module.app)
    response = client.post("/predict", json={"tenure_months": 5})
    assert response.status_code == 422
