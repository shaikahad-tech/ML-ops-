"""Extended tests for the serving app: predict, batch, error paths, model loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def trained_app(tmp_path: Path, monkeypatch):
    """Train a tiny model and point the app at it via MLOPS_MODEL_PATH."""
    from mlops.config import Config
    from mlops.data.generate import generate_customer_data, save_data
    from mlops.serving import app as app_module
    from mlops.train import train

    cfg = Config()
    cfg.data.n_samples = 200
    cfg.data.data_dir = str(tmp_path / "data")
    cfg.train.tracking_uri = f"file:{tmp_path}/mlruns"
    cfg.train.register_model = False
    cfg.train.experiment_name = "serve-test"
    cfg.train.run_name = "pytest"
    cfg.model.n_estimators = 10

    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    train(cfg, train_path, test_path)

    # Find the model artifact directory under mlruns.
    import mlflow

    mlflow.set_tracking_uri(cfg.train.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("serve-test")
    assert exp is not None, "experiment not created"
    runs = client.search_runs([exp.experiment_id])
    assert runs, "no runs created"
    artifact_uri = runs[0].info.artifact_uri
    # artifact_uri may be file:///...; convert to a path.
    if artifact_uri.startswith("file:"):
        artifact_uri = artifact_uri[len("file:") :]
    model_path = Path(artifact_uri) / "model"

    # Reset module-level cache and point at the local artifact.
    monkeypatch.setattr(app_module, "_MODEL", None)
    monkeypatch.setenv("MLOPS_MODEL_PATH", str(model_path))
    return app_module


def _valid_payload():
    return {
        "tenure_months": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0,
        "num_support_calls": 3,
        "num_products": 2,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }


def test_predict_success(trained_app):
    client = TestClient(trained_app.app)
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in (0, 1)
    assert "model_version" in body


def test_predict_validation_rejects_out_of_range():
    from mlops.serving import app as app_module

    client = TestClient(app_module.app)
    payload = _valid_payload()
    payload["tenure_months"] = 999  # max is 72
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_validation_rejects_unknown_enum():
    from mlops.serving import app as app_module

    client = TestClient(app_module.app)
    payload = _valid_payload()
    payload["contract_type"] = "Weekly"  # invalid literal
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_returns_503_when_model_missing(tmp_path: Path, monkeypatch):
    from mlops.serving import app as app_module

    # No model registered and no local path set.
    monkeypatch.setattr(app_module, "_MODEL", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/empty-mlruns")
    monkeypatch.delenv("MLOPS_MODEL_PATH", raising=False)
    client = TestClient(app_module.app)
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_predict_batch_success(trained_app):
    client = TestClient(trained_app.app)
    rows = [_valid_payload() for _ in range(3)]
    rows[1]["tenure_months"] = 60
    rows[2]["monthly_charges"] = 20.0
    resp = client.post("/predict/batch", json=rows)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert p["churn_label"] in (0, 1)
    assert "model_version" in body


def test_health_returns_ok():
    from mlops.serving import app as app_module

    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_cached_after_first_load(trained_app, monkeypatch):
    """The second /predict call should reuse the cached model (no reload)."""
    trained_app.get_model()  # prime cache
    cached = trained_app.get_model()
    # Calling again returns the same object.
    assert trained_app.get_model() is cached
