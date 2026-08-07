"""Extended tests for the training module: roc_auc fallback, registration, MLflow logging."""

from __future__ import annotations

import math
from pathlib import Path

from mlops.config import Config
from mlops.data.generate import generate_customer_data, save_data
from mlops.train import evaluate_model, train


def _tiny_cfg(tmp_path: Path, register: bool = False) -> Config:
    cfg = Config()
    cfg.data.n_samples = 200
    cfg.data.data_dir = str(tmp_path / "data")
    cfg.train.tracking_uri = f"file:{tmp_path}/mlruns"
    cfg.train.register_model = register
    cfg.train.experiment_name = "train-test"
    cfg.train.run_name = "pytest"
    cfg.model.n_estimators = 10
    return cfg


def test_train_logs_metrics_and_returns_summary(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    summary = train(cfg, train_path, test_path)

    assert summary["n_train"] > 0
    assert summary["n_test"] > 0
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert k in summary
        assert 0.0 <= summary[k] <= 1.0
    assert summary["experiment"] == "train-test"


def test_train_writes_metrics_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    train(cfg, train_path, test_path)
    assert (tmp_path / "metrics.json").exists()
    import json

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert "f1" in metrics


def test_train_registers_model(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path, register=True)
    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    summary = train(cfg, train_path, test_path)
    assert summary["model_name"] == "churn-classifier"

    # Verify the model is in the registry.
    import mlflow

    mlflow.set_tracking_uri(cfg.train.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='churn-classifier'")
    assert versions, "model was not registered"


def test_evaluate_model_all_zeros():
    """roc_auc should fall back to 0.5 when only one class is present (nan guard)."""
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.3, 0.4]
    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert metrics["roc_auc"] == 0.5
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_single_class_true():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    y_prob = [0.9, 0.8, 0.7]
    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert metrics["roc_auc"] == 0.5
    assert metrics["f1"] == 1.0


def test_evaluate_model_imperfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.2, 0.6, 0.9, 0.8]
    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.75
    assert 0 < metrics["f1"] < 1.0


def test_evaluate_model_metrics_are_floats():
    y_true = [0, 1]
    y_pred = [0, 1]
    y_prob = [0.3, 0.7]
    metrics = evaluate_model(y_true, y_pred, y_prob)
    for _k, v in metrics.items():
        assert isinstance(v, float)
        assert not math.isnan(v)


def test_train_deterministic_with_same_seed(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    s1 = train(cfg, train_path, test_path)
    s2 = train(cfg, train_path, test_path)
    assert abs(s1["f1"] - s2["f1"]) < 1e-6
    assert abs(s1["roc_auc"] - s2["roc_auc"]) < 1e-6
