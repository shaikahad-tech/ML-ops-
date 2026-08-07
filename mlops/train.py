"""Training pipeline: fit the model, evaluate, and log everything to MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops.config import Config
from mlops.data.preprocess import build_model, split_features_target
from mlops.logging_utils import get_logger

log = get_logger("train")


def evaluate_model(y_true, y_pred, y_prob) -> dict[str, float]:
    """Compute a standard classification-metric bundle."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def train(cfg: Config, train_path: str | Path, test_path: str | Path) -> dict:
    """Run one training pass and log it to MLflow. Returns a summary dict."""
    mlflow.set_tracking_uri(cfg.train.tracking_uri)
    mlflow.set_experiment(cfg.train.experiment_name)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    model = build_model(cfg)

    with mlflow.start_run(run_name=cfg.train.run_name) as run:
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "n_estimators": cfg.model.n_estimators,
                "max_depth": cfg.model.max_depth,
                "min_samples_split": cfg.model.min_samples_split,
                "min_samples_leaf": cfg.model.min_samples_leaf,
                "class_weight": cfg.model.class_weight,
                "seed": cfg.seed,
                "version": cfg.version,
                "config_fingerprint": cfg.fingerprint(),
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
            }
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        if cfg.train.register_model:
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.register_model(model_uri=model_uri, name=cfg.train.model_name)
                log.info("Registered model '%s'", cfg.train.model_name)
            except Exception as exc:  # MlflowException on duplicate; non-fatal.
                log.warning("Model registration skipped: %s", exc)

        # Save a local metrics snapshot for non-MLflow consumers (CI, etc.).
        Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = {
        "run_id": run.info.run_id,
        "model_name": cfg.train.model_name,
        "experiment": cfg.train.experiment_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        **metrics,
    }
    log.info(
        "Training complete | run_id=%s | f1=%.4f | roc_auc=%.4f",
        summary["run_id"],
        summary["f1"],
        summary["roc_auc"],
    )
    return summary
