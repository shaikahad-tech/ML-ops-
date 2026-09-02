"""FastAPI model-serving application.

Loads the latest version of the registered model from the MLflow model
registry (or a local artifact path) and exposes ``/predict`` and ``/health``
endpoints. The same ``Pipeline`` used in training is served, so feature
preprocessing is identical between train and serve.

Also serves a Minimalist Monochrome web UI at ``/`` for interactive predictions.

Run:
    uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mlops.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Scores a single customer (or batch) for churn probability.",
    version="0.1.0",
)

# ── Serve the web UI ──────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the Minimalist Monochrome prediction UI."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Churn Prediction API</h1><p>Web UI not found. Use /docs for API.</p>",
        status_code=200,
    )


_MODEL = None


class CustomerFeatures(BaseModel):
    """One customer's features — order-independent (sent as JSON object)."""

    tenure_months: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0, le=20000)
    num_support_calls: int = Field(..., ge=0, le=50)
    num_products: int = Field(..., ge=1, le=20)
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_label: int
    model_version: str


def _load_model():
    """Load the model. Prefers the MLflow registry; falls back to a local path."""
    # Explicit override via env var MLOPS_MODEL_PATH for local dev / Docker.
    import os

    local_path = os.environ.get("MLOPS_MODEL_PATH")
    if local_path and Path(local_path).exists():
        return mlflow.sklearn.load_model(local_path), "local"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='churn-classifier'")
    if not versions:
        raise RuntimeError(
            "No registered model 'churn-classifier' found. "
            "Run training first or set MLOPS_MODEL_PATH."
        )
    latest = sorted(versions, key=lambda v: v.version, reverse=True)[0]
    return mlflow.sklearn.load_model(f"models:/churn-classifier/{latest.version}"), latest.version


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        model, version = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc

    row = {k: [getattr(features, k)] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    X = pd.DataFrame(row)
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_label=label,
        model_version=str(version),
    )


@app.post("/predict/batch")
def predict_batch(rows: list[CustomerFeatures]):
    """Score a batch of customers."""
    try:
        model, version = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc

    records = [r.model_dump() for r in rows]
    X = pd.DataFrame(records)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return {
        "predictions": [
            {"churn_probability": round(float(p), 4), "churn_label": int(p >= 0.5)} for p in proba
        ],
        "model_version": str(version),
    }
