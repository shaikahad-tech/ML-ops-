# Customer Churn Prediction — MLOps Project

A complete, production-style MLOps project for predicting customer churn. It replaces the previous minimal pipeline with a full end-to-end system: **synthetic data generation → preprocessing → model training → experiment tracking → model registry → REST API serving → tests → CI/CD → Docker**.

The project is fully self-contained — it generates its own synthetic customer dataset, so there are no external data downloads and every run is reproducible.

---

## What it does

Predicts whether a telecom customer will churn (`1`) or stay (`0`) based on features like tenure, monthly charges, support calls, contract type, and payment method. A `RandomForestClassifier` wrapped in a scikit-learn `Pipeline` (preprocessing + model in one artifact) is trained, tracked with **MLflow**, registered to the **Model Registry**, and served via **FastAPI**.

---

## Project structure

```
.
├── configs/
│   ├── default.yaml          # Full-size training config
│   └── ci.yaml               # Lightweight config for CI / smoke tests
├── data/                     # Generated CSVs land here (gitignored)
├── mlops/
│   ├── __init__.py
│   ├── __main__.py           # python -m mlops ...
│   ├── config.py             # Dataclass config + YAML loading + CLI overrides
│   ├── logging_utils.py      # Structured logging
│   ├── train.py              # Training + evaluation + MLflow logging
│   ├── train_pipeline.py     # CLI entry-point: data → train
│   ├── data/
│   │   ├── generate.py       # Synthetic customer data generation
│   │   └── preprocess.py     # Feature engineering pipeline (ColumnTransformer)
│   └── serving/
│       └── app.py            # FastAPI prediction API
├── tests/                    # Unit + end-to-end tests (pytest)
├── .github/workflows/ci.yml  # GitHub Actions: test + lint on push/PR
├── Dockerfile                # Serve the API
├── Dockerfile.serving        # Multi-stage: train inside build, then serve
├── docker-compose.yaml       # Local dev: API + optional training container
├── Makefile                  # make train / make test / make serve
├── pyproject.toml            # Ruff lint + format config
├── .pre-commit-config.yaml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── run.py                    # Unified CLI: data / train / serve
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 2. Generate data
python run.py data --config configs/default.yaml

# 3. Train (logs to MLflow, registers the model)
python run.py train --config configs/default.yaml

# 4. Serve the model
python run.py serve --port 8000

# 5. Test
pytest -v
```

Or with Make:

```bash
make install data train test serve
```

---

## API usage

Once the server is running (`make serve`):

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 3,
    "monthly_charges": 89.5,
    "total_charges": 250.0,
    "num_support_calls": 4,
    "num_products": 1,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

Response:

```json
{
  "churn_probability": 0.8721,
  "churn_label": 1,
  "model_version": "1"
}
```

Batch predictions are available at `POST /predict/batch`.

---

## Configuration

All configuration is driven by YAML files in `configs/` and can be overridden via CLI flags. See `configs/default.yaml` for every available option.

```bash
python run.py train --seed 7 --run-name experiment-b
```

---

## Docker

```bash
# Build
docker build -t churn-api .

# Train inside a container (creates a registered model in ./mlruns)
docker run --rm -v "$PWD/mlruns:/app/mlruns" -v "$PWD/data:/app/data" churn-api \
  python run.py train --config configs/default.yaml

# Serve
docker run -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" churn-api
```

Multi-stage (trains during build, serves in the final image):

```bash
docker build -t churn-api -f Dockerfile.serving .
docker run -p 8000:8000 churn-api
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

- Installs dependencies across Python 3.11 and 3.12
- Runs the full test suite
- Runs a training smoke test with `configs/ci.yaml`
- Lints and format-checks with Ruff

---

## Experiment tracking

Training runs are logged to **MLflow** with a local file store (`./mlruns`). To view the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Each run logs:
- Model hyperparameters and config fingerprint
- Metrics: accuracy, precision, recall, F1, ROC-AUC
- The model artifact (sklearn pipeline)
- Registered model version in the Model Registry

---

## Design principles

- **Config-driven, not hardcoded** — every parameter lives in YAML
- **Reproducible** — fixed seeds, config fingerprinting, deterministic splits
- **Train/serve parity** — the same `Pipeline` artifact is used in training and serving
- **Structured logging** — consistent format across modules
- **Self-contained** — synthetic data generation means zero external dependencies
- **Tested** — unit tests for each module + an end-to-end smoke test
