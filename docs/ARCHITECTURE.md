# MLOps Platform Architecture

## System Overview
```
┌──────────┬──────────┬──────────┬──────────┬──────────────────┐
│  Data    │ Features  │ Training │ Serving  │  Monitoring       │
│  Layer   │  Layer    │  Layer   │  Layer   │  Layer            │
│ (Phase 3)│ (Phase 4) │(Phase 6) │(Phase 10)│ (Phase 11)        │
├──────────┴──────────┴──────────┴──────────┴──────────────────┤
│              Pipeline Orchestration (Phase 12)                │
├──────────────────────────────────────────────────────────────┤
│        CI/CD (Phase 14)  │  Infra (Phase 15)  │  K8s (20)    │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow
1. Data Generation → Synthetic customer data
2. Data Validation → Schema, quality, statistical checks
3. Feature Engineering → Derived features + preprocessing
4. Feature Store → Offline (parquet) + online (redis)
5. Training → Multi-model (RF/XGB/LGBM/Ensemble) + HPO
6. Experiment Tracking → MLflow with model comparison
7. Model Registry → Versioning with staging→production
8. Evaluation → Metrics, ROC/PR curves, fairness
9. Serving → FastAPI with A/B testing, canary, batching
10. Monitoring → Drift detection, prediction logging, alerts
11. Pipeline → DAG with retries and scheduled retraining

## Component Details
- **Config (P2):** Pydantic schemas, YAML+env+CLI, fingerprinting
- **Data (P3):** CSV/Parquet/SQL/S3 connectors + DVC versioning
- **Features (P4):** Registry, derived features, sklearn pipeline
- **Validation (P5):** Schema, quality metrics, null/dup detection
- **Training (P6):** RF/XGB/LGBM/Ensemble + Optuna HPO
- **Tracking (P7):** MLflow, model comparison, lineage
- **Evaluation (P8):** Metrics, ROC/PR, fairness/bias
- **Registry (P9):** Versioning, staging→production, approval
- **Serving (P10):** FastAPI, A/B, canary, batch, Prometheus
- **Monitoring (P11):** PSI drift, prediction log, alerts, health
- **Orchestration (P12):** DAG, retries, scheduled retraining
- **Feature Store (P13):** Offline/online, point-in-time, freshness
- **CI/CD (P14):** GitHub Actions, multi-env, security scan
- **Infra (P15):** Docker Compose, Terraform, Prometheus, Grafana
- **Security (P16):** JWT, API keys, audit, PII
- **Testing (P17):** Unit + integration, fixtures, coverage
- **Performance (P19):** ONNX, quantization, caching, async batch
- **Deployment (P20):** K8s, Helm, blue-green, release scripts
