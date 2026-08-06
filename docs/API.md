# MLOps Platform API Reference

## Base URL
http://localhost:8000

## Endpoints

### Health & Readiness
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |

### Predictions
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 1000) |
| POST | `/predict/ab` | A/B test prediction |

### Monitoring
| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus metrics |
| GET | `/model/info` | Model version info |

## POST /predict Request
```json
{"tenure_months": 3, "monthly_charges": 89.5, "total_charges": 250.0,
 "num_support_calls": 4, "num_products": 1,
 "contract_type": "Month-to-month", "internet_service": "Fiber optic",
 "payment_method": "Electronic check"}
```

## Response
```json
{"churn_probability": 0.8721, "churn_label": 1,
 "model_name": "churn-classifier", "model_version": "3", "latency_ms": 2.45}
```

## Error Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request |
| 422 | Invalid input |
| 429 | Rate limit exceeded |
| 503 | Model not loaded |
| 500 | Internal server error |
