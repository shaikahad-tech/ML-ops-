# MLOps Platform Runbook

## Daily Operations

### Check System Health
```bash
python -m mlops.cli monitor health
```

### Check for Data Drift
```bash
python -m mlops.cli monitor drift --model-name churn-classifier
```

### List Registered Models
```bash
python -m mlops.cli registry list
```

## Incident Response

### Model Not Loading
1. Check MLflow: `mlflow ui --port 5000`
2. Verify model exists: `python -m mlops.cli registry versions --name churn-classifier`
3. Check API logs: `tail -f logs/audit.jsonl`
4. Retrain if needed: `python -m mlops.cli train --config configs/production.yaml`

### High Latency
1. Check Prometheus metrics: `curl localhost:8000/metrics`
2. Look at p99 latency histogram
3. Check if model needs retraining
4. Consider model quantization (Phase 19)

### Drift Detected
1. Check drift report: `python -m mlops.cli monitor drift`
2. Review feature drift scores
3. Trigger retraining: `python -m mlops.cli pipeline --config configs/production.yaml`
4. Compare new model vs production
5. Deploy if better: `python -m mlops.cli deploy --model-name churn-classifier --version latest --env staging`

### Pipeline Failure
1. Check pipeline logs
2. Identify failed step
3. Fix root cause
4. Re-run: `python -m mlops.cli pipeline --config configs/production.yaml`

## Scheduled Jobs
| Job | Schedule | Description |
|-----|----------|-------------|
| Drift check | Every hour | Check PSI against baseline |
| Retrain | Daily 2 AM | Full pipeline if drift detected |
| Health check | Every 5 min | Component health monitoring |

## Backup & Recovery
- MLflow DB: Backup `mlflow.db` daily
- Model artifacts: Stored in `mlruns/` or S3
- Data versions: Tracked in `data/.versions/manifest.json`
- Audit logs: Append-only in `logs/audit.jsonl`
