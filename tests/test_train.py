"""End-to-end smoke test: generate → train → evaluate."""

from mlops.config import Config
from mlops.data.generate import generate_customer_data, save_data
from mlops.train import evaluate_model, train


def test_end_to_end_training(tmp_path):
    cfg = Config()
    # Keep it tiny for speed.
    cfg.data.n_samples = 300
    cfg.data.data_dir = str(tmp_path)
    cfg.train.tracking_uri = f"file:{tmp_path}/mlruns"
    cfg.train.register_model = False
    cfg.train.experiment_name = "smoke-test"
    cfg.train.run_name = "pytest"
    cfg.model.n_estimators = 20

    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    summary = train(cfg, train_path, test_path)

    assert "f1" in summary
    assert 0.0 <= summary["f1"] <= 1.0
    assert 0.0 <= summary["roc_auc"] <= 1.0
    assert summary["n_train"] > 0
    assert summary["n_test"] > 0


def test_evaluate_model_metrics():
    # Trivial case: perfect predictions.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.8]
    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
