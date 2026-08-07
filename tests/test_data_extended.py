"""Extended tests for data generation: persistence, calibration, edge cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops.config import Config
from mlops.data.generate import generate_customer_data, save_data


def test_save_data_creates_csvs(tmp_path: Path):
    cfg = Config()
    cfg.data.n_samples = 200
    cfg.data.data_dir = str(tmp_path)
    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)

    assert train_path.exists()
    assert test_path.exists()
    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # 80/20 split.
    assert len(train_df) + len(test_df) == 200
    assert abs(len(test_df) / 200 - 0.2) < 0.05


def test_save_data_preserves_columns(tmp_path: Path):
    cfg = Config()
    cfg.data.n_samples = 100
    cfg.data.data_dir = str(tmp_path)
    df = generate_customer_data(cfg)
    train_path, _ = save_data(df, cfg)
    out = pd.read_csv(train_path)
    for col in ["customer_id", "tenure_months", "churn"]:
        assert col in out.columns


def test_churn_is_valid_binary_rate():
    """Churn should be binary with a reasonable positive rate."""
    cfg = Config()
    cfg.data.n_samples = 5000
    df = generate_customer_data(cfg)
    rate = df["churn"].mean()
    assert 0.0 < rate < 1.0  # not degenerate


def test_generate_data_deterministic_with_seed():
    cfg1 = Config()
    cfg1.data.n_samples = 100
    cfg1.seed = 42
    cfg2 = Config()
    cfg2.data.n_samples = 100
    cfg2.seed = 42
    df1 = generate_customer_data(cfg1)
    df2 = generate_customer_data(cfg2)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_data_different_seeds_differ():
    cfg1 = Config()
    cfg1.data.n_samples = 100
    cfg1.seed = 1
    cfg2 = Config()
    cfg2.data.n_samples = 100
    cfg2.seed = 999
    df1 = generate_customer_data(cfg1)
    df2 = generate_customer_data(cfg2)
    assert not df1.equals(df2)


def test_total_charges_is_numeric():
    cfg = Config()
    cfg.data.n_samples = 200
    df = generate_customer_data(cfg)
    assert pd.api.types.is_numeric_dtype(df["total_charges"])


def test_contract_type_values():
    cfg = Config()
    cfg.data.n_samples = 500
    df = generate_customer_data(cfg)
    assert set(df["contract_type"].unique()).issubset({"Month-to-month", "One year", "Two year"})


def test_internet_service_values():
    cfg = Config()
    cfg.data.n_samples = 500
    df = generate_customer_data(cfg)
    assert set(df["internet_service"].unique()).issubset({"DSL", "Fiber optic", "No"})


def test_payment_method_values():
    cfg = Config()
    cfg.data.n_samples = 500
    df = generate_customer_data(cfg)
    assert set(df["payment_method"].unique()).issubset(
        {"Electronic check", "Mailed check", "Bank transfer", "Credit card"}
    )


def test_customer_ids_unique():
    cfg = Config()
    cfg.data.n_samples = 300
    df = generate_customer_data(cfg)
    assert df["customer_id"].is_unique


def test_tenure_within_range():
    cfg = Config()
    cfg.data.n_samples = 500
    df = generate_customer_data(cfg)
    assert df["tenure_months"].min() >= 1
    assert df["tenure_months"].max() <= 72


def test_num_products_within_range():
    cfg = Config()
    cfg.data.n_samples = 500
    df = generate_customer_data(cfg)
    assert df["num_products"].min() >= 1
    assert df["num_products"].max() <= 4


def test_data_dir_created_if_missing(tmp_path: Path):
    cfg = Config()
    cfg.data.n_samples = 50
    data_dir = tmp_path / "nested" / "data"
    cfg.data.data_dir = str(data_dir)
    df = generate_customer_data(cfg)
    save_data(df, cfg)
    assert data_dir.exists()
