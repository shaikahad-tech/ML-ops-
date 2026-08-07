"""Synthetic data generation for the customer-churn problem.

We generate a realistic-ish customer table (tenure, monthly charges, support
calls, contract type, etc.) with a churn label driven by a logistic function of
those features, so a model can actually learn the signal. No external data
downloads are required — the project is fully self-contained and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlops.config import Config
from mlops.logging_utils import get_logger

log = get_logger("data.generate")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_customer_data(cfg: Config) -> pd.DataFrame:
    """Generate a synthetic customer dataset with a churn target."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.data.n_samples

    tenure_months = rng.integers(1, 73, size=n)
    monthly_charges = np.round(rng.normal(65, 30, size=n), 2)
    total_charges = np.round(tenure_months * monthly_charges * rng.uniform(0.85, 1.15, n), 2)
    num_support_calls = rng.integers(0, 8, size=n)
    num_products = rng.integers(1, 5, size=n)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.52, 0.14])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Logistic churn signal: short tenure, high charges, many support calls → churn.
    linear = (
        -0.04 * tenure_months
        + 0.015 * monthly_charges
        + 0.55 * num_support_calls
        - 0.30 * num_products
        + 1.6 * (contract_type == "Month-to-month")
        - 1.2 * (contract_type == "Two year")
        + 0.5 * (internet_service == "Fiber optic")
        + 0.4 * (payment_method == "Electronic check")
        + rng.normal(0, 0.6, n)
    )
    prob = _sigmoid(linear)
    target_churn = (rng.uniform(0, 1, n) < prob).astype(int)

    # Calibrate the positive rate toward the configured target.
    current_rate = target_churn.mean()
    target_rate = cfg.data.churn_positive_rate
    if abs(current_rate - target_rate) > 0.02:
        extra_pos_needed = max(0, int((target_rate - current_rate) * n))
        negatives = np.where(target_churn == 0)[0]
        if extra_pos_needed and len(negatives) > 0:
            flip = rng.choice(negatives, size=min(extra_pos_needed, len(negatives)), replace=False)
            target_churn[flip] = 1

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "num_products": num_products,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "churn": target_churn,
        }
    )
    log.info("Generated %d rows | churn rate=%.3f", len(df), df["churn"].mean())
    return df


def save_data(df: pd.DataFrame, cfg: Config) -> tuple[Path, Path]:
    """Split into train/test CSVs and persist under ``data_dir``."""
    from sklearn.model_selection import train_test_split

    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=cfg.data.test_size, random_state=cfg.data.random_state, stratify=df["churn"]
    )
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    log.info("Saved train=%d rows -> %s", len(train_df), train_path)
    log.info("Saved test=%d rows  -> %s", len(test_df), test_path)
    return train_path, test_path
