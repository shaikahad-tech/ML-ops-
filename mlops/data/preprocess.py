"""Feature engineering and preprocessing pipeline.

Builds a scikit-learn ``ColumnTransformer`` that handles numeric scaling and
categorical one-hot encoding, then wraps it with the estimator into a single
``Pipeline``. The same pipeline is fit on training data and reused for serving,
which guarantees train/serve feature parity.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops.config import Config

NUMERIC_FEATURES: list[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "num_products",
]

CATEGORICAL_FEATURES: list[str] = [
    "contract_type",
    "internet_service",
    "payment_method",
]

TARGET = "churn"


def build_preprocessor() -> ColumnTransformer:
    """Return the feature-processing ``ColumnTransformer``."""
    numeric_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model(cfg: Config) -> Pipeline:
    """Build the full training pipeline: preprocessor + estimator."""
    model_cfg = cfg.model
    estimator = RandomForestClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        min_samples_split=model_cfg.min_samples_split,
        min_samples_leaf=model_cfg.min_samples_leaf,
        random_state=model_cfg.random_state,
        class_weight=model_cfg.class_weight,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def split_features_target(df: pd.DataFrame):
    """Separate the feature matrix from the target column."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
