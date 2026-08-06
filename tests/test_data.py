"""Unit tests for data generation and preprocessing."""

import pandas as pd

from mlops.config import Config
from mlops.data.generate import generate_customer_data
from mlops.data.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    split_features_target,
)


def test_generate_data_shape_and_columns():
    cfg = Config()
    cfg.data.n_samples = 200
    df = generate_customer_data(cfg)
    assert len(df) == 200
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in df.columns


def test_churn_is_binary():
    cfg = Config()
    cfg.data.n_samples = 200
    df = generate_customer_data(cfg)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_build_model_pipeline():
    cfg = Config()
    pipe = build_model(cfg)
    assert pipe.steps[0][0] == "preprocessor"
    assert pipe.steps[1][0] == "classifier"


def test_split_features_target():
    cfg = Config()
    cfg.data.n_samples = 200
    df = generate_customer_data(cfg)
    X, y = split_features_target(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "churn" not in X.columns
    assert len(X) == len(y)
