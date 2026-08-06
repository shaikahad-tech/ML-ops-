"""Configuration management.

Loads a YAML config file, overrides with CLI args, and returns a typed
``Config`` object. Everything in this project reads from a Config instance,
so runs are deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    n_samples: int = 5000
    test_size: float = 0.2
    random_state: int = 42
    churn_positive_rate: float = 0.265
    data_dir: str = "data"


@dataclass
class ModelConfig:
    name: str = "random_forest"
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    class_weight: str = "balanced_subsample"


@dataclass
class TrainConfig:
    experiment_name: str = "churn-prediction"
    run_name: str = "baseline"
    tracking_uri: str = "file:./mlruns"
    register_model: bool = True
    model_name: str = "churn-classifier"


@dataclass
class Config:
    seed: int = 42
    version: str = "v1"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        seed = raw.get("seed", 42)
        version = raw.get("version", "v1")
        data = DataConfig(**raw.get("data", {}))
        model = ModelConfig(**raw.get("model", {}))
        train = TrainConfig(**raw.get("train", {}))
        return cls(seed=seed, version=version, data=data, model=model, train=train)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the config, used for run naming and artifact paths."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def load_config(cli_args: argparse.Namespace | None = None) -> Config:
    """Load config from an optional YAML file, then apply CLI overrides."""
    config_path = getattr(cli_args, "config", None) if cli_args else None
    if config_path:
        cfg = Config.from_yaml(config_path)
    else:
        cfg = Config()

    if cli_args is not None:
        if getattr(cli_args, "seed", None) is not None:
            cfg.seed = cli_args.seed
            cfg.data.random_state = cli_args.seed
            cfg.model.random_state = cli_args.seed
        if getattr(cli_args, "experiment_name", None):
            cfg.train.experiment_name = cli_args.experiment_name
        if getattr(cli_args, "run_name", None):
            cfg.train.run_name = cli_args.run_name
        if getattr(cli_args, "tracking_uri", None):
            cfg.train.tracking_uri = cli_args.tracking_uri

    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI flags added to every subcommand."""
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the global random seed.")
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="MLflow experiment name."
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name.")
    parser.add_argument(
        "--tracking-uri", type=str, default=None, help="MLflow tracking URI."
    )
    return parser
