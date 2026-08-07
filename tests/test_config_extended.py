"""Extended tests for the config module: YAML loading, CLI overrides, serialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mlops.config import Config, DataConfig, ModelConfig, TrainConfig, add_common_args, load_config


def test_from_yaml(tmp_path: Path):
    raw = {
        "seed": 7,
        "version": "test",
        "data": {"n_samples": 100, "test_size": 0.25, "data_dir": str(tmp_path)},
        "model": {"n_estimators": 10, "max_depth": 3},
        "train": {"experiment_name": "exp-x", "run_name": "run-x", "register_model": False},
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(cfg_file)
    assert cfg.seed == 7
    assert cfg.version == "test"
    assert cfg.data.n_samples == 100
    assert cfg.data.test_size == 0.25
    assert cfg.data.data_dir == str(tmp_path)
    assert cfg.model.n_estimators == 10
    assert cfg.model.max_depth == 3
    assert cfg.train.experiment_name == "exp-x"
    assert cfg.train.run_name == "run-x"
    assert cfg.train.register_model is False


def test_from_yaml_missing_file_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "nonexistent.yaml")


def test_from_dict_defaults_for_missing_sections():
    cfg = Config.from_dict({"seed": 1})
    assert cfg.seed == 1
    assert cfg.data == DataConfig()
    assert cfg.model == ModelConfig()
    assert cfg.train == TrainConfig()


def test_from_dict_empty():
    cfg = Config.from_dict({})
    assert cfg == Config()


def test_to_dict_roundtrips():
    cfg = Config()
    cfg.model.n_estimators = 42
    d = cfg.to_dict()
    assert d["model"]["n_estimators"] == 42
    assert d["data"]["n_samples"] == cfg.data.n_samples
    assert "train" in d


def test_load_config_no_args_returns_default():
    cfg = load_config(None)
    assert cfg == Config()


def test_load_config_with_config_path(tmp_path: Path):
    raw = {"seed": 99, "model": {"n_estimators": 5}}
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = argparse.Namespace(
        config=str(cfg_file), seed=None, experiment_name=None, run_name=None, tracking_uri=None
    )
    cfg = load_config(args)
    assert cfg.seed == 99
    assert cfg.model.n_estimators == 5


def test_load_config_cli_overrides_seed_propagates():
    args = argparse.Namespace(
        config=None, seed=123, experiment_name=None, run_name=None, tracking_uri=None
    )
    cfg = load_config(args)
    assert cfg.seed == 123
    assert cfg.data.random_state == 123
    assert cfg.model.random_state == 123


def test_load_config_cli_overrides_all():
    args = argparse.Namespace(
        config=None,
        seed=None,
        experiment_name="override-exp",
        run_name="override-run",
        tracking_uri="file:/tmp/x",
    )
    cfg = load_config(args)
    assert cfg.train.experiment_name == "override-exp"
    assert cfg.train.run_name == "override-run"
    assert cfg.train.tracking_uri == "file:/tmp/x"


def test_add_common_args_returns_parser():
    parser = argparse.ArgumentParser()
    result = add_common_args(parser)
    assert result is parser
    # All common flags should parse.
    ns = parser.parse_args(["--seed", "5", "--run-name", "r"])
    assert ns.seed == 5
    assert ns.run_name == "r"


def test_fingerprint_is_hex_string():
    fp = Config().fingerprint()
    assert isinstance(fp, str)
    assert len(fp) == 12
    int(fp, 16)  # valid hex
