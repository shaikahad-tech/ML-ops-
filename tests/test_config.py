"""Unit tests for the config module."""

from mlops.config import Config


def test_default_config():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.data.n_samples > 0
    assert cfg.model.n_estimators > 0
    assert cfg.train.experiment_name


def test_fingerprint_stable():
    cfg1 = Config()
    cfg2 = Config()
    assert cfg1.fingerprint() == cfg2.fingerprint()


def test_fingerprint_changes_on_change():
    cfg1 = Config()
    cfg2 = Config()
    cfg2.model.n_estimators = 999
    assert cfg1.fingerprint() != cfg2.fingerprint()


def test_from_dict():
    raw = {"seed": 7, "model": {"n_estimators": 50}}
    cfg = Config.from_dict(raw)
    assert cfg.seed == 7
    assert cfg.model.n_estimators == 50
