"""Tests for logging utilities: setup, handlers, get_logger."""

from __future__ import annotations

import logging
from pathlib import Path

from mlops.logging_utils import get_logger, setup_logging


def test_setup_logging_returns_logger():
    logger = setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "mlops"


def test_setup_logging_level():
    logger = setup_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logging_clears_existing_handlers():
    logger = setup_logging()
    initial = len(logger.handlers)
    # Second call clears and re-adds — should not accumulate handlers.
    setup_logging()
    assert len(logger.handlers) == initial


def test_setup_logging_with_file(tmp_path: Path):
    log_file = tmp_path / "subdir" / "test.log"
    logger = setup_logging(log_file=str(log_file))
    assert log_file.parent.exists()
    logger.info("hello world")
    for h in logger.handlers:
        h.flush()
    assert log_file.exists()
    assert "hello world" in log_file.read_text()


def test_get_logger_namespaced():
    logger = get_logger("data")
    assert logger.name == "mlops.data"


def test_get_logger_is_child_of_mlops():
    logger = get_logger("train")
    assert logger.parent.name == "mlops"
