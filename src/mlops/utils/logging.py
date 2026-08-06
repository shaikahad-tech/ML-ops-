"""Structured logging."""
from __future__ import annotations
import logging, sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

def setup_logging(level: str = "INFO", log_file: str | None = None):
    global _configured
    if _configured and not log_file:
        return logging.getLogger("mlops")
    root = logging.getLogger("mlops")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(fh)
    _configured = True
    return root

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mlops.{name}")
