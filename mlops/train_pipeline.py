"""CLI entry-point to run a full training pass.

Usage:
    python -m mlops.train_pipeline --config configs/default.yaml
    python -m mlops.train_pipeline --seed 7 --run-name experiment-b
"""

from __future__ import annotations

import argparse
import sys

from mlops.config import Config, add_common_args, load_config
from mlops.data.generate import generate_customer_data, save_data
from mlops.logging_utils import setup_logging
from mlops.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the churn-prediction training pipeline.")
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg: Config = load_config(args)
    logger = setup_logging(log_file="run.log")
    logger.info("Config fingerprint: %s", cfg.fingerprint())

    df = generate_customer_data(cfg)
    train_path, test_path = save_data(df, cfg)
    summary = train(cfg, train_path, test_path)

    print("\n=== Training summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
