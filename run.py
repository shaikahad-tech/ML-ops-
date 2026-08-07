#!/usr/bin/env python
"""Unified CLI entry-point.

Subcommands:
    data     Generate synthetic data and split into train/test CSVs.
    train    Run the full training pipeline (data + train + MLflow logging).
    serve    Start the FastAPI model server.

Examples:
    python run.py train --config configs/default.yaml
    python run.py data  --seed 7
    python run.py serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlops", description="Customer-churn MLOps CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    # data
    p_data = sub.add_parser("data", help="Generate and split synthetic data.")
    from mlops.config import add_common_args

    add_common_args(p_data)
    p_data.add_argument("--n-samples", type=int, default=None, help="Override n_samples.")
    p_data.add_argument("--output-dir", type=str, default=None, help="Override data_dir.")

    # train
    p_train = sub.add_parser("train", help="Run the training pipeline.")
    add_common_args(p_train)
    p_train.add_argument("--n-samples", type=int, default=None, help="Override n_samples.")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI model server.")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "data":
        from mlops.config import load_config
        from mlops.data.generate import generate_customer_data, save_data
        from mlops.logging_utils import setup_logging

        cfg = load_config(args)
        if getattr(args, "n_samples", None):
            cfg.data.n_samples = args.n_samples
        if getattr(args, "output_dir", None):
            cfg.data.data_dir = args.output_dir
        setup_logging(log_file="run.log")
        df = generate_customer_data(cfg)
        save_data(df, cfg)

    elif args.command == "train":
        from mlops.config import load_config
        from mlops.data.generate import generate_customer_data, save_data
        from mlops.logging_utils import setup_logging
        from mlops.train import train

        cfg = load_config(args)
        if getattr(args, "n_samples", None):
            cfg.data.n_samples = args.n_samples
        logger = setup_logging(log_file="run.log")
        logger.info("Config fingerprint: %s", cfg.fingerprint())

        df = generate_customer_data(cfg)
        train_path, test_path = save_data(df, cfg)
        summary = train(cfg, train_path, test_path)

        print("\n=== Training summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "mlops.serving.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
