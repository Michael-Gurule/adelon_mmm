"""CLI entrypoint for Bayesian MMM training.

Usage:
    adelon-train
    adelon-train --config config/model_config.yaml --draws 2000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace | None = None) -> None:
    """Build and fit the Bayesian MMM, save trace to disk.

    Args:
        args: Parsed CLI arguments. If None, parses from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Adelon: Train Bayesian MMM via MCMC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trace (default: from config)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Posterior draws per chain (default: from config)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=None,
        help="Tuning steps per chain (default: from config)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Number of MCMC chains (default: from config)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["pymc", "nutpie", "numpyro"],
        help="NUTS sampler backend (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to log file",
    )
    args = args or parser.parse_args()

    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(level=args.log_level, log_file=log_file)

    from src.bayesian_mmm import BayesianMMM
    from src.preprocessing import load_config

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    config = load_config(str(config_path))
    data_path = Path(config["data"]["path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Run `adelon-generate` first."
        )

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["date"])

    logger.info(
        "Building model: %d days, %d channels",
        len(df),
        len(config["data"]["channels"]),
    )
    model = BayesianMMM(config)
    model.build_model(df)

    logger.info("Starting MCMC sampling...")
    model.fit(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        nuts_sampler=args.sampler,
    )

    output_path = args.output
    if output_path is None:
        output_path = str(
            Path(config["artifacts"]["traces_dir"])
            / config["artifacts"]["results_filename"]
        )

    model.save_results(output_path)
    logger.info("Training complete. Trace saved to %s", output_path)


if __name__ == "__main__":
    main()
