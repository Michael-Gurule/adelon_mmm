"""CLI entrypoint for full pipeline orchestration.

Runs: generate -> train -> evaluate in sequence.

Usage:
    adelon-run
    adelon-run --skip-generate
    adelon-run --config config/model_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the full MMM pipeline: generate -> train -> evaluate.

    Args:
        args: Parsed CLI arguments. If None, parses from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Adelon: Run full MMM pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation step",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training step",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/mmm_ground_truth.json",
        help="Path to ground truth JSON for evaluation",
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

    from src.pipeline.evaluate import main as evaluate_main
    from src.pipeline.generate import main as generate_main
    from src.pipeline.train import main as train_main

    # Step 1: Generate data
    if not args.skip_generate:
        logger.info("=" * 60)
        logger.info("STEP 1/3: Generating synthetic data")
        logger.info("=" * 60)
        generate_args = argparse.Namespace(
            n_days=1095,
            start_date="2022-01-01",
            seed=42,
            output_dir="data/",
            log_level=args.log_level,
            log_file=args.log_file,
        )
        generate_main(generate_args)
    else:
        logger.info("Skipping data generation (--skip-generate)")

    # Step 2: Train model
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("STEP 2/3: Training Bayesian MMM")
        logger.info("=" * 60)
        train_args = argparse.Namespace(
            config=args.config,
            output=None,
            draws=None,
            tune=None,
            chains=None,
            log_level=args.log_level,
            log_file=args.log_file,
        )
        train_main(train_args)
    else:
        logger.info("Skipping model training (--skip-train)")

    # Step 3: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 3/3: Evaluating model")
    logger.info("=" * 60)
    evaluate_args = argparse.Namespace(
        config=args.config,
        trace=None,
        output_dir="artifacts/",
        ground_truth=args.ground_truth,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    evaluate_main(evaluate_args)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
