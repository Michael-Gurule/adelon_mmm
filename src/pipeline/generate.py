"""CLI entrypoint for synthetic data generation.

Usage:
    adelon-generate
    adelon-generate --n-days 365 --seed 0 --output-dir data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.logging_config import setup_logging


def main(args: argparse.Namespace | None = None) -> None:
    """Generate synthetic MMM data and save to disk.

    Args:
        args: Parsed CLI arguments. If None, parses from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Adelon: Generate synthetic MMM data"
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=1095,
        help="Number of days to generate (default: 1095)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Directory to save generated data",
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

    # Import after logging is configured
    from data.generate_data import MMMDataGenerator

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MMMDataGenerator(
        n_days=args.n_days,
        start_date=args.start_date,
        seed=args.seed,
    )
    mmm_df, contrib_df, ground_truth = (
        generator.generate_complete_dataset()
    )
    generator.save_data(mmm_df, contrib_df, ground_truth, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
