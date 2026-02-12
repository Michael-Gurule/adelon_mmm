"""CLI entrypoint for model evaluation and artifact generation.

Usage:
    adelon-evaluate
    adelon-evaluate --ground-truth data/mmm_ground_truth.json
    adelon-evaluate --output-dir artifacts/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace | None = None) -> None:
    """Evaluate fitted model: compute metrics, contributions, ROAS.

    Produces artifacts in the output directory:
        - metrics.json: R-squared, MAPE, convergence diagnostics
        - contributions.csv: posterior mean channel contributions
        - roas.csv: ROAS per channel
        - parameter_recovery.csv: summary with ground truth comparison

    Args:
        args: Parsed CLI arguments. If None, parses from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Adelon: Evaluate fitted MMM and produce artifacts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Path to trace NetCDF (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/",
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth JSON for parameter recovery",
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

    import arviz as az

    from src.bayesian_mmm import BayesianMMM
    from src.preprocessing import load_config

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )
    config = load_config(str(config_path))

    # Resolve trace path
    trace_path = args.trace
    if trace_path is None:
        trace_path = str(
            Path(config["artifacts"]["traces_dir"])
            / config["artifacts"]["results_filename"]
        )
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {trace_path}. "
            "Run `adelon-train` first."
        )

    # Load data and model
    data_path = Path(config["data"]["path"])
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["date"])

    logger.info("Building model and loading trace from %s", trace_path)
    model = BayesianMMM(config)
    model.build_model(df)
    model.load_results(str(trace_path))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics
    logger.info("Computing posterior predictions and metrics...")
    preds = model.get_posterior_predictions()

    trace = model._trace
    var_names = [
        "adstock_alpha",
        "saturation_K",
        "saturation_S",
        "beta_media",
        "intercept",
        "trend_coef",
        "sigma",
    ]
    summary = az.summary(trace, var_names=var_names)

    metrics = {
        "r_squared": preds["r_squared"],
        "mape": preds["mape"],
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergences": int(
            trace.sample_stats["diverging"].values.sum()
        ),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "R²=%.4f, MAPE=%.2f%%, max_r_hat=%.4f, divergences=%d",
        metrics["r_squared"],
        metrics["mape"],
        metrics["max_r_hat"],
        metrics["n_divergences"],
    )

    # 2. Channel contributions
    logger.info("Computing channel contributions...")
    contrib_df = model.get_channel_contributions()
    contrib_path = output_dir / "contributions.csv"
    contrib_df.to_csv(contrib_path, index=False)
    logger.info("Saved contributions to %s", contrib_path)

    # 3. ROAS
    logger.info("Computing ROAS...")
    roas_df = model.get_roas()
    roas_path = output_dir / "roas.csv"
    roas_df.to_csv(roas_path, index=False)
    logger.info("Saved ROAS to %s", roas_path)

    # 4. Parameter recovery (optional, requires ground truth)
    gt = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            with open(gt_path) as f:
                gt = json.load(f)

    logger.info("Computing parameter summary...")
    summary_df = model.summary(ground_truth=gt)
    recovery_path = output_dir / "parameter_recovery.csv"
    summary_df.to_csv(recovery_path)
    logger.info("Saved parameter recovery to %s", recovery_path)

    logger.info(
        "Evaluation complete. Artifacts saved to %s", output_dir
    )


if __name__ == "__main__":
    main()
