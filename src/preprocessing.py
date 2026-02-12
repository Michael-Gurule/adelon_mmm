"""
Data preparation and media transformation utilities for Bayesian MMM.

All transform functions are standalone (no class required) so they can be
called from both the PyMC model graph and from plain NumPy code in notebooks.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# Media Transforms
def geometric_adstock(x, alpha):
    """
    Apply geometric adstock (carryover) transform.

    adstock[t] = x[t] + alpha * adstock[t-1]

    Args:
        x:     Daily spend array, shape (T,).
        alpha: Decay parameter in [0, 1]. 0 = no carryover; 1 = infinite memory.

    Returns:
        Adstocked array of same shape as x.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + alpha * adstocked[t - 1]
    return adstocked


def hill_saturation(x, K, S):
    """
    Apply Hill saturation function.

    f(x) = x^S / (K^S + x^S), result is in (0, 1).

    Args:
        x: Adstocked spend array (non-negative).
        K: Half-saturation point. At x=K, f(K) = 0.5.
        S: Shape (steepness) parameter. Higher S = sharper S-curve.

    Returns:
        Saturated values in (0, 1).
    """
    if np.any(np.asarray(K) <= 0):
        raise ValueError(f"K must be > 0, got {K}")
    if np.any(np.asarray(S) <= 0):
        raise ValueError(f"S must be > 0, got {S}")
    x_s = np.power(np.maximum(np.asarray(x, dtype=float), 0.0), S)
    K_s = np.power(K, S)
    return x_s / (K_s + x_s)


# Data Preparation
def prepare_mmm_data(df, config):
    """
    Validate columns, extract and scale arrays from the MMM DataFrame.

    Args:
        df:     Raw MMM DataFrame (mmm_daily_data.csv loaded as DataFrame).
        config: Loaded YAML config dict (from load_config()).

    Returns:
        Tuple of (spend_matrix, revenue, fourier_matrix, dow_matrix,
                  controls_matrix, t_index) where:
          spend_matrix:    shape (T, n_channels), raw daily spend
          revenue:         shape (T,)
          fourier_matrix:  shape (T, 2 * fourier_order)
          dow_matrix:      shape (T, 6) — one-hot DOW, Monday dropped as reference
          controls_matrix: shape (T, n_controls)
          t_index:         shape (T,), integer time index
    """
    data_cfg = config["data"]

    # Validate required columns
    required = (
        [data_cfg["date_col"], data_cfg["revenue_col"]]
        + data_cfg["spend_cols"]
        + data_cfg["control_cols"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Revenue
    revenue = df[data_cfg["revenue_col"]].values.astype(float)

    # Spend matrix (T, n_channels)
    spend_matrix = df[data_cfg["spend_cols"]].values.astype(float)

    # Fourier terms
    order = data_cfg["fourier_order"]
    fourier_cols = []
    for k in range(1, order + 1):
        fourier_cols += [f"fourier_sin_{k}", f"fourier_cos_{k}"]
    fourier_matrix = df[fourier_cols].values.astype(float)

    # Day-of-week one-hot, drop Monday (reference category)
    dow_dummies = pd.get_dummies(df["day_of_week"], drop_first=True).values.astype(
        float
    )

    # Controls
    controls_matrix = df[data_cfg["control_cols"]].values.astype(float)

    # Time index
    t_index = df["t"].values.astype(float)

    logger.info(
        "Data prepared: %d days, %d channels, %d controls",
        len(revenue),
        spend_matrix.shape[1],
        controls_matrix.shape[1],
    )

    return spend_matrix, revenue, fourier_matrix, dow_dummies, controls_matrix, t_index


def compute_roas(contribution, spend):

    if len(contribution) != len(spend):
        raise ValueError("contribution and spend must have the same length")
    total_spend = float(np.sum(spend))
    if total_spend == 0.0:
        return 0.0
    return float(np.sum(contribution)) / total_spend


def load_config(config_path="config/model_config.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)
