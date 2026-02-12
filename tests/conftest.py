"""Shared fixtures for marketing mix model tests."""

import copy
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_mmm_df():
    """Small synthetic MMM DataFrame (90 days) for fast testing."""
    np.random.seed(42)
    n = 90
    dates = pd.date_range("2022-01-01", periods=n)
    t = np.arange(n)

    df = pd.DataFrame({
        "date": dates,
        "t": t,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "quarter": dates.quarter,
        "is_weekend": dates.dayofweek >= 5,
        "is_us_holiday": False,
    })

    for k in range(1, 4):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / 365.25)

    for ch, base in [("tv", 25000), ("search", 18000), ("social", 12000),
                     ("display", 8000), ("print_ooh", 5000)]:
        df[f"{ch}_spend"] = np.random.lognormal(np.log(base), 0.2, n)

    df["price_index"] = 1.0 + np.random.normal(0, 0.01, n)
    df["is_promotion"] = np.random.binomial(1, 0.1, n)
    df["competitor_spend"] = np.random.lognormal(np.log(20000), 0.15, n)
    df["economic_index"] = np.clip(
        1.0 + np.random.normal(0, 0.003, n).cumsum(), 0.85, 1.15
    )

    df["revenue"] = np.clip(70000 + 15 * t + np.random.normal(0, 2500, n), 0, None)
    return df


@pytest.fixture(scope="session")
def config():
    """Load the real project config."""
    from src.preprocessing import load_config
    return load_config("config/model_config.yaml")


@pytest.fixture(scope="session")
def small_config(config):
    """Config copy with small MCMC settings for fast tests."""
    cfg = copy.deepcopy(config)
    cfg["mcmc"]["draws"] = 50
    cfg["mcmc"]["tune"] = 50
    cfg["mcmc"]["chains"] = 2
    return cfg


@pytest.fixture(scope="session")
def ground_truth():
    """Load ground truth JSON."""
    with open("data/mmm_ground_truth.json") as f:
        return json.load(f)
