"""
Adelon — Generate synthetic Marketing Mix Model (MMM) data with known ground truth.

Produces aggregate daily time-series data where channel spend drives revenue
through adstock (carryover) and saturation (diminishing returns) transformations.
Ground truth parameters are saved alongside the data so a Bayesian MMM can be
validated by checking posterior recovery against the true values.

Output files:
    mmm_daily_data.csv          — 1,095 rows of daily spend, controls, revenue
    mmm_contributions_truth.csv — per-channel daily contribution breakdown
    mmm_ground_truth.json       — exact parameter values used in generation
"""

import json
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


""" US Federal Holidays (2022-2024) + Black Friday / Cyber Monday """

US_HOLIDAYS = {
    # 2022
    datetime(2022, 1, 1),  # New Year's Day
    datetime(2022, 1, 17),  # MLK Day
    datetime(2022, 2, 21),  # Presidents' Day
    datetime(2022, 5, 30),  # Memorial Day
    datetime(2022, 7, 4),  # Independence Day
    datetime(2022, 9, 5),  # Labor Day
    datetime(2022, 11, 24),  # Thanksgiving
    datetime(2022, 11, 25),  # Black Friday
    datetime(2022, 11, 28),  # Cyber Monday
    datetime(2022, 12, 25),  # Christmas
    # 2023
    datetime(2023, 1, 1),
    datetime(2023, 1, 16),
    datetime(2023, 2, 20),
    datetime(2023, 5, 29),
    datetime(2023, 7, 4),
    datetime(2023, 9, 4),
    datetime(2023, 11, 23),
    datetime(2023, 11, 24),
    datetime(2023, 11, 27),
    datetime(2023, 12, 25),
    # 2024
    datetime(2024, 1, 1),
    datetime(2024, 1, 15),
    datetime(2024, 2, 19),
    datetime(2024, 5, 27),
    datetime(2024, 7, 4),
    datetime(2024, 9, 2),
    datetime(2024, 11, 28),
    datetime(2024, 11, 29),
    datetime(2024, 12, 2),
    datetime(2024, 12, 25),
}


class MMMDataGenerator:
    """Generate synthetic daily time-series data for Bayesian Marketing Mix Modeling."""

    CHANNELS = ["tv", "search", "social", "display", "print_ooh"]

    def __init__(self, n_days=1095, start_date="2022-01-01", seed=42):
        self.n_days = n_days
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.ground_truth = {
            "metadata": {
                "n_days": n_days,
                "start_date": start_date,
                "seed": seed,
                "channels": self.CHANNELS,
            },
            "baseline": {
                "intercept": 50_000.0,
                "trend_slope": 15.0,
                "noise_sigma": 2_500.0,
            },
            "seasonality": {
                "yearly_fourier_order": 3,
                "yearly_amplitudes": {
                    "sin_1": 4_000.0,
                    "cos_1": 2_000.0,
                    "sin_2": -1_500.0,
                    "cos_2": 500.0,
                    "sin_3": 800.0,
                    "cos_3": -300.0,
                },
                "day_of_week_effects": {
                    "Monday": 0.0,
                    "Tuesday": 500.0,
                    "Wednesday": 800.0,
                    "Thursday": 1_200.0,
                    "Friday": 2_000.0,
                    "Saturday": -3_000.0,
                    "Sunday": -4_500.0,
                },
            },
            "channels": {
                "tv": {
                    "adstock_alpha": 0.85,
                    "saturation_K": 150_000.0,
                    "saturation_S": 2.5,
                    "beta": 0.35,
                    "base_daily_spend": 25_000.0,
                    "spend_noise_scale": 0.20,
                },
                "search": {
                    "adstock_alpha": 0.20,
                    "saturation_K": 30_000.0,
                    "saturation_S": 3.0,
                    "beta": 0.55,
                    "base_daily_spend": 18_000.0,
                    "spend_noise_scale": 0.15,
                },
                "social": {
                    "adstock_alpha": 0.50,
                    "saturation_K": 45_000.0,
                    "saturation_S": 2.8,
                    "beta": 0.42,
                    "base_daily_spend": 12_000.0,
                    "spend_noise_scale": 0.25,
                },
                "display": {
                    "adstock_alpha": 0.40,
                    "saturation_K": 25_000.0,
                    "saturation_S": 2.2,
                    "beta": 0.28,
                    "base_daily_spend": 8_000.0,
                    "spend_noise_scale": 0.30,
                },
                "print_ooh": {
                    "adstock_alpha": 0.70,
                    "saturation_K": 60_000.0,
                    "saturation_S": 1.8,
                    "beta": 0.18,
                    "base_daily_spend": 5_000.0,
                    "spend_noise_scale": 0.10,
                },
            },
            "controls": {
                "price_index_effect": -8_000.0,
                "holiday_effect": 12_000.0,
                "promotion_effect": 18_000.0,
                "competitor_spend_effect": -0.08,
                "economic_index_effect": 6_000.0,
            },
            "spend_patterns": {
                "weekend_multiplier": 0.60,
                "q4_multiplier": 1.80,
                "q1_multiplier": 0.75,
                "campaign_burst_probability": 0.08,
                "campaign_burst_multiplier": 2.50,
                "inter_channel_correlation": 0.45,
            },
        }

    """ Private helpers """

    def _build_date_index(self):
        """Build a DataFrame of calendar features and Fourier seasonality terms."""
        dates = [self.start_date + timedelta(days=i) for i in range(self.n_days)]
        df = pd.DataFrame({"date": dates})
        df["t"] = np.arange(self.n_days)
        df["day_of_week"] = df["date"].apply(lambda d: d.weekday())
        df["day_name"] = df["date"].apply(lambda d: d.strftime("%A"))
        df["month"] = df["date"].apply(lambda d: d.month)
        df["quarter"] = df["date"].apply(lambda d: (d.month - 1) // 3 + 1)
        df["is_weekend"] = df["day_of_week"] >= 5
        df["is_us_holiday"] = df["date"].isin(US_HOLIDAYS)

        # Fourier terms for yearly seasonality
        order = self.ground_truth["seasonality"]["yearly_fourier_order"]
        t = df["t"].values
        for k in range(1, order + 1):
            df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
            df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / 365.25)

        return df

    @staticmethod
    def _apply_adstock(spend, alpha):
        """Geometric adstock: adstock[t] = spend[t] + alpha * adstock[t-1]."""
        adstock = np.zeros_like(spend, dtype=float)
        adstock[0] = spend[0]
        for t in range(1, len(spend)):
            adstock[t] = spend[t] + alpha * adstock[t - 1]
        return adstock

    @staticmethod
    def _apply_saturation(x, K, S):
        """Hill saturation function: x^S / (K^S + x^S), returns values in (0, 1)."""
        x_s = np.power(x, S)
        return x_s / (np.power(K, S) + x_s)

    """ Public generators """

    def generate_spend(self, date_df):
        """Generate daily channel spend with realistic patterns and correlated noise."""
        n = len(date_df)
        n_ch = len(self.CHANNELS)
        sp = self.ground_truth["spend_patterns"]

        # Correlated noise across channels
        corr = sp["inter_channel_correlation"]
        cov_matrix = np.full((n_ch, n_ch), corr) + np.eye(n_ch) * (1 - corr)
        noise = self.rng.multivariate_normal(np.zeros(n_ch), cov_matrix, size=n)

        # Campaign bursts (correlated across channels)
        burst_base = self.rng.random(n) < sp["campaign_burst_probability"]
        burst_matrix = np.zeros((n, n_ch), dtype=bool)
        for i in range(n_ch):
            # If any burst fires, 60% chance each other channel co-bursts
            burst_matrix[:, i] = burst_base | (self.rng.random(n) < 0.60) & burst_base
            # Also allow independent bursts
            burst_matrix[:, i] |= (
                self.rng.random(n) < sp["campaign_burst_probability"] * 0.3
            )

        spend_df = date_df[["date"]].copy()

        for i, channel in enumerate(self.CHANNELS):
            ch_params = self.ground_truth["channels"][channel]
            base = ch_params["base_daily_spend"]
            noise_scale = ch_params["spend_noise_scale"]

            # Seasonal multiplier per day
            seasonal_mult = np.ones(n)
            months = date_df["month"].values
            seasonal_mult[np.isin(months, [11, 12])] = sp["q4_multiplier"]
            seasonal_mult[np.isin(months, [1, 2, 3])] = sp["q1_multiplier"]

            # Weekend multiplier
            weekend_mult = np.where(
                date_df["is_weekend"].values, sp["weekend_multiplier"], 1.0
            )

            # Campaign burst multiplier
            burst_mult = np.where(
                burst_matrix[:, i], sp["campaign_burst_multiplier"], 1.0
            )

            # Combine: base * multipliers * lognormal noise
            raw_spend = (
                base
                * seasonal_mult
                * weekend_mult
                * burst_mult
                * np.exp(noise[:, i] * noise_scale)
            )

            # Print/OOH: smooth with 7-day rolling mean (weekly billing)
            if channel == "print_ooh":
                raw_spend = pd.Series(raw_spend).rolling(7, min_periods=1).mean().values

            spend_df[f"{channel}_spend"] = np.round(raw_spend, 2)

        return spend_df

    def generate_media_contributions(self, spend_df):
        """Apply adstock + saturation transforms to compute channel revenue contributions."""
        intercept = self.ground_truth["baseline"]["intercept"]
        contrib_df = spend_df[["date"]].copy()

        for channel in self.CHANNELS:
            ch = self.ground_truth["channels"][channel]
            raw_spend = spend_df[f"{channel}_spend"].values

            adstocked = self._apply_adstock(raw_spend, ch["adstock_alpha"])
            saturated = self._apply_saturation(
                adstocked, ch["saturation_K"], ch["saturation_S"]
            )
            contribution = ch["beta"] * saturated * intercept

            contrib_df[f"{channel}_adstocked"] = np.round(adstocked, 2)
            contrib_df[f"{channel}_saturated"] = np.round(saturated, 6)
            contrib_df[f"{channel}_contribution"] = np.round(contribution, 2)

        contrib_df["total_media_contribution"] = sum(
            contrib_df[f"{ch}_contribution"] for ch in self.CHANNELS
        )

        return contrib_df

    def generate_controls(self, date_df):
        """Generate non-media control variables."""
        n = len(date_df)
        control_df = date_df[["date"]].copy()

        # Price index: AR(1) mean-reverting to 1.0
        price = np.zeros(n)
        price[0] = 1.0
        for t in range(1, n):
            price[t] = 0.95 * price[t - 1] + 0.05 * 1.0 + self.rng.normal(0, 0.01)
        control_df["price_index"] = np.round(price, 4)

        # Promotions: ~6 bursts/year, 5-10 days each, near holidays
        is_promo = np.zeros(n, dtype=int)
        holiday_mask = date_df["is_us_holiday"].values
        # Elevated promo probability near holidays
        promo_prob = np.where(holiday_mask, 0.7, 0.0)
        # Also spread probability +-5 days around holidays
        for shift in range(-5, 6):
            shifted = np.roll(holiday_mask, shift)
            promo_prob = np.maximum(promo_prob, np.where(shifted, 0.4, 0.0))
        # Add some random bursts outside holidays (~6 per year total → ~18 over 3 years)
        promo_prob += 0.005
        promo_triggers = self.rng.random(n) < promo_prob
        # Extend each trigger to a 5-10 day burst
        for t in range(n):
            if promo_triggers[t] and is_promo[t] == 0:
                burst_len = self.rng.randint(5, 11)
                is_promo[t : min(t + burst_len, n)] = 1
        control_df["is_promotion"] = is_promo

        # Competitor spend: log-normal base $20K/day with seasonal peaks
        competitor_base = 20_000.0
        months = date_df["month"].values
        comp_seasonal = np.where(np.isin(months, [11, 12]), 1.5, 1.0)
        # Poisson spike events (~4/year = 12 over 3 years)
        n_spikes = self.rng.poisson(12)
        spike_days = self.rng.choice(n, size=min(n_spikes, n), replace=False)
        comp_spike = np.ones(n)
        for d in spike_days:
            duration = self.rng.randint(5, 15)
            comp_spike[d : min(d + duration, n)] = self.rng.uniform(2.0, 3.5)
        competitor_spend = (
            competitor_base
            * comp_seasonal
            * comp_spike
            * np.exp(self.rng.normal(0, 0.15, n))
        )
        control_df["competitor_spend"] = np.round(competitor_spend, 2)

        # Economic index: random walk with mean reversion, bounded [0.85, 1.15]
        econ = np.zeros(n)
        econ[0] = 1.0
        for t in range(1, n):
            drift = 0.001 * (1.0 - econ[t - 1])  # mean-revert to 1.0
            econ[t] = econ[t - 1] + drift + self.rng.normal(0, 0.003)
            econ[t] = np.clip(econ[t], 0.85, 1.15)
        control_df["economic_index"] = np.round(econ, 4)

        return control_df

    def generate_revenue(self, date_df, spend_df, contrib_df, control_df):
        """Assemble the revenue time series from all components."""
        gt = self.ground_truth
        baseline = gt["baseline"]
        seas = gt["seasonality"]
        ctrl = gt["controls"]
        n = len(date_df)

        # Intercept + trend
        intercept = baseline["intercept"]
        trend = baseline["trend_slope"] * date_df["t"].values

        # Fourier seasonality
        amps = seas["yearly_amplitudes"]
        fourier_effect = np.zeros(n)
        for k in range(1, seas["yearly_fourier_order"] + 1):
            fourier_effect += amps[f"sin_{k}"] * date_df[f"fourier_sin_{k}"].values
            fourier_effect += amps[f"cos_{k}"] * date_df[f"fourier_cos_{k}"].values

        # Day-of-week effects
        dow_effects = seas["day_of_week_effects"]
        dow_contribution = date_df["day_name"].map(dow_effects).values

        # Media contributions
        media = contrib_df["total_media_contribution"].values

        # Control effects
        control_effect = (
            ctrl["price_index_effect"] * (control_df["price_index"].values - 1.0)
            + ctrl["holiday_effect"] * date_df["is_us_holiday"].astype(float).values
            + ctrl["promotion_effect"] * control_df["is_promotion"].values
            + ctrl["competitor_spend_effect"] * control_df["competitor_spend"].values
            + ctrl["economic_index_effect"]
            * (control_df["economic_index"].values - 1.0)
        )

        # Noise
        noise = self.rng.normal(0, baseline["noise_sigma"], n)

        # Assemble
        revenue = (
            intercept
            + trend
            + fourier_effect
            + dow_contribution
            + media
            + control_effect
            + noise
        )
        revenue = np.maximum(revenue, 0)

        return pd.Series(np.round(revenue, 2), name="revenue")

    """ Orchestration """

    def generate_complete_dataset(self):
        """Generate all datasets and return (mmm_df, contributions_df, ground_truth)."""
        logger.info("Generating synthetic MMM time-series data...")
        end_date = self.start_date + timedelta(days=self.n_days - 1)
        logger.info(
            "Date range: %s to %s (%s days)",
            self.start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            f"{self.n_days:,}",
        )

        # Calendar index
        date_df = self._build_date_index()
        logger.info(
            "Built date index with %d Fourier orders",
            self.ground_truth["seasonality"]["yearly_fourier_order"],
        )

        # Spend
        logger.info("Generating channel spend patterns...")
        spend_df = self.generate_spend(date_df)
        for ch in self.CHANNELS:
            total = spend_df[f"{ch}_spend"].sum()
            logger.info("  %s: $%s total spend", ch, f"{total:>12,.0f}")

        # Media contributions
        logger.info("Applying adstock + saturation transforms...")
        contrib_df = self.generate_media_contributions(spend_df)
        total_media = contrib_df["total_media_contribution"].sum()
        logger.info("Total media contribution: $%s", f"{total_media:,.0f}")

        # Controls
        logger.info("Generating control variables...")
        control_df = self.generate_controls(date_df)
        n_promo_days = control_df["is_promotion"].sum()
        logger.info(
            "Promotion days: %d (%.1f%%)",
            n_promo_days,
            n_promo_days / self.n_days * 100,
        )

        # Revenue
        logger.info("Assembling revenue time series...")
        revenue = self.generate_revenue(date_df, spend_df, contrib_df, control_df)
        logger.info("Total revenue: $%s", f"{revenue.sum():,.0f}")
        logger.info(
            "Daily range: $%s - $%s",
            f"{revenue.min():,.0f}",
            f"{revenue.max():,.0f}",
        )
        logger.info("Daily mean: $%s", f"{revenue.mean():,.0f}")

        # Assemble final DataFrame
        mmm_df = date_df.drop(columns=["day_name"]).copy()
        for ch in self.CHANNELS:
            mmm_df[f"{ch}_spend"] = spend_df[f"{ch}_spend"]
        mmm_df["price_index"] = control_df["price_index"]
        mmm_df["is_promotion"] = control_df["is_promotion"]
        mmm_df["competitor_spend"] = control_df["competitor_spend"]
        mmm_df["economic_index"] = control_df["economic_index"]
        mmm_df["revenue"] = revenue.values

        return mmm_df, contrib_df, self.ground_truth

    def save_data(self, mmm_df, contrib_df, ground_truth, output_dir=None):
        """Save datasets and ground truth parameters.

        Args:
            mmm_df: Main MMM DataFrame.
            contrib_df: Contributions truth DataFrame.
            ground_truth: Ground truth parameter dict.
            output_dir: Directory to save files. Defaults to the
                directory containing this script (data/).
        """
        script_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
        os.makedirs(script_dir, exist_ok=True)

        # Compute derived summary statistics
        revenue = mmm_df["revenue"].values
        total_revenue = float(revenue.sum())

        total_spend_by_channel = {}
        for ch in self.CHANNELS:
            total_spend_by_channel[ch] = float(mmm_df[f"{ch}_spend"].sum())

        total_media_contrib = float(contrib_df["total_media_contribution"].sum())
        baseline_trend_contrib = float(
            ground_truth["baseline"]["intercept"] * self.n_days
            + ground_truth["baseline"]["trend_slope"] * np.arange(self.n_days).sum()
        )

        ground_truth["derived"] = {
            "total_revenue": total_revenue,
            "mean_daily_revenue": float(revenue.mean()),
            "total_spend_by_channel": total_spend_by_channel,
            "total_media_contribution": total_media_contrib,
            "revenue_decomposition_pct": {
                "baseline_trend": round(baseline_trend_contrib / total_revenue, 4),
                "media": round(total_media_contrib / total_revenue, 4),
                "other": round(
                    1
                    - baseline_trend_contrib / total_revenue
                    - total_media_contrib / total_revenue,
                    4,
                ),
            },
        }

        # Save main data CSV
        csv_path = os.path.join(script_dir, "mmm_daily_data.csv")
        mmm_df.to_csv(csv_path, index=False)
        logger.info("Saved daily data to: %s", csv_path)
        logger.info("Shape: %s", mmm_df.shape)
        logger.info("Size: ~%.1f KB", os.path.getsize(csv_path) / 1024)

        # Save contributions truth CSV
        contrib_path = os.path.join(script_dir, "mmm_contributions_truth.csv")
        contrib_df.to_csv(contrib_path, index=False)
        logger.info("Saved contributions truth to: %s", contrib_path)
        logger.info("Shape: %s", contrib_df.shape)

        # Save ground truth JSON
        json_path = os.path.join(script_dir, "mmm_ground_truth.json")
        with open(json_path, "w") as f:
            json.dump(ground_truth, f, indent=2, default=str)
        logger.info("Saved ground truth to: %s", json_path)

        # Summary
        logger.info("=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info("Days: %s", f"{self.n_days:,}")
        logger.info("Total revenue: $%s", f"{total_revenue:,.0f}")
        logger.info("Mean daily revenue: $%s", f"{revenue.mean():,.0f}")
        logger.info("Revenue decomposition:")
        for k, v in ground_truth["derived"]["revenue_decomposition_pct"].items():
            logger.info("  %s: %.1f%%", k, v * 100)
        logger.info("Total spend by channel:")
        for ch, spend in total_spend_by_channel.items():
            logger.info("  %s: $%s", ch, f"{spend:>12,.0f}")
        logger.info("=" * 60)
        logger.info("Data generation complete!")
        logger.info("=" * 60)


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    generator = MMMDataGenerator(n_days=1095, start_date="2022-01-01", seed=42)
    mmm_df, contrib_df, ground_truth = generator.generate_complete_dataset()
    generator.save_data(mmm_df, contrib_df, ground_truth)
    logger.info("Ready for Bayesian MMM modeling!")
