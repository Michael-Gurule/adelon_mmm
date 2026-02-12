"""
Adelon — Bayesian Marketing Mix Model using PyMC.

The model decomposes revenue into:
  - Baseline: intercept + linear trend
  - Seasonality: Fourier terms (yearly) + day-of-week effects
  - Media: adstock -> saturation -> beta for each channel
  - Controls: price index, promotions, competitor spend, economic index
  - Noise: Normal likelihood

Adstock is implemented via pytensor.scan (recursive operation).
Saturation uses the Hill function (x^S / (K^S + x^S)).
"""

import logging
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan

from src.exceptions import DataValidationError, ModelNotFittedError
from src.preprocessing import (
    compute_roas,
    hill_saturation,
    prepare_mmm_data,
)

logger = logging.getLogger(__name__)


class BayesianMMM:
    """
    Bayesian Marketing Mix Model.

    Fits a full generative model of daily revenue using PyMC with NUTS sampling.
    Supports parameter recovery validation against known ground truth.

    Usage:
        config = load_config("config/model_config.yaml")
        df = pd.read_csv("data/mmm_daily_data.csv", parse_dates=["date"])
        model = BayesianMMM(config)
        model.build_model(df)
        model.fit()
        contributions = model.get_channel_contributions()
        model.save_results("traces/mmm_trace.netcdf")
    """

    def __init__(self, config):
        self.config = config
        self.channels = config["data"]["channels"]
        self.n_channels = len(self.channels)
        self._model = None
        self._trace = None
        self._df = None
        self._spend_matrix = None
        self._revenue = None

    def build_model(self, df):
        """
        Construct the PyMC model graph from the raw DataFrame.

        Args:
            df: Raw MMM DataFrame (output of pd.read_csv on mmm_daily_data.csv).

        Returns:
            The constructed pm.Model object.
        """
        spend_matrix, revenue, fourier_matrix, dow_matrix, controls_matrix, t_index = (
            prepare_mmm_data(df, self.config)
        )

        self._df = df
        self._spend_matrix = spend_matrix
        self._revenue = revenue

        n_fourier = fourier_matrix.shape[1]
        n_dow = dow_matrix.shape[1]
        n_controls = controls_matrix.shape[1]
        priors = self.config["model"]["priors"]
        ch_priors = priors["channels"]

        with pm.Model() as model:
            # Baseline priors
            intercept = pm.Normal(
                "intercept",
                mu=priors["intercept"]["mu"],
                sigma=priors["intercept"]["sigma"],
            )
            trend_coef = pm.Normal(
                "trend_coef",
                mu=priors["trend"]["mu"],
                sigma=priors["trend"]["sigma"],
            )
            sigma = pm.HalfNormal("sigma", sigma=priors["sigma_obs"]["sigma"])

            # Seasonality priors
            fourier_coeffs = pm.Normal(
                "fourier_coeffs",
                mu=priors["fourier_coeffs"]["mu"],
                sigma=priors["fourier_coeffs"]["sigma"],
                shape=n_fourier,
            )
            dow_betas = pm.Normal(
                "dow_betas",
                mu=priors["dow_effects"]["mu"],
                sigma=priors["dow_effects"]["sigma"],
                shape=n_dow,
            )

            # Control variable priors
            control_betas = pm.Normal(
                "control_betas",
                mu=priors["control_betas"]["mu"],
                sigma=priors["control_betas"]["sigma"],
                shape=n_controls,
            )

            # Media channel priors (per-channel from config)
            alpha_a = np.array(
                [ch_priors[c]["adstock_alpha"]["alpha"] for c in self.channels]
            )
            alpha_b = np.array(
                [ch_priors[c]["adstock_alpha"]["beta"] for c in self.channels]
            )
            K_sigma = np.array(
                [ch_priors[c]["saturation_K"]["sigma"] for c in self.channels]
            )
            S_sigma = np.array(
                [ch_priors[c]["saturation_S"]["sigma"] for c in self.channels]
            )
            beta_sigma = np.array(
                [ch_priors[c]["beta_media"]["sigma"] for c in self.channels]
            )

            adstock_alpha = pm.Beta(
                "adstock_alpha", alpha=alpha_a, beta=alpha_b, shape=self.n_channels
            )
            saturation_K = pm.HalfNormal(
                "saturation_K", sigma=K_sigma, shape=self.n_channels
            )
            saturation_S = pm.HalfNormal(
                "saturation_S", sigma=S_sigma, shape=self.n_channels
            )
            beta_media = pm.HalfNormal(
                "beta_media", sigma=beta_sigma, shape=self.n_channels
            )

            # Adstock transform (single vectorized scan)
            spend_tensor = pt.as_tensor_variable(
                spend_matrix.astype(float)
            )  # (T, n_channels)

            def adstock_step(spend_t, adstock_prev, alpha):
                return spend_t + alpha * adstock_prev

            adstocked_tensor, _ = scan(
                fn=adstock_step,
                sequences=[spend_tensor],
                outputs_info=[pt.zeros(self.n_channels)],
                non_sequences=[adstock_alpha],
            )  # (T, n_channels)

            # Hill saturation (vectorized across channels)
            x_s = pt.power(adstocked_tensor, saturation_S)
            K_s = pt.power(saturation_K, saturation_S)
            saturated_tensor = x_s / (K_s + x_s)  # (T, n_channels)

            # Media contribution: beta * saturated * intercept
            media_contribution = pt.sum(
                beta_media * saturated_tensor * intercept, axis=1
            )

            # Deterministic components
            baseline = intercept + trend_coef * pt.as_tensor_variable(t_index)
            seasonality = pt.dot(
                pt.as_tensor_variable(fourier_matrix), fourier_coeffs
            ) + pt.dot(pt.as_tensor_variable(dow_matrix), dow_betas)
            controls = pt.dot(pt.as_tensor_variable(controls_matrix), control_betas)

            mu = baseline + seasonality + media_contribution + controls

            # Store for post-processing
            pm.Deterministic("mu", mu)
            pm.Deterministic(
                "channel_contributions", beta_media * saturated_tensor * intercept
            )

            # Likelihood
            pm.Normal("revenue_obs", mu=mu, sigma=sigma, observed=revenue)

        self._model = model
        logger.info(
            "Model built: %d timesteps, %d channels", len(revenue), self.n_channels
        )
        return model

    # Fitting
    def fit(
        self,
        draws=None,
        tune=None,
        chains=None,
        target_accept=None,
        random_seed=None,
        nuts_sampler=None,
        progressbar=True,
    ):
        """
        Sample the posterior using NUTS.

        Args:
            draws:         Posterior draws per chain (default from config).
            tune:          Tuning steps per chain (default from config).
            chains:        Number of MCMC chains (default from config).
            target_accept: NUTS target acceptance rate (default from config).
            random_seed:   Random seed (default from config).
            nuts_sampler:  NUTS backend ("pymc", "nutpie", "numpyro").
                           Default from config, falls back to "pymc".
            progressbar:   Show progress bar.

        Returns:
            ArviZ InferenceData stored in self._trace.
        """
        if self._model is None:
            raise ModelNotFittedError("Call build_model(df) before fit().")

        mcmc_cfg = self.config["mcmc"]
        draws = draws or mcmc_cfg["draws"]
        tune = tune or mcmc_cfg["tune"]
        chains = chains or mcmc_cfg["chains"]
        target_accept = target_accept or mcmc_cfg["target_accept"]
        random_seed = random_seed or mcmc_cfg["random_seed"]
        nuts_sampler = nuts_sampler or mcmc_cfg.get("nuts_sampler", "pymc")

        logger.info("Using NUTS sampler: %s", nuts_sampler)

        with self._model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    nuts_sampler=nuts_sampler,
                    progressbar=progressbar,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                )

        logger.info("Sampling complete: %d draws x %d chains", draws, chains)
        return self._trace

    # Posterior predictive
    def predict(self):
        """
        Draw posterior predictive samples on training data.

        Returns:
            Posterior predictive array, shape (n_samples, T).
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() before predict().")
        with self._model:
            ppc = pm.sample_posterior_predictive(
                self._trace,
                var_names=["revenue_obs"],
                progressbar=False,
            )
        return ppc.posterior_predictive["revenue_obs"].values.reshape(
            -1, len(self._revenue)
        )

    def get_posterior_predictions(
        self,
    ) -> dict[str, np.ndarray | float]:
        """Compute posterior predictive statistics from the fitted model.

        Extracts the deterministic ``mu`` variable from the posterior
        and computes point estimates, credible intervals, and fit
        metrics (R-squared, MAPE).

        Returns:
            Dict with keys:
                predicted_mean:  Posterior mean of mu, shape (T,).
                predicted_lower: 3rd percentile of mu, shape (T,).
                predicted_upper: 97th percentile of mu, shape (T,).
                r_squared:       Coefficient of determination.
                mape:            Mean Absolute Percentage Error (%).
                residuals:       actual - predicted_mean, shape (T,).

        Raises:
            ModelNotFittedError: If no trace is available.
            DataValidationError: If build_model() has not been called.
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")
        if self._revenue is None:
            raise DataValidationError(
                "Call build_model(df) before get_posterior_predictions()."
            )

        mu_samples = self._trace.posterior["mu"].values.reshape(-1, len(self._revenue))
        pred_mean = mu_samples.mean(axis=0)
        pred_lower = np.percentile(mu_samples, 3, axis=0)
        pred_upper = np.percentile(mu_samples, 97, axis=0)

        residuals = self._revenue - pred_mean
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self._revenue - self._revenue.mean()) ** 2)
        r_squared = float(1 - ss_res / ss_tot)
        mape = float(np.mean(np.abs(residuals / self._revenue)) * 100)

        return {
            "predicted_mean": pred_mean,
            "predicted_lower": pred_lower,
            "predicted_upper": pred_upper,
            "r_squared": r_squared,
            "mape": mape,
            "residuals": residuals,
        }

    # Post-hoc analysis
    def get_channel_contributions(self):
        """
        Compute posterior mean per-channel revenue contributions.

        Returns:
            DataFrame with date and per-channel contribution columns.
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")

        contrib_samples = self._trace.posterior["channel_contributions"].values
        # Mean over chains and draws -> (T, n_channels)
        contrib_mean = contrib_samples.reshape(
            -1, len(self._revenue), self.n_channels
        ).mean(axis=0)

        result = pd.DataFrame(
            contrib_mean,
            columns=[f"{ch}_contribution" for ch in self.channels],
        )
        result.insert(0, "date", self._df["date"].values)
        result["total_media"] = contrib_mean.sum(axis=1)
        return result

    def get_response_curves(self, n_points=100, spend_range_multiplier=2.0):
        """
        Compute marginal spend-response curves per channel using posterior means.

        Args:
            n_points:              Spend grid points per channel.
            spend_range_multiplier: Upper bound = this * max observed spend.

        Returns:
            Dict mapping channel name -> DataFrame(spend, contribution_mean,
            contribution_lower, contribution_upper).
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")

        response_curves = {}
        posterior = self._trace.posterior

        for c_idx, channel in enumerate(self.channels):
            max_spend = float(self._spend_matrix[:, c_idx].max())
            spend_grid = np.linspace(0, spend_range_multiplier * max_spend, n_points)

            # Posterior samples
            K_samples = posterior["saturation_K"].values[:, :, c_idx].flatten()
            S_samples = posterior["saturation_S"].values[:, :, c_idx].flatten()
            beta_samples = posterior["beta_media"].values[:, :, c_idx].flatten()
            intercept_samples = posterior["intercept"].values.flatten()

            # Compute contribution at each spend level across posterior samples
            n_samples = len(K_samples)
            contribs = np.zeros((n_samples, n_points))
            for s_idx, spend_val in enumerate(spend_grid):
                sat = hill_saturation(
                    np.full(n_samples, spend_val), K_samples, S_samples
                )
                contribs[:, s_idx] = beta_samples * sat * intercept_samples

            response_curves[channel] = pd.DataFrame(
                {
                    "spend": spend_grid,
                    "contribution_mean": contribs.mean(axis=0),
                    "contribution_lower": np.percentile(contribs, 3, axis=0),
                    "contribution_upper": np.percentile(contribs, 97, axis=0),
                }
            )

        return response_curves

    def get_roas(self):
        """
        Compute posterior ROAS per channel.

        Returns:
            DataFrame with channel, roas_mean, total_spend, total_contribution.
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")

        contrib_df = self.get_channel_contributions()
        rows = []
        for channel in self.channels:
            spend_col = f"{channel}_spend"
            contrib_col = f"{channel}_contribution"
            total_spend = float(self._df[spend_col].sum())
            total_contrib = float(contrib_df[contrib_col].sum())
            roas = compute_roas(
                contrib_df[contrib_col].values, self._df[spend_col].values
            )
            rows.append(
                {
                    "channel": channel,
                    "roas_mean": roas,
                    "total_spend": total_spend,
                    "total_contribution": total_contrib,
                }
            )
        return pd.DataFrame(rows)

    def summary(self, ground_truth=None):
        """
        Return a formatted parameter recovery table.

        Args:
            ground_truth: Optional ground truth dict (mmm_ground_truth.json).
                          Adds truth columns and recovery error if provided.

        Returns:
            DataFrame with posterior mean, SD, 94% HDI per parameter.
        """
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")

        var_names = [
            "adstock_alpha",
            "saturation_K",
            "saturation_S",
            "beta_media",
            "intercept",
            "trend_coef",
            "sigma",
        ]
        summary_df = az.summary(self._trace, var_names=var_names, round_to=4)

        if ground_truth is not None:
            true_vals = {}
            for c_idx, ch in enumerate(self.channels):
                ch_gt = ground_truth["channels"][ch]
                true_vals[f"adstock_alpha[{c_idx}]"] = ch_gt["adstock_alpha"]
                true_vals[f"saturation_K[{c_idx}]"] = ch_gt["saturation_K"]
                true_vals[f"saturation_S[{c_idx}]"] = ch_gt["saturation_S"]
                true_vals[f"beta_media[{c_idx}]"] = ch_gt["beta"]
            true_vals["intercept"] = ground_truth["baseline"]["intercept"]
            true_vals["trend_coef"] = ground_truth["baseline"]["trend_slope"]
            true_vals["sigma"] = ground_truth["baseline"]["noise_sigma"]

            summary_df["true_value"] = [
                true_vals.get(idx, np.nan) for idx in summary_df.index
            ]
            mask = summary_df["true_value"].notna()
            summary_df.loc[mask, "recovery_error_pct"] = (
                (summary_df.loc[mask, "mean"] - summary_df.loc[mask, "true_value"])
                / summary_df.loc[mask, "true_value"]
                * 100
            ).round(1)

        return summary_df

    def save_results(self, path="traces/mmm_trace.netcdf"):
        """Save ArviZ InferenceData to NetCDF."""
        if self._trace is None:
            raise ModelNotFittedError("Call fit() or load_results() first.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._trace.to_netcdf(str(out))
        logger.info("Trace saved to %s", out)

    def load_results(self, path="traces/mmm_trace.netcdf"):
        """Load ArviZ InferenceData from NetCDF."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        self._trace = az.from_netcdf(str(path_obj))
        logger.info("Trace loaded from %s", path_obj)
        return self._trace
