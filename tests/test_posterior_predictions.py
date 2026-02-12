"""Tests for BayesianMMM.get_posterior_predictions() (requires MCMC)."""

import numpy as np
import pytest

from src.bayesian_mmm import BayesianMMM


@pytest.mark.slow
class TestGetPosteriorPredictions:
    """Full MCMC tests for get_posterior_predictions()."""

    @pytest.fixture(scope="class")
    def fitted_model(self, small_mmm_df, small_config):
        """Fit a small model once and share across tests in this class."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        return model

    def test_returns_dict_with_expected_keys(self, fitted_model):
        """Return dict should contain all documented keys."""
        preds = fitted_model.get_posterior_predictions()
        expected_keys = {
            "predicted_mean",
            "predicted_lower",
            "predicted_upper",
            "r_squared",
            "mape",
            "residuals",
        }
        assert set(preds.keys()) == expected_keys

    def test_predicted_arrays_correct_shape(
        self, fitted_model, small_mmm_df
    ):
        """Predicted arrays should match the number of timesteps."""
        preds = fitted_model.get_posterior_predictions()
        n = len(small_mmm_df)
        assert preds["predicted_mean"].shape == (n,)
        assert preds["predicted_lower"].shape == (n,)
        assert preds["predicted_upper"].shape == (n,)
        assert preds["residuals"].shape == (n,)

    def test_credible_interval_ordering(self, fitted_model):
        """Lower bound should be <= mean <= upper bound (on average)."""
        preds = fitted_model.get_posterior_predictions()
        assert np.all(preds["predicted_lower"] <= preds["predicted_upper"])
        # Mean should generally be between bounds
        mean_in_bounds = np.mean(
            (preds["predicted_mean"] >= preds["predicted_lower"])
            & (preds["predicted_mean"] <= preds["predicted_upper"])
        )
        assert mean_in_bounds > 0.9

    def test_r_squared_is_scalar(self, fitted_model):
        """R-squared should be a float."""
        preds = fitted_model.get_posterior_predictions()
        assert isinstance(preds["r_squared"], float)

    def test_mape_is_non_negative(self, fitted_model):
        """MAPE should be non-negative."""
        preds = fitted_model.get_posterior_predictions()
        assert preds["mape"] >= 0

    def test_residuals_sum_near_zero(self, fitted_model):
        """Residuals should roughly center around zero."""
        preds = fitted_model.get_posterior_predictions()
        mean_residual = np.mean(preds["residuals"])
        mean_revenue = np.mean(preds["predicted_mean"])
        # Residual mean should be small relative to revenue
        assert abs(mean_residual / mean_revenue) < 0.5
