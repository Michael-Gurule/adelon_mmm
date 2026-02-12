"""Tests for src/bayesian_mmm.py"""

import pytest

from src.bayesian_mmm import BayesianMMM
from src.exceptions import DataValidationError, ModelNotFittedError


class TestBayesianMMMBuild:
    def test_build_model_returns_pymc_model(self, small_mmm_df, small_config):
        """build_model should return a pm.Model."""
        import pymc as pm

        model = BayesianMMM(small_config)
        result = model.build_model(small_mmm_df)
        assert isinstance(result, pm.Model)

    def test_model_has_expected_variables(self, small_mmm_df, small_config):
        """Model should contain all core RVs."""
        model = BayesianMMM(small_config)
        pymc_model = model.build_model(small_mmm_df)
        var_names = {v.name for v in pymc_model.free_RVs}
        for expected in ["adstock_alpha", "saturation_K", "saturation_S",
                         "beta_media", "intercept", "sigma"]:
            assert any(expected in name for name in var_names), \
                f"Variable {expected} not found in model"

    def test_fit_before_build_raises(self, small_config):
        """fit() without build_model() should raise ModelNotFittedError."""
        model = BayesianMMM(small_config)
        with pytest.raises(ModelNotFittedError, match="build_model"):
            model.fit()

    def test_predict_before_fit_raises(self, small_mmm_df, small_config):
        """predict() without fit() should raise ModelNotFittedError."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        with pytest.raises(ModelNotFittedError, match="fit"):
            model.predict()

    def test_get_posterior_predictions_before_fit_raises(self, small_config):
        """get_posterior_predictions() without fit should raise."""
        model = BayesianMMM(small_config)
        with pytest.raises(ModelNotFittedError):
            model.get_posterior_predictions()

    def test_get_posterior_predictions_without_build_raises(self, small_config):
        """get_posterior_predictions() without build_model should raise."""
        model = BayesianMMM(small_config)
        model._trace = "fake"
        with pytest.raises(DataValidationError):
            model.get_posterior_predictions()


@pytest.mark.slow
class TestBayesianMMMSampling:
    """Full MCMC tests — run with: pytest -m slow"""

    def test_fit_returns_inference_data(self, small_mmm_df, small_config):
        """fit() should return ArviZ InferenceData."""
        import arviz as az

        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        trace = model.fit(progressbar=False)
        assert isinstance(trace, az.InferenceData)
        assert hasattr(trace, "posterior")

    def test_posterior_has_expected_variables(self, small_mmm_df, small_config):
        """Posterior should contain media parameter variables."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        for var in ["adstock_alpha", "saturation_K", "beta_media"]:
            assert var in model._trace.posterior.data_vars

    def test_channel_contributions_shape(self, small_mmm_df, small_config):
        """Channel contributions should have correct shape."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        contrib = model.get_channel_contributions()
        assert len(contrib) == len(small_mmm_df)
        for ch in ["tv", "search", "social", "display", "print_ooh"]:
            assert f"{ch}_contribution" in contrib.columns

    def test_roas_positive(self, small_mmm_df, small_config):
        """ROAS should be non-negative for all channels."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        roas_df = model.get_roas()
        assert (roas_df["roas_mean"] >= 0).all()

    def test_summary_with_ground_truth(self, small_mmm_df, small_config, ground_truth):
        """summary() with ground truth should add recovery error columns."""
        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        summary = model.summary(ground_truth=ground_truth)
        assert "true_value" in summary.columns
        assert "recovery_error_pct" in summary.columns

    def test_save_load_roundtrip(self, small_mmm_df, small_config, tmp_path):
        """Save and load should produce equivalent InferenceData."""
        import arviz as az

        model = BayesianMMM(small_config)
        model.build_model(small_mmm_df)
        model.fit(progressbar=False)
        save_path = str(tmp_path / "test_trace.netcdf")
        model.save_results(save_path)

        model2 = BayesianMMM(small_config)
        trace = model2.load_results(save_path)
        assert isinstance(trace, az.InferenceData)
