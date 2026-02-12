"""Tests for data/generate_data.py"""

import pytest

from data.generate_data import MMMDataGenerator


class TestMMMDataGenerator:
    @pytest.fixture
    def small_generator(self):
        return MMMDataGenerator(n_days=90, start_date="2022-01-01", seed=0)

    def test_dataset_row_count(self, small_generator):
        """Generated DataFrame should have n_days rows."""
        df, _, _ = small_generator.generate_complete_dataset()
        assert len(df) == 90

    def test_expected_columns_present(self, small_generator):
        """Key columns must be in the output."""
        df, _, _ = small_generator.generate_complete_dataset()
        for col in ["date", "t", "revenue", "tv_spend", "search_spend",
                     "social_spend", "display_spend", "print_ooh_spend"]:
            assert col in df.columns

    def test_no_negative_revenue(self, small_generator):
        """Revenue should never be negative."""
        df, _, _ = small_generator.generate_complete_dataset()
        assert (df["revenue"] >= 0).all()

    def test_no_negative_spend(self, small_generator):
        """Spend should never be negative."""
        df, _, _ = small_generator.generate_complete_dataset()
        for ch in ["tv", "search", "social", "display", "print_ooh"]:
            assert (df[f"{ch}_spend"] >= 0).all()

    def test_contributions_truth_shape(self, small_generator):
        """Contributions DataFrame should have correct columns."""
        _, contrib_df, _ = small_generator.generate_complete_dataset()
        assert len(contrib_df) == 90
        for ch in ["tv", "search", "social", "display", "print_ooh"]:
            assert f"{ch}_contribution" in contrib_df.columns

    def test_ground_truth_has_required_keys(self, small_generator):
        """Ground truth dict must have all top-level keys."""
        _, _, gt = small_generator.generate_complete_dataset()
        for key in ["baseline", "seasonality", "channels", "controls"]:
            assert key in gt

    def test_channel_count(self, small_generator):
        """Should have 5 channels."""
        assert len(small_generator.CHANNELS) == 5

    def test_saturation_in_zero_one(self, small_generator):
        """All saturated values must be in [0, 1]."""
        _, contrib_df, _ = small_generator.generate_complete_dataset()
        for ch in small_generator.CHANNELS:
            assert contrib_df[f"{ch}_saturated"].between(0, 1).all()
