"""Tests for src/preprocessing.py"""

import numpy as np
import pytest

from src.preprocessing import compute_roas, geometric_adstock, hill_saturation, prepare_mmm_data


class TestGeometricAdstock:
    def test_zero_alpha_returns_spend(self):
        """With no carryover, adstock equals raw spend."""
        x = np.array([100.0, 200.0, 300.0])
        result = geometric_adstock(x, alpha=0.0)
        np.testing.assert_allclose(result, x)

    def test_high_alpha_shows_carryover(self):
        """An impulse with high alpha should decay slowly."""
        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(x, alpha=0.9)
        assert all(result[1:] > 0)
        assert all(result[i] > result[i + 1] for i in range(1, 4))

    def test_convergence_to_zero(self):
        """Adstock with alpha < 1 decays to zero after impulse."""
        x = np.zeros(100)
        x[0] = 1000.0
        result = geometric_adstock(x, alpha=0.5)
        assert result[-1] < 0.01

    def test_steady_state_convergence(self):
        """Constant spend should converge to spend / (1 - alpha)."""
        spend = 10000.0
        alpha = 0.85
        x = np.full(500, spend)
        result = geometric_adstock(x, alpha)
        expected_steady_state = spend / (1 - alpha)
        assert abs(result[-1] - expected_steady_state) / expected_steady_state < 0.01

    def test_invalid_alpha_raises(self):
        """Alpha outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            geometric_adstock(np.array([1.0, 2.0]), alpha=1.5)

    def test_output_same_shape(self):
        """Output shape must match input."""
        x = np.random.rand(200)
        assert geometric_adstock(x, 0.7).shape == x.shape


class TestHillSaturation:
    def test_output_in_zero_one(self):
        """All outputs must be in [0, 1]."""
        x = np.array([0.0, 10000.0, 50000.0, 100000.0, 1e9])
        result = hill_saturation(x, K=50000, S=2.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_half_saturation_at_K(self):
        """f(K) must equal exactly 0.5."""
        K = 50000.0
        result = hill_saturation(np.array([K]), K=K, S=2.0)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_zero_spend_gives_zero(self):
        """f(0) = 0."""
        result = hill_saturation(np.array([0.0]), K=50000, S=2.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_monotonically_increasing(self):
        """Saturation curve must be monotonically increasing."""
        x = np.linspace(0, 200000, 100)
        result = hill_saturation(x, K=50000, S=2.5)
        assert np.all(np.diff(result) >= 0)

    def test_invalid_K_raises(self):
        """K <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="K must be"):
            hill_saturation(np.array([1.0]), K=0.0, S=2.0)

    def test_invalid_S_raises(self):
        """S <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="S must be"):
            hill_saturation(np.array([1.0]), K=50000, S=-1.0)


class TestComputeRoas:
    def test_basic_calculation(self):
        """ROAS = total contribution / total spend."""
        contrib = np.array([10000.0, 20000.0, 30000.0])
        spend = np.array([5000.0, 10000.0, 15000.0])
        roas = compute_roas(contrib, spend)
        assert roas == pytest.approx(2.0)

    def test_zero_spend_returns_zero(self):
        """ROAS is 0 when spend is 0."""
        assert compute_roas(np.array([100.0]), np.array([0.0])) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise."""
        with pytest.raises(ValueError):
            compute_roas(np.array([1.0, 2.0]), np.array([1.0]))


class TestPrepareMMMData:
    def test_output_shapes(self, small_mmm_df, config):
        """All arrays should have correct shapes."""
        spend, revenue, fourier, dow, controls, t = prepare_mmm_data(small_mmm_df, config)
        T = len(small_mmm_df)
        assert spend.shape == (T, 5)
        assert revenue.shape == (T,)
        assert fourier.shape == (T, 6)
        assert controls.shape == (T, 4)
        assert t.shape == (T,)

    def test_missing_column_raises(self, small_mmm_df, config):
        """Missing required column should raise ValueError."""
        bad_df = small_mmm_df.drop(columns=["tv_spend"])
        with pytest.raises(ValueError, match="Missing columns"):
            prepare_mmm_data(bad_df, config)
