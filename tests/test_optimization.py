"""Tests for src/optimization.py"""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import OptimizationInfeasibleError
from src.optimization import (
    estimate_revenue,
    greedy_budget_allocate,
    optimize_constrained,
)


@pytest.fixture
def sample_response_curves() -> dict[str, pd.DataFrame]:
    """Create sample response curves for testing."""
    curves = {}
    for ch, max_contrib in [("tv", 10000), ("search", 8000), ("social", 5000)]:
        spend = np.linspace(0, 100000, 200)
        # Concave response: diminishing returns
        contribution = max_contrib * (1 - np.exp(-spend / 30000))
        curves[ch] = pd.DataFrame({
            "spend": spend,
            "contribution_mean": contribution,
        })
    return curves


class TestGreedyBudgetAllocate:
    def test_total_allocation_equals_budget(self, sample_response_curves):
        """Total allocated budget should equal total_budget."""
        # Arrange
        total_budget = 50000.0

        # Act
        allocation = greedy_budget_allocate(
            sample_response_curves, total_budget
        )

        # Assert
        assert sum(allocation.values()) == pytest.approx(
            total_budget, rel=1e-2
        )

    def test_all_channels_present(self, sample_response_curves):
        """Every channel should appear in the allocation."""
        allocation = greedy_budget_allocate(
            sample_response_curves, 50000.0
        )
        assert set(allocation.keys()) == set(sample_response_curves.keys())

    def test_allocations_non_negative(self, sample_response_curves):
        """No channel should have negative allocation."""
        allocation = greedy_budget_allocate(
            sample_response_curves, 50000.0
        )
        for ch, amount in allocation.items():
            assert amount >= 0, f"Channel {ch} has negative allocation"

    def test_highest_roi_channel_gets_most(self, sample_response_curves):
        """Channel with steepest response should get the most budget."""
        allocation = greedy_budget_allocate(
            sample_response_curves, 10000.0, n_grid=100
        )
        # TV has the highest max contribution, so should get the most
        assert allocation["tv"] >= allocation["social"]

    def test_zero_budget_raises(self, sample_response_curves):
        """total_budget <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="total_budget"):
            greedy_budget_allocate(sample_response_curves, 0.0)

    def test_negative_budget_raises(self, sample_response_curves):
        """Negative total_budget should raise ValueError."""
        with pytest.raises(ValueError, match="total_budget"):
            greedy_budget_allocate(sample_response_curves, -1000.0)

    def test_zero_n_grid_raises(self, sample_response_curves):
        """n_grid <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_grid"):
            greedy_budget_allocate(
                sample_response_curves, 50000.0, n_grid=0
            )

    def test_empty_response_curves_raises(self):
        """Empty response_curves dict should raise ValueError."""
        with pytest.raises(ValueError, match="response_curves"):
            greedy_budget_allocate({}, 50000.0)

    def test_finer_grid_produces_similar_results(
        self, sample_response_curves
    ):
        """Higher n_grid should produce similar allocation."""
        # Arrange
        coarse = greedy_budget_allocate(
            sample_response_curves, 50000.0, n_grid=50
        )
        fine = greedy_budget_allocate(
            sample_response_curves, 50000.0, n_grid=500
        )

        # Assert: allocations should be within 20% of each other
        for ch in coarse:
            if fine[ch] > 0:
                ratio = coarse[ch] / fine[ch]
                assert 0.8 < ratio < 1.2, (
                    f"Channel {ch}: coarse={coarse[ch]:.0f}, "
                    f"fine={fine[ch]:.0f}"
                )


class TestEstimateRevenue:
    def test_returns_float(self, sample_response_curves):
        """estimate_revenue should return a float."""
        allocation = {"tv": 20000.0, "search": 15000.0, "social": 10000.0}
        result = estimate_revenue(sample_response_curves, allocation)
        assert isinstance(result, float)

    def test_zero_allocation_returns_zero(self, sample_response_curves):
        """Zero spend across all channels should return ~0 revenue."""
        allocation = {"tv": 0.0, "search": 0.0, "social": 0.0}
        result = estimate_revenue(sample_response_curves, allocation)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_higher_spend_gives_higher_revenue(
        self, sample_response_curves
    ):
        """More total spend should yield more revenue."""
        low = estimate_revenue(
            sample_response_curves,
            {"tv": 5000.0, "search": 5000.0, "social": 5000.0},
        )
        high = estimate_revenue(
            sample_response_curves,
            {"tv": 30000.0, "search": 30000.0, "social": 30000.0},
        )
        assert high > low

    def test_consistent_with_allocation(self, sample_response_curves):
        """Revenue from greedy allocation should be reproducible."""
        allocation = greedy_budget_allocate(
            sample_response_curves, 50000.0
        )
        rev1 = estimate_revenue(sample_response_curves, allocation)
        rev2 = estimate_revenue(sample_response_curves, allocation)
        assert rev1 == rev2


@pytest.fixture
def equal_response_curves() -> dict[str, pd.DataFrame]:
    """Response curves with identical shapes across all channels."""
    curves = {}
    for ch in ("tv", "search", "social"):
        spend = np.linspace(0, 100000, 200)
        contribution = 10000.0 * (1 - np.exp(-spend / 30000))
        curves[ch] = pd.DataFrame(
            {"spend": spend, "contribution_mean": contribution}
        )
    return curves


class TestOptimizeConstrained:
    def test_total_allocation_equals_budget(self, sample_response_curves):
        """Constrained optimizer allocation must sum to total_budget."""
        total_budget = 50000.0
        allocation = optimize_constrained(sample_response_curves, total_budget)
        assert sum(allocation.values()) == pytest.approx(total_budget, abs=1.0)

    def test_all_channels_present(self, sample_response_curves):
        """Every channel must appear in the allocation."""
        allocation = optimize_constrained(sample_response_curves, 50000.0)
        assert set(allocation.keys()) == set(sample_response_curves.keys())

    def test_per_channel_min_bounds_respected(self, sample_response_curves):
        """Allocations must be >= their per-channel minimums."""
        min_spend = {"tv": 5000.0, "search": 3000.0, "social": 2000.0}
        allocation = optimize_constrained(
            sample_response_curves, 50000.0, min_spend=min_spend
        )
        for ch, floor in min_spend.items():
            assert allocation[ch] >= floor - 1e-3, (
                f"Channel {ch}: allocation {allocation[ch]:.2f} < min {floor}"
            )

    def test_per_channel_max_bounds_respected(self, sample_response_curves):
        """Allocations must be <= their per-channel maximums."""
        max_spend = {"tv": 20000.0, "search": 15000.0, "social": 10000.0}
        allocation = optimize_constrained(
            sample_response_curves, 40000.0, max_spend=max_spend
        )
        for ch, cap in max_spend.items():
            assert allocation[ch] <= cap + 1e-3, (
                f"Channel {ch}: allocation {allocation[ch]:.2f} > max {cap}"
            )

    def test_constrained_revenue_gte_greedy(self, sample_response_curves):
        """Constrained optimizer must achieve equal or better revenue than greedy."""
        total_budget = 50000.0
        greedy_alloc = greedy_budget_allocate(sample_response_curves, total_budget)
        constrained_alloc = optimize_constrained(
            sample_response_curves, total_budget
        )
        greedy_rev = estimate_revenue(sample_response_curves, greedy_alloc)
        constrained_rev = estimate_revenue(sample_response_curves, constrained_alloc)
        # Allow a small numerical tolerance
        assert constrained_rev >= greedy_rev - 1.0, (
            f"Constrained revenue {constrained_rev:.2f} < greedy {greedy_rev:.2f}"
        )

    def test_equal_curves_roughly_equal_allocation(self, equal_response_curves):
        """When all channels have identical response curves, allocation is roughly equal."""
        total_budget = 60000.0
        allocation = optimize_constrained(equal_response_curves, total_budget)
        expected = total_budget / len(equal_response_curves)
        for ch, amount in allocation.items():
            assert amount == pytest.approx(expected, rel=0.05), (
                f"Channel {ch}: allocation {amount:.0f} deviates from equal "
                f"split {expected:.0f} by more than 5%"
            )

    def test_infeasible_budget_raises(self, sample_response_curves):
        """Budget smaller than sum of minimums must raise OptimizationInfeasibleError."""
        min_spend = {"tv": 20000.0, "search": 20000.0, "social": 20000.0}
        with pytest.raises(OptimizationInfeasibleError):
            optimize_constrained(
                sample_response_curves, 50000.0, min_spend=min_spend
            )

    def test_zero_budget_raises(self, sample_response_curves):
        """total_budget <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="total_budget"):
            optimize_constrained(sample_response_curves, 0.0)

    def test_empty_response_curves_raises(self):
        """Empty response_curves must raise ValueError."""
        with pytest.raises(ValueError, match="response_curves"):
            optimize_constrained({}, 50000.0)

    def test_min_exceeds_max_raises(self, sample_response_curves):
        """min_spend > max_spend for a channel must raise ValueError."""
        with pytest.raises(ValueError, match="min_spend"):
            optimize_constrained(
                sample_response_curves,
                50000.0,
                min_spend={"tv": 30000.0},
                max_spend={"tv": 10000.0},
            )
