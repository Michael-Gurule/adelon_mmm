"""Budget optimization utilities for Adelon MMM."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def greedy_budget_allocate(
    response_curves: dict[str, pd.DataFrame],
    total_budget: float,
    n_grid: int = 200,
    current_allocation: dict[str, float] | None = None,
) -> dict[str, float]:
    """Allocate budget across channels via greedy marginal ROI.

    Starting from the current allocation (or proportional split),
    redistributes budget in small increments toward channels with
    the highest marginal return.

    Args:
        response_curves: Dict mapping channel name to DataFrame with
            columns (spend, contribution_mean).
        total_budget: Total budget to allocate.
        n_grid: Number of allocation steps (higher = finer
            granularity).
        current_allocation: Optional starting allocation per channel.
            If provided, the optimizer scales it to total_budget and
            redistributes from there. If None, starts from an equal
            split across channels.

    Returns:
        Dict mapping channel name to allocated budget amount.

    Raises:
        ValueError: If total_budget or n_grid is not positive, or if
            response_curves is empty.
    """
    if total_budget <= 0:
        raise ValueError("total_budget must be positive.")
    if n_grid <= 0:
        raise ValueError("n_grid must be positive.")
    if not response_curves:
        raise ValueError("response_curves must not be empty.")

    channels = list(response_curves.keys())

    # Initialize allocation: scale current spend to target budget
    if current_allocation is not None:
        current_total = sum(current_allocation.values())
        if current_total > 0:
            scale = total_budget / current_total
            allocation = {
                ch: current_allocation.get(ch, 0.0) * scale for ch in channels
            }
        else:
            allocation = {ch: total_budget / len(channels) for ch in channels}
    else:
        allocation = {ch: total_budget / len(channels) for ch in channels}

    step = total_budget / n_grid

    # Pre-extract arrays for performance
    spend_arrays = {ch: rc["spend"].values for ch, rc in response_curves.items()}
    contrib_arrays = {
        ch: rc["contribution_mean"].values for ch, rc in response_curves.items()
    }

    # Iteratively move budget from lowest-marginal to highest-marginal
    for _ in range(n_grid):
        # Find channel with highest marginal ROI
        marginals = {}
        for ch in channels:
            current = allocation[ch]
            contrib_current = np.interp(current, spend_arrays[ch], contrib_arrays[ch])
            contrib_next = np.interp(
                current + step, spend_arrays[ch], contrib_arrays[ch]
            )
            marginals[ch] = (contrib_next - contrib_current) / step

        best_ch = max(marginals, key=marginals.get)  # type: ignore[arg-type]
        worst_ch = min(marginals, key=marginals.get)  # type: ignore[arg-type]

        # Stop if no improvement possible
        if best_ch == worst_ch or marginals[best_ch] <= marginals[worst_ch]:
            break

        # Move one step from worst to best (if worst has budget)
        if allocation[worst_ch] >= step:
            allocation[worst_ch] -= step
            allocation[best_ch] += step

    logger.debug(
        "Budget allocated: %s",
        {ch: f"${v:,.0f}" for ch, v in allocation.items()},
    )
    return allocation


def estimate_revenue(
    response_curves: dict[str, pd.DataFrame],
    allocation: dict[str, float],
) -> float:
    """Estimate total media revenue for a given budget allocation.

    Args:
        response_curves: Same format as greedy_budget_allocate input.
        allocation: Dict mapping channel to spend amount.

    Returns:
        Estimated total daily media revenue.
    """
    total = 0.0
    for ch, spend in allocation.items():
        rc = response_curves[ch]
        total += np.interp(spend, rc["spend"].values, rc["contribution_mean"].values)
    return total
