"""Budget optimization utilities for Adelon MMM."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.exceptions import OptimizationInfeasibleError

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


def optimize_constrained(
    response_curves: dict[str, pd.DataFrame],
    total_budget: float,
    min_spend: dict[str, float] | None = None,
    max_spend: dict[str, float] | None = None,
    min_roas: dict[str, float] | None = None,
) -> dict[str, float]:
    """Allocate budget via SciPy SLSQP constrained optimization.

    Maximizes total predicted media revenue subject to a total-budget
    equality constraint plus optional per-channel lower/upper bounds
    and per-channel minimum ROAS floors.

    **Solver choice — SciPy SLSQP vs CVXPY:**
    The Hill saturation function ``f(x) = x^S / (K^S + x^S)`` is
    non-convex (S-shaped), so CVXPY's Disciplined Convex Programming
    (DCP) rules reject it directly. CVXPY's non-convex backends (SCIP,
    IPOPT) add heavy optional dependencies and platform-specific
    compilation. SciPy SLSQP (Sequential Least Squares Programming)
    handles non-linear objectives and constraints natively, is already
    a core project dependency, and produces clean, maintainable code.
    The trade-off is that SLSQP finds a *local* optimum and may miss
    the global one; in practice, the revenue surface for typical MMM
    response curves is quasi-concave and SLSQP converges reliably.

    **Revenue model:**
    For each channel the contribution is read from the pre-computed
    ``response_curves`` DataFrame via linear interpolation over the
    fitted (adstock → Hill saturation → beta) curve. This matches the
    posterior-mean response surface used by the greedy optimizer and the
    dashboard.

    Args:
        response_curves: Dict mapping channel name to DataFrame with
            columns ``(spend, contribution_mean)``.
        total_budget: Total budget to allocate (must be positive).
        min_spend: Optional per-channel lower bounds on spend. Channels
            not listed default to 0.
        max_spend: Optional per-channel upper bounds on spend. Channels
            not listed default to ``total_budget``.
        min_roas: Optional per-channel minimum ROAS floor. For channel
            ``ch``, enforces
            ``contribution(x_ch) / x_ch >= min_roas[ch]``.

    Returns:
        Dict mapping channel name to allocated budget amount.

    Raises:
        ValueError: If ``total_budget`` is not positive, if
            ``response_curves`` is empty, or if any per-channel
            ``min_spend > max_spend``.
        OptimizationInfeasibleError: If the problem has no feasible
            solution (e.g., sum of minimums exceeds ``total_budget``).
    """
    if total_budget <= 0:
        raise ValueError("total_budget must be positive.")
    if not response_curves:
        raise ValueError("response_curves must not be empty.")

    channels = list(response_curves.keys())
    n = len(channels)

    # Resolve per-channel bounds
    lo = np.array(
        [float((min_spend or {}).get(ch, 0.0)) for ch in channels]
    )
    hi = np.array(
        [float((max_spend or {}).get(ch, total_budget)) for ch in channels]
    )

    for i, ch in enumerate(channels):
        if lo[i] < 0:
            raise ValueError(
                f"min_spend[{ch!r}] must be non-negative, got {lo[i]}."
            )
        if lo[i] > hi[i]:
            raise ValueError(
                f"min_spend[{ch!r}] ({lo[i]}) exceeds "
                f"max_spend[{ch!r}] ({hi[i]})."
            )

    if lo.sum() > total_budget:
        raise OptimizationInfeasibleError(
            f"Sum of minimum spend bounds ({lo.sum():,.0f}) exceeds "
            f"total_budget ({total_budget:,.0f}). Problem is infeasible."
        )

    # Pre-extract arrays for interpolation
    spend_arrays = [rc["spend"].values for rc in response_curves.values()]
    contrib_arrays = [rc["contribution_mean"].values for rc in response_curves.values()]

    def _revenue(x: np.ndarray) -> float:
        """Total media revenue for allocation vector x (negated for minimizer)."""
        total = 0.0
        for i in range(n):
            total += np.interp(x[i], spend_arrays[i], contrib_arrays[i])
        return -total  # negate: scipy.minimize minimizes

    # Constraints
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: x.sum() - total_budget,
            "jac": lambda x: np.ones(n),
        }
    ]

    if min_roas:
        for i, ch in enumerate(channels):
            floor = min_roas.get(ch)
            if floor is None:
                continue
            idx = i  # capture loop variable

            def _roas_con(x: np.ndarray, _i: int = idx, _f: float = floor) -> float:
                contrib = np.interp(x[_i], spend_arrays[_i], contrib_arrays[_i])
                if x[_i] <= 0:
                    return 0.0  # treat zero-spend as feasible (ROAS undefined)
                return contrib - _f * x[_i]

            constraints.append({"type": "ineq", "fun": _roas_con})

    # Initial point: start from lower bounds, distribute remaining budget
    # proportionally by per-channel headroom. This guarantees x0.sum() ==
    # total_budget and lo <= x0 <= hi without any post-hoc correction.
    x0 = lo.copy()
    remaining = total_budget - lo.sum()
    headroom = hi - lo
    total_headroom = float(headroom.sum())
    if total_headroom > 0:
        x0 += remaining * headroom / total_headroom
    else:
        # All channels pinned to their bounds; equality already satisfied.
        pass

    bounds = list(zip(lo.tolist(), hi.tolist()))

    result = minimize(
        fun=_revenue,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        raise OptimizationInfeasibleError(
            f"SLSQP solver did not converge: {result.message}"
        )

    allocation = {ch: float(result.x[i]) for i, ch in enumerate(channels)}
    logger.debug(
        "Constrained allocation: %s",
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
