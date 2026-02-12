"""Adelon MMM — core package."""

from src.exceptions import (
    ConfigurationError,
    DataValidationError,
    MMMError,
    ModelNotFittedError,
)
from src.optimization import estimate_revenue, greedy_budget_allocate
from src.preprocessing import (
    compute_roas,
    geometric_adstock,
    hill_saturation,
    load_config,
    prepare_mmm_data,
)
from src.visualization import (
    CHANNEL_COLORS,
    CHANNEL_LABELS,
    plot_adstock_decay_curves,
    plot_budget_optimizer,
    plot_channel_spend_distribution,
    plot_contribution_waterfall,
    plot_model_fit,
    plot_posterior_distributions,
    plot_residual_diagnostics,
    plot_response_curves,
    plot_roas_comparison,
    plot_saturation_curves,
    plot_spend_revenue_trends,
    plot_trace_diagnostics,
)

try:
    from src.bayesian_mmm import BayesianMMM
except ImportError:
    pass
