"""
Adelon — Plotly visualization utilities for Bayesian Marketing Mix Modeling.

All functions return go.Figure objects. No class state — pure functions only.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from src.preprocessing import geometric_adstock, hill_saturation

# Adelon Color Palette
ADELON_PALETTE = ["#002D40", "#C5A059", "#4F7942", "#A63446", "#1A1A1A"]

CHANNEL_COLORS = {
    "tv": "#002D40",  # Deep Aegean
    "search": "#C5A059",  # Athenian Gold
    "social": "#4F7942",  # Laurel Green
    "display": "#A63446",  # Pompeian Red
    "print_ooh": "#1A1A1A",  # Obsidian
}

CHANNEL_LABELS = {
    "tv": "TV",
    "search": "Search",
    "social": "Social",
    "display": "Display",
    "print_ooh": "Print/OOH",
}


def _color(channel):
    return CHANNEL_COLORS.get(channel, "#999999")


def _label(channel):
    return CHANNEL_LABELS.get(channel, channel)


# EDA Charts
def plot_spend_revenue_trends(df, spend_cols=None):
    """Dual-axis time series: stacked monthly channel spend (area) and revenue (line).

    Data is aggregated to monthly totals before plotting. Plotly's built-in
    zoom controls allow drilling down to shorter windows after render.

    Args:
        df: MMM DataFrame with date, revenue, and spend columns.
        spend_cols: Explicit list of owned-channel spend column names. If
            omitted, all columns ending in ``_spend`` are used (which may
            include control-variable columns such as ``competitor_spend``).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if spend_cols is None:
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
    channels = [c.replace("_spend", "") for c in spend_cols]

    # Aggregate to monthly totals (MS = month-start anchor)
    df_monthly = (
        df.set_index("date")[spend_cols + ["revenue"]]
        .resample("MS")
        .sum()
        .reset_index()
    )

    for ch, col in zip(channels, spend_cols):
        fig.add_trace(
            go.Scatter(
                x=df_monthly["date"],
                y=df_monthly[col],
                name=_label(ch),
                stackgroup="spend",
                line=dict(width=0),
                fillcolor=_color(ch),
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=df_monthly["date"],
            y=df_monthly["revenue"],
            name="Revenue",
            line=dict(color="black", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Monthly Channel Spend & Revenue",
        xaxis_title="Month",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Spend ($)", secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=True)
    return fig


def plot_channel_spend_distribution(df, spend_cols=None):
    """Box plots showing spend distribution per channel.

    Args:
        df: MMM DataFrame with spend columns.
        spend_cols: Explicit list of owned-channel spend column names. If
            omitted, all columns ending in ``_spend`` are used (which may
            include control-variable columns such as ``competitor_spend``).
    """
    if spend_cols is None:
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
    channels = [c.replace("_spend", "") for c in spend_cols]

    fig = go.Figure()
    for ch, col in zip(channels, spend_cols):
        fig.add_trace(
            go.Box(
                y=df[col],
                name=_label(ch),
                marker_color=_color(ch),
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Daily Spend Distribution by Channel",
        yaxis_title="Spend ($)",
        template="plotly_white",
        showlegend=False,
    )
    return fig


# Transform Visualization
def plot_adstock_decay_curves(alphas, n_days=28):
    """
    Decay curves showing adstock carryover for different alpha values.

    Args:
        alphas: Dict mapping channel name -> alpha value.
        n_days: Number of days to show decay.
    """
    fig = go.Figure()
    impulse = np.zeros(n_days)
    impulse[0] = 1.0

    for ch, alpha in alphas.items():
        decay = geometric_adstock(impulse, alpha)
        fig.add_trace(
            go.Scatter(
                x=list(range(n_days)),
                y=decay,
                name=f"{_label(ch)} (α={alpha})",
                line=dict(color=_color(ch), width=2),
            )
        )

    fig.update_layout(
        title="Adstock Decay Curves (Impulse Response)",
        xaxis_title="Days After Exposure",
        yaxis_title="Remaining Effect",
        template="plotly_white",
    )
    return fig


def plot_saturation_curves(K_values, S_values, spend_max=300_000, n_points=200):
    """
    Hill saturation curves showing diminishing returns per channel.

    Args:
        K_values: Dict channel -> K parameter.
        S_values: Dict channel -> S parameter.
        spend_max: Upper limit of spend axis.
        n_points:  Grid points.
    """
    fig = go.Figure()
    x = np.linspace(0, spend_max, n_points)

    for ch in K_values:
        K, S = K_values[ch], S_values[ch]
        y = hill_saturation(x, K, S)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"{_label(ch)} (K={K:,.0f}, S={S})",
                line=dict(color=_color(ch), width=2),
            )
        )
        # Half-saturation marker
        fig.add_trace(
            go.Scatter(
                x=[K],
                y=[0.5],
                mode="markers",
                marker=dict(color=_color(ch), size=10, symbol="diamond"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Hill Saturation Curves",
        xaxis_title="Adstocked Spend ($)",
        yaxis_title="Saturation (0–1)",
        template="plotly_white",
    )
    return fig


# Model Results Charts
def plot_contribution_waterfall(contributions, total_revenue):
    """
    Waterfall chart decomposing total revenue into components.

    Args:
        contributions: Dict mapping component name -> revenue contribution.
        total_revenue: Total observed revenue.
    """
    names = list(contributions.keys()) + ["Total"]
    values = list(contributions.values()) + [total_revenue]
    measures = ["relative"] * len(contributions) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            x=names,
            y=values,
            measure=measures,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#002D40"}},
            decreasing={"marker": {"color": "#A63446"}},
            totals={"marker": {"color": "#C5A059"}},
        )
    )
    fig.update_layout(
        title="Revenue Decomposition",
        yaxis_title="Revenue ($)",
        template="plotly_white",
    )
    return fig


def plot_posterior_distributions(trace, params, ground_truth=None, channels=None):
    """
    Grid of posterior KDE plots with optional ground truth lines.

    Args:
        trace:        ArviZ InferenceData.
        params:       Variable names to plot (e.g., ["adstock_alpha", "beta_media"]).
        ground_truth: Optional dict of {param_label: true_value}.
        channels:     Channel names for labeling vector params.
    """
    n_params = sum(
        trace.posterior[p].shape[-1] if len(trace.posterior[p].shape) > 2 else 1
        for p in params
    )
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols

    subplot_titles = []
    for p in params:
        shape = trace.posterior[p].shape
        if len(shape) > 2:
            n = shape[-1]
            for i in range(n):
                label = channels[i] if channels and i < len(channels) else str(i)
                subplot_titles.append(f"{p} [{label}]")
        else:
            subplot_titles.append(p)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    plot_idx = 0
    for p in params:
        samples = trace.posterior[p].values
        if len(samples.shape) > 2:
            for i in range(samples.shape[-1]):
                s = samples[:, :, i].flatten()
                row = plot_idx // cols + 1
                col = plot_idx % cols + 1
                kde = gaussian_kde(s)
                x_grid = np.linspace(s.min(), s.max(), 200)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=kde(x_grid),
                        fill="tozeroy",
                        showlegend=False,
                        line=dict(color="#002D40"),
                    ),
                    row=row,
                    col=col,
                )
                if ground_truth and subplot_titles[plot_idx] in ground_truth:
                    tv = ground_truth[subplot_titles[plot_idx]]
                    fig.add_vline(
                        x=tv, line_dash="dash", line_color="red", row=row, col=col
                    )
                plot_idx += 1
        else:
            s = samples.flatten()
            row = plot_idx // cols + 1
            col = plot_idx % cols + 1
            kde = gaussian_kde(s)
            x_grid = np.linspace(s.min(), s.max(), 200)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=kde(x_grid),
                    fill="tozeroy",
                    showlegend=False,
                    line=dict(color="#002D40"),
                ),
                row=row,
                col=col,
            )
            if ground_truth and subplot_titles[plot_idx] in ground_truth:
                tv = ground_truth[subplot_titles[plot_idx]]
                fig.add_vline(
                    x=tv, line_dash="dash", line_color="red", row=row, col=col
                )
            plot_idx += 1

    fig.update_layout(
        title="Posterior Distributions",
        template="plotly_white",
        height=300 * rows,
        showlegend=False,
    )
    return fig


def plot_response_curves(response_curves, channels=None):
    """
    Spend vs incremental revenue curves with 94% credible interval shading.

    Args:
        response_curves: Dict channel -> DataFrame(spend, contribution_mean,
                         contribution_lower, contribution_upper).
        channels:        Subset to plot. None = all.
    """
    fig = go.Figure()
    plot_channels = channels or list(response_curves.keys())

    for ch in plot_channels:
        rc = response_curves[ch]
        color = _color(ch)
        fig.add_trace(
            go.Scatter(
                x=rc["spend"],
                y=rc["contribution_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rc["spend"],
                y=rc["contribution_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=color.replace(")", ", 0.2)")
                .replace("rgb", "rgba")
                .replace("#", "rgba(")
                if color.startswith("rgb")
                else "rgba(0,45,64,0.15)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rc["spend"],
                y=rc["contribution_mean"],
                name=_label(ch),
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title="Media Response Curves (Marginal)",
        xaxis_title="Daily Spend ($)",
        yaxis_title="Incremental Revenue ($)",
        template="plotly_white",
    )
    return fig


def plot_roas_comparison(roas_df):
    """Horizontal bar chart of ROAS per channel."""
    fig = go.Figure(
        go.Bar(
            x=roas_df["roas_mean"],
            y=[_label(ch) for ch in roas_df["channel"]],
            orientation="h",
            marker_color=[_color(ch) for ch in roas_df["channel"]],
        )
    )
    fig.update_layout(
        title="Return on Ad Spend (ROAS) by Channel",
        xaxis_title="ROAS",
        template="plotly_white",
    )
    return fig


def plot_budget_optimizer(
    allocation,
    response_curves,
    total_budget,
):
    """Plot pre-computed optimal budget allocation and expected revenue.

    Args:
        allocation: Dict mapping channel name to allocated spend
            (from ``greedy_budget_allocate``).
        response_curves: Dict mapping channel name to DataFrame with
            columns (spend, contribution_mean).
        total_budget: Total daily budget (used in title).
    """
    from src.optimization import estimate_revenue

    channels = list(allocation.keys())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Optimal Allocation", "Expected Revenue"],
    )

    fig.add_trace(
        go.Bar(
            x=[_label(ch) for ch in channels],
            y=[allocation[ch] for ch in channels],
            marker_color=[_color(ch) for ch in channels],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Per-channel revenue at allocated spend
    rev_by_ch = {}
    for ch in channels:
        rev_by_ch[ch] = estimate_revenue(response_curves, {ch: allocation[ch]})

    fig.add_trace(
        go.Bar(
            x=[_label(ch) for ch in channels],
            y=[rev_by_ch[ch] for ch in channels],
            marker_color=[_color(ch) for ch in channels],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Budget Optimization (Total: ${total_budget:,.0f}/day)",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Allocated Spend ($)", row=1, col=1)
    fig.update_yaxes(title_text="Expected Revenue ($)", row=1, col=2)
    return fig


def plot_model_fit(
    actual, predicted_mean, predicted_lower, predicted_upper, dates=None
):
    """
    Actual vs posterior predictive mean with credible interval shading.

    Args:
        actual:           Observed revenue, shape (T,).
        predicted_mean:   Posterior predictive mean, shape (T,).
        predicted_lower:  Lower CI bound, shape (T,).
        predicted_upper:  Upper CI bound, shape (T,).
        dates:            Optional date array for x-axis.
    """
    x = dates if dates is not None else np.arange(len(actual))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=predicted_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=predicted_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,45,64,0.2)",
            name="94% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=predicted_mean,
            name="Predicted",
            line=dict(color="#002D40", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            name="Actual",
            line=dict(color="#1A1A1A", width=1, dash="dot"),
        )
    )

    fig.update_layout(
        title="Model Fit: Actual vs Predicted Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        template="plotly_white",
    )
    return fig


def plot_residual_diagnostics(residuals, dates=None):
    """
    Three-panel residual diagnostics: time series, histogram, ACF.

    Args:
        residuals: Model residuals (actual - predicted), shape (T,).
        dates:     Optional date array.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Residuals Over Time", "Distribution", "Autocorrelation"],
    )

    x = dates if dates is not None else np.arange(len(residuals))

    # Time series
    fig.add_trace(
        go.Scatter(
            x=x,
            y=residuals,
            mode="lines",
            line=dict(color="#002D40", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#A63446", row=1, col=1)

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=40,
            marker_color="#002D40",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # ACF (up to 30 lags)
    n = len(residuals)
    max_lag = min(30, n - 1)
    acf = np.correlate(
        residuals - residuals.mean(), residuals - residuals.mean(), mode="full"
    )
    acf = acf[n - 1 :] / acf[n - 1]  # normalize
    fig.add_trace(
        go.Bar(
            x=list(range(max_lag + 1)),
            y=acf[: max_lag + 1],
            marker_color="#002D40",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    # Significance bounds
    sig = 1.96 / np.sqrt(n)
    fig.add_hline(y=sig, line_dash="dash", line_color="#A63446", row=1, col=3)
    fig.add_hline(y=-sig, line_dash="dash", line_color="#A63446", row=1, col=3)

    fig.update_layout(
        title="Residual Diagnostics",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_trace_diagnostics(trace, var_names, channels=None):
    """
    Trace plots showing chain mixing for key parameters.

    Args:
        trace:     ArviZ InferenceData.
        var_names: Parameters to plot.
        channels:  Channel names for labeling.
    """
    # Collect all scalar/first-element traces
    plots = []
    for var in var_names:
        samples = trace.posterior[var].values  # (chains, draws, ...)
        if len(samples.shape) == 2:
            plots.append((var, samples))
        elif len(samples.shape) == 3:
            for i in range(samples.shape[-1]):
                label = (
                    f"{var}[{channels[i]}]"
                    if channels and i < len(channels)
                    else f"{var}[{i}]"
                )
                plots.append((label, samples[:, :, i]))

    n_plots = len(plots)
    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=[p[0] for p in plots])

    chain_colors = ["#002D40", "#C5A059", "#4F7942", "#A63446"]
    for row_idx, (name, data) in enumerate(plots, 1):
        n_chains = data.shape[0]
        for c in range(n_chains):
            fig.add_trace(
                go.Scatter(
                    y=data[c],
                    mode="lines",
                    line=dict(color=chain_colors[c % len(chain_colors)], width=0.5),
                    opacity=0.7,
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        title="MCMC Trace Plots",
        template="plotly_white",
        height=200 * n_plots,
        showlegend=False,
    )
    return fig
