"""Adelon — Marketing Mix Model Dashboard.

Run with: streamlit run dashboards/app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.bayesian_mmm import BayesianMMM
from src.exceptions import OptimizationInfeasibleError
from src.optimization import (
    estimate_revenue,
    greedy_budget_allocate,
    optimize_constrained,
)
from src.preprocessing import (
    geometric_adstock,
    hill_saturation,
    load_config,
)
from src.visualization import (
    CHANNEL_COLORS,
    CHANNEL_LABELS,
    plot_adstock_decay_curves,
    plot_budget_optimizer,
    plot_channel_spend_distribution,
    plot_contribution_waterfall,
    plot_model_fit,
    plot_residual_diagnostics,
    plot_response_curves,
    plot_roas_comparison,
    plot_saturation_curves,
    plot_spend_revenue_trends,
    plot_trace_diagnostics,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Page config
st.set_page_config(
    page_title="Adelon MMM Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# Data loading (cached)
@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict, dict]:
    """Load MMM daily data, ground truth, and config."""
    config = load_config(PROJECT_ROOT / "config" / "model_config.yaml")
    df = pd.read_csv(
        PROJECT_ROOT / config["data"]["path"],
        parse_dates=["date"],
    )
    gt_path = PROJECT_ROOT / config["data"]["ground_truth_path"]
    with open(gt_path) as f:
        gt = json.load(f)
    return df, gt, config


@st.cache_resource
def load_model(
    _config: dict,
) -> BayesianMMM | None:
    """Load pre-fitted BayesianMMM instance with trace."""
    trace_path = (
        PROJECT_ROOT
        / _config["artifacts"]["traces_dir"]
        / _config["artifacts"]["results_filename"]
    )
    if not trace_path.exists():
        return None
    df = pd.read_csv(
        PROJECT_ROOT / _config["data"]["path"],
        parse_dates=["date"],
    )
    model = BayesianMMM(_config)
    model.build_model(df)
    model.load_results(str(trace_path))
    return model


# Helpers


def _label(ch: str) -> str:
    return CHANNEL_LABELS.get(ch, ch)


def _color(ch: str) -> str:
    return CHANNEL_COLORS.get(ch, "#999999")


def _fmt(val: float, prefix: str = "$", decimals: int = 0) -> str:
    if decimals == 0:
        return f"{prefix}{val:,.0f}"
    return f"{prefix}{val:,.{decimals}f}"


# Navigation


def main() -> None:
    """Dashboard entry point."""
    df, gt, config = load_data()
    model = load_model(config)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Channel Deep Dive",
            "Model Results",
            "Response Curves",
            "Budget Optimizer",
            "MCMC Diagnostics",
        ],
    )

    if model is None:
        st.sidebar.warning(
            "No fitted trace found. Run `adelon-train` first to generate "
            "`traces/mmm_trace.netcdf`. Pages requiring posterior results "
            "will show limited content."
        )

    if page == "Overview":
        show_overview(df, gt, config, model)
    elif page == "Channel Deep Dive":
        show_channel_deep_dive(df, gt, config)
    elif page == "Model Results":
        show_model_results(df, gt, config, model)
    elif page == "Response Curves":
        show_response_curves(df, gt, config, model)
    elif page == "Budget Optimizer":
        show_budget_optimizer(df, gt, config, model)
    elif page == "MCMC Diagnostics":
        show_mcmc_diagnostics(config, model)


# Page 1: Overview


def show_overview(
    df: pd.DataFrame,
    gt: dict,
    config: dict,
    model: BayesianMMM | None,
) -> None:
    """Overview page with KPIs and trends."""
    st.title("Adelon — Marketing Mix Model")
    st.markdown(
        "End-to-end revenue decomposition using Bayesian inference "
        "with PyMC. Synthetic data with known ground truth for "
        "parameter recovery validation."
    )
    st.markdown("---")

    channels = config["data"]["channels"]
    spend_cols = config["data"]["spend_cols"]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total_revenue = float(df["revenue"].sum())
    total_spend = float(df[spend_cols].sum().sum())
    avg_daily_revenue = float(df["revenue"].mean())
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0

    with col1:
        st.metric("Total Revenue (3yr)", _fmt(total_revenue))
    with col2:
        st.metric("Total Media Spend", _fmt(total_spend))
    with col3:
        st.metric("Avg Daily Revenue", _fmt(avg_daily_revenue))
    with col4:
        st.metric("Overall ROAS", f"{overall_roas:.2f}x")

    st.markdown("---")

    # Spend & revenue trends
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = plot_spend_revenue_trends(df, spend_cols=spend_cols)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Avg Monthly Spend")
        monthly_spend = df.set_index("date")[spend_cols].resample("MS").sum()
        for ch in channels:
            avg_mo = float(monthly_spend[f"{ch}_spend"].mean())
            st.write(f"**{_label(ch)}**: {_fmt(avg_mo)}/mo")
        st.write(f"**Total**: {_fmt(float(monthly_spend.sum(axis=1).mean()))}/mo")

    # Revenue decomposition (ground truth)
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue Decomposition (Ground Truth)")
        decomp = gt.get("derived", {}).get("revenue_decomposition_pct", {})
        if decomp:
            fig = go.Figure(
                go.Pie(
                    labels=[k.replace("_", " ").title() for k in decomp.keys()],
                    values=list(decomp.values()),
                    hole=0.4,
                    marker=dict(
                        colors=[
                            "#002D40",
                            "#C5A059",
                            "#4F7942",
                            "#A63446",
                        ]
                    ),
                )
            )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `adelon-generate` to compute decomposition.")

    with col2:
        st.subheader("Channel Spend Distribution")
        fig = plot_channel_spend_distribution(df, spend_cols=spend_cols)
        st.plotly_chart(fig, use_container_width=True)

    # Dataset info
    st.markdown("---")
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Rows**: {len(df):,}")
        st.write(
            f"**Date range**: {df['date'].min().date()} to {df['date'].max().date()}"
        )
    with col2:
        st.write(f"**Channels**: {len(channels)}")
        st.write(f"**Controls**: {len(config['data']['control_cols'])}")
    with col3:
        st.write(f"**Fourier order**: {config['data']['fourier_order']}")
        st.write(f"**Trace loaded**: {'Yes' if model is not None else 'No'}")


# Page 2: Channel Deep Dive


def show_channel_deep_dive(
    df: pd.DataFrame,
    gt: dict,
    config: dict,
) -> None:
    """Channel-level parameter exploration."""
    st.title("Channel Deep Dive")
    channels = config["data"]["channels"]

    selected = st.selectbox(
        "Select Channel",
        channels,
        format_func=_label,
    )

    params = gt["channels"][selected]
    st.markdown(f"### {_label(selected)} — Ground Truth Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Adstock Alpha", f"{params['adstock_alpha']:.2f}")
    with col2:
        st.metric("Saturation K", _fmt(params["saturation_K"]))
    with col3:
        st.metric("Saturation S", f"{params['saturation_S']:.1f}")
    with col4:
        st.metric("Beta", f"{params['beta']:.4f}")

    st.markdown("---")

    # Spend time series for selected channel
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Spend")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"{selected}_spend"],
                line=dict(color=_color(selected), width=1.5),
                name=_label(selected),
            )
        )
        fig.update_layout(
            yaxis_title="Spend ($)",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Spend Distribution")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df[f"{selected}_spend"],
                nbinsx=50,
                marker_color=_color(selected),
            )
        )
        fig.update_layout(
            xaxis_title="Daily Spend ($)",
            yaxis_title="Count",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Adstock & saturation transforms
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adstock Decay Curves")
        alphas = {ch: gt["channels"][ch]["adstock_alpha"] for ch in channels}
        fig = plot_adstock_decay_curves(alphas, n_days=42)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Saturation Curves")
        k_vals = {ch: gt["channels"][ch]["saturation_K"] for ch in channels}
        s_vals = {ch: gt["channels"][ch]["saturation_S"] for ch in channels}
        fig = plot_saturation_curves(k_vals, s_vals)
        st.plotly_chart(fig, use_container_width=True)

    # Combined transform for selected channel
    st.markdown("---")
    st.subheader(f"{_label(selected)}: Raw Spend -> Adstock -> Saturation")

    raw = df[f"{selected}_spend"].values
    adstocked = geometric_adstock(raw, params["adstock_alpha"])
    saturated = hill_saturation(
        adstocked, params["saturation_K"], params["saturation_S"]
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Raw Spend",
            "After Adstock",
            "After Saturation",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=raw,
            line=dict(color=_color(selected), width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=adstocked,
            line=dict(color=_color(selected), width=1),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=saturated,
            line=dict(color=_color(selected), width=1),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="$", row=1, col=2)
    fig.update_yaxes(title_text="Saturation (0-1)", row=1, col=3)
    fig.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Ground truth parameter table (all channels)
    st.markdown("---")
    st.subheader("All Channel Parameters (Ground Truth)")
    rows = []
    for ch in channels:
        p = gt["channels"][ch]
        rows.append(
            {
                "Channel": _label(ch),
                "Adstock Alpha": p["adstock_alpha"],
                "Saturation K": f"${p['saturation_K']:,.0f}",
                "Saturation S": p["saturation_S"],
                "Beta": f"{p['beta']:.4f}",
                "Base Spend": f"${p['base_daily_spend']:,.0f}",
            }
        )
    st.dataframe(
        pd.DataFrame(rows).set_index("Channel"),
        use_container_width=True,
    )


# Page 3: Model Results


def show_model_results(
    df: pd.DataFrame,
    gt: dict,
    config: dict,
    model: BayesianMMM | None,
) -> None:
    """Parameter recovery, decomposition, and model fit."""
    st.title("Model Results")
    channels = config["data"]["channels"]

    if model is None:
        st.error("No fitted trace found. Run `adelon-train` to generate the trace.")
        return

    # Parameter recovery table — delegate to model
    st.subheader("Parameter Recovery")
    st.markdown(
        "Comparing posterior estimates to the known ground truth "
        "values used to generate the synthetic data."
    )

    summary_df = model.summary(ground_truth=gt)
    display_cols = [
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "true_value",
        "recovery_error_pct",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    st.dataframe(summary_df[display_cols], use_container_width=True)

    st.markdown("---")

    # Contribution waterfall
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue Decomposition (Posterior)")
        contrib_df = model.get_channel_contributions()

        contributions = {}
        for ch in channels:
            contributions[_label(ch)] = float(contrib_df[f"{ch}_contribution"].sum())

        total_media = sum(contributions.values())
        total_revenue = float(df["revenue"].sum())
        contributions["Baseline + Other"] = total_revenue - total_media

        fig = plot_contribution_waterfall(contributions, total_revenue)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Media Share of Revenue")
        media_pct = total_media / total_revenue * 100
        gt_media_pct = (
            gt.get("derived", {}).get("revenue_decomposition_pct", {}).get("media", 0)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Estimated", "Ground Truth"],
                y=[media_pct, gt_media_pct],
                marker_color=["#002D40", "#4F7942"],
                text=[
                    f"{media_pct:.1f}%",
                    f"{gt_media_pct:.1f}%",
                ],
                textposition="auto",
            )
        )
        fig.update_layout(
            yaxis_title="Media % of Revenue",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model fit — use model method
    st.subheader("Model Fit: Actual vs Predicted")
    preds = model.get_posterior_predictions()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("R-squared", f"{preds['r_squared']:.4f}")
    with col2:
        st.metric("MAPE", f"{preds['mape']:.2f}%")

    fig = plot_model_fit(
        df["revenue"].values,
        preds["predicted_mean"],
        preds["predicted_lower"],
        preds["predicted_upper"],
        dates=df["date"].values,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual diagnostics
    st.subheader("Residual Diagnostics")
    fig = plot_residual_diagnostics(preds["residuals"], dates=df["date"].values)
    st.plotly_chart(fig, use_container_width=True)


# Page 4: Response Curves


def show_response_curves(
    df: pd.DataFrame,
    gt: dict,
    config: dict,
    model: BayesianMMM | None,
) -> None:
    """Media response curves and ROAS."""
    st.title("Media Response Curves")
    channels = config["data"]["channels"]

    if model is None:
        st.error("No fitted trace found. Run `adelon-train` to generate the trace.")
        return

    st.markdown(
        "Response curves show the relationship between daily spend "
        "and incremental revenue per channel. The shaded region is "
        "the 94% credible interval from the posterior."
    )

    # Use model method directly
    response_curves = model.get_response_curves()

    # All channels together
    fig = plot_response_curves(response_curves)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Individual channel exploration
    st.subheader("Channel Explorer")
    selected = st.selectbox(
        "Select Channel",
        channels,
        format_func=_label,
        key="rc_channel",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_response_curves(response_curves, channels=[selected])
        fig.update_layout(title=f"{_label(selected)} Response Curve")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rc = response_curves[selected]
        max_spend = float(df[f"{selected}_spend"].max())
        avg_spend = float(df[f"{selected}_spend"].mean())

        st.markdown("### Spend Levels")
        st.write(f"**Avg daily**: {_fmt(avg_spend)}")
        st.write(f"**Max daily**: {_fmt(max_spend)}")

        # Find contribution at current avg spend
        idx = np.searchsorted(rc["spend"].values, avg_spend) - 1
        idx = max(0, min(idx, len(rc) - 1))
        contrib_at_avg = rc["contribution_mean"].iloc[idx]
        st.write(f"**Revenue at avg spend**: {_fmt(contrib_at_avg)}")

        # ROAS at current spend
        if avg_spend > 0:
            marginal_roas = contrib_at_avg / avg_spend
            st.write(f"**Marginal ROAS**: {marginal_roas:.2f}x")

    # ROAS comparison — use model method
    st.markdown("---")
    st.subheader("Return on Ad Spend (ROAS)")

    roas_data = model.get_roas()
    fig = plot_roas_comparison(roas_data)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(roas_data, use_container_width=True)


# Page 5: Budget Optimizer


def show_budget_optimizer(
    df: pd.DataFrame,
    gt: dict,
    config: dict,
    model: BayesianMMM | None,
) -> None:
    """Budget allocation optimization."""
    st.title("Budget Optimizer")
    channels = config["data"]["channels"]

    if model is None:
        st.error("No fitted trace found. Run `adelon-train` to generate the trace.")
        return

    st.markdown(
        "Explore how reallocating budget across channels affects projected "
        "revenue. Choose between the **Greedy Marginal ROI** algorithm or the "
        "**Constrained Optimizer** (SciPy SLSQP) which supports per-channel "
        "minimum and maximum spend bounds."
    )

    response_curves = model.get_response_curves()
    spend_cols = config["data"]["spend_cols"]
    current_daily_budget = float(df[spend_cols].sum(axis=1).mean())

    # Budget slider
    st.subheader("Total Daily Budget")
    budget = st.slider(
        "Adjust total daily budget",
        min_value=int(current_daily_budget * 0.5),
        max_value=int(current_daily_budget * 2.0),
        value=int(current_daily_budget),
        step=1000,
        format="$%d",
    )

    # Optimizer selector
    st.subheader("Optimization Method")
    method = st.radio(
        "Select optimizer",
        options=["Greedy Marginal ROI", "Constrained Optimization"],
        horizontal=True,
    )

    # Current average daily allocation
    current_alloc = {ch: float(df[f"{ch}_spend"].mean()) for ch in channels}

    # Per-channel bounds (only shown for constrained mode)
    min_spend: dict[str, float] = {}
    max_spend: dict[str, float] = {}
    if method == "Constrained Optimization":
        st.subheader("Per-Channel Spend Bounds")
        st.caption(
            "Set minimum (contractual floors) and maximum (platform caps) daily "
            "spend per channel. Defaults: min = $0, max = total budget."
        )
        cols = st.columns(len(channels))
        for col, ch in zip(cols, channels):
            with col:
                st.markdown(f"**{_label(ch)}**")
                ch_min = st.number_input(
                    "Min ($)",
                    min_value=0,
                    max_value=int(budget),
                    value=0,
                    step=500,
                    key=f"min_{ch}",
                )
                ch_max = st.number_input(
                    "Max ($)",
                    min_value=0,
                    max_value=int(budget),
                    value=int(budget),
                    step=500,
                    key=f"max_{ch}",
                )
                min_spend[ch] = float(ch_min)
                max_spend[ch] = float(ch_max)

    # Run the selected optimizer
    optimal_alloc: dict[str, float] | None = None
    greedy_alloc: dict[str, float] | None = None
    error_msg: str | None = None

    if method == "Greedy Marginal ROI":
        greedy_alloc = greedy_budget_allocate(
            response_curves, budget, current_allocation=current_alloc
        )
        optimal_alloc = greedy_alloc
    else:
        greedy_alloc = greedy_budget_allocate(
            response_curves, budget, current_allocation=current_alloc
        )
        try:
            optimal_alloc = optimize_constrained(
                response_curves,
                budget,
                min_spend=min_spend or None,
                max_spend=max_spend or None,
            )
        except OptimizationInfeasibleError as exc:
            error_msg = str(exc)
        except ValueError as exc:
            error_msg = str(exc)

    if error_msg:
        st.error(f"Optimization failed: {error_msg}")
        return

    assert optimal_alloc is not None

    # Optimal allocation chart
    fig = plot_budget_optimizer(optimal_alloc, response_curves, budget)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Current vs optimal comparison
    st.subheader("Current vs Optimal Allocation")

    comp_rows = []
    for ch in channels:
        curr = current_alloc[ch]
        opt = optimal_alloc[ch]
        diff = opt - curr
        row: dict[str, str] = {
            "Channel": _label(ch),
            "Current ($)": f"{curr:,.0f}",
            "Optimal ($)": f"{opt:,.0f}",
            "Change ($)": f"{diff:+,.0f}",
            "Change (%)": (f"{diff / curr * 100:+.1f}%" if curr > 0 else "N/A"),
        }
        if method == "Constrained Optimization":
            row["Min ($)"] = f"{min_spend.get(ch, 0):,.0f}"
            row["Max ($)"] = f"{max_spend.get(ch, budget):,.0f}"
        comp_rows.append(row)
    st.dataframe(
        pd.DataFrame(comp_rows).set_index("Channel"),
        use_container_width=True,
    )

    # Revenue comparison
    st.markdown("---")
    st.subheader("Projected Revenue Impact")

    current_rev = estimate_revenue(response_curves, current_alloc)
    optimal_rev = estimate_revenue(response_curves, optimal_alloc)
    lift = optimal_rev - current_rev
    lift_pct = lift / current_rev * 100 if current_rev > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Daily Media Revenue", _fmt(current_rev))
    with col2:
        st.metric("Optimal Daily Media Revenue", _fmt(optimal_rev))
    with col3:
        st.metric("Revenue Lift", _fmt(lift), f"{lift_pct:+.1f}%")

    # Comparison view: show both methods side-by-side (constrained mode only)
    if method == "Constrained Optimization" and greedy_alloc is not None:
        st.markdown("---")
        st.subheader("Method Comparison: Constrained vs Greedy")
        greedy_rev = estimate_revenue(response_curves, greedy_alloc)
        diff_rev = optimal_rev - greedy_rev
        diff_rev_pct = diff_rev / greedy_rev * 100 if greedy_rev > 0 else 0

        cmp_rows = []
        for ch in channels:
            cmp_rows.append(
                {
                    "Channel": _label(ch),
                    "Greedy ($)": f"{greedy_alloc[ch]:,.0f}",
                    "Constrained ($)": f"{optimal_alloc[ch]:,.0f}",
                    "Difference ($)": f"{optimal_alloc[ch] - greedy_alloc[ch]:+,.0f}",
                }
            )
        st.dataframe(
            pd.DataFrame(cmp_rows).set_index("Channel"),
            use_container_width=True,
        )

        col_g, col_c, col_d = st.columns(3)
        with col_g:
            st.metric("Greedy Revenue", _fmt(greedy_rev))
        with col_c:
            st.metric("Constrained Revenue", _fmt(optimal_rev))
        with col_d:
            st.metric("Constrained vs Greedy", _fmt(diff_rev), f"{diff_rev_pct:+.1f}%")


# Page 6: MCMC Diagnostics


def show_mcmc_diagnostics(
    config: dict,
    model: BayesianMMM | None,
) -> None:
    """MCMC convergence diagnostics."""
    st.title("MCMC Diagnostics")
    channels = config["data"]["channels"]

    if model is None:
        st.error("No fitted trace found. Run `adelon-train` to generate the trace.")
        return

    import arviz as az

    trace = model._trace

    st.markdown(
        "Diagnostics for assessing MCMC convergence and sampling "
        "quality. All parameters should have R-hat < 1.01, high "
        "ESS, and zero divergences."
    )

    var_names = [
        "adstock_alpha",
        "saturation_K",
        "saturation_S",
        "beta_media",
        "intercept",
        "trend_coef",
        "sigma",
    ]

    # Summary statistics
    st.subheader("Convergence Summary")
    summary_df = az.summary(trace, var_names=var_names, round_to=4)

    max_rhat = summary_df["r_hat"].max()
    min_ess_bulk = summary_df["ess_bulk"].min()
    min_ess_tail = summary_df["ess_tail"].min()
    n_divergences = int(trace.sample_stats["diverging"].values.sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "normal" if max_rhat < 1.01 else "inverse"
        st.metric("Max R-hat", f"{max_rhat:.4f}", delta_color=color)
    with col2:
        st.metric("Min ESS (bulk)", f"{min_ess_bulk:.0f}")
    with col3:
        st.metric("Min ESS (tail)", f"{min_ess_tail:.0f}")
    with col4:
        color = "normal" if n_divergences == 0 else "inverse"
        st.metric("Divergences", str(n_divergences), delta_color=color)

    st.markdown("---")

    # Full summary table
    st.subheader("Parameter Summary")
    display_cols = [
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "mcse_mean",
        "mcse_sd",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    st.dataframe(summary_df[display_cols], use_container_width=True)

    st.markdown("---")

    # Trace plots
    st.subheader("Trace Plots")
    trace_params = st.multiselect(
        "Select parameters to plot",
        var_names,
        default=["adstock_alpha", "beta_media"],
    )

    if trace_params:
        fig = plot_trace_diagnostics(trace, trace_params, channels=channels)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sampling info
    st.subheader("Sampling Configuration")
    mcmc_cfg = config["mcmc"]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Draws**: {mcmc_cfg['draws']}")
        st.write(f"**Tune**: {mcmc_cfg['tune']}")
    with col2:
        st.write(f"**Chains**: {mcmc_cfg['chains']}")
        st.write(f"**Target Accept**: {mcmc_cfg['target_accept']}")

    n_chains = mcmc_cfg["chains"]
    n_draws = mcmc_cfg["draws"]
    st.write(f"**Total posterior samples**: {n_chains * n_draws:,}")


if __name__ == "__main__":
    main()
