"""Generate HTML report from backtest results."""

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from .config import Config
from .metrics import compute_drawdown, compute_performance_stats

logger = logging.getLogger(__name__)

# Common Plotly layout for dark theme
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1d27",
    plot_bgcolor="#1a1d27",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e6"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=370,
)


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


def _nav_chart(nav: pd.Series, benchmark_nav=None) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav.values, mode="lines",
        name="GMV Portfolio", line=dict(color="#6c8cff", width=2),
    ))
    if benchmark_nav is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_nav.index, y=benchmark_nav.values, mode="lines",
            name="Benchmark", line=dict(color="#8b8fa3", width=1.5, dash="dot"),
        ))
    fig.update_layout(**_LAYOUT, yaxis_title="Growth of $1")
    return _fig_html(fig)


def _dd_chart(nav: pd.Series) -> str:
    dd = compute_drawdown(nav)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100, mode="lines",
        fill="tozeroy", fillcolor="rgba(224,85,85,0.15)",
        line=dict(color="#e05555", width=1.5), name="Drawdown",
    ))
    fig.update_layout(**_LAYOUT, yaxis_title="Drawdown (%)")
    return _fig_html(fig)


def _vol_chart(returns: pd.Series, window: int = 126) -> str:
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values, mode="lines",
        line=dict(color="#f0a050", width=1.5), name=f"{window}-day Vol",
    ))
    fig.update_layout(**_LAYOUT, yaxis_title="Annualized Vol (%)")
    return _fig_html(fig)


def _weights_chart(weights_df: pd.DataFrame) -> str:
    fig = go.Figure()
    # Filter columns with nonzero weight at some point
    active = weights_df.columns[weights_df.sum() > 0]
    colors = [
        "#6c8cff", "#4caf82", "#f0a050", "#e05555", "#a78bfa",
        "#38bdf8", "#f472b6", "#facc15", "#34d399", "#fb923c",
        "#c084fc", "#22d3ee",
    ]
    for i, col in enumerate(active):
        fig.add_trace(go.Scatter(
            x=weights_df.index, y=weights_df[col].values * 100,
            mode="lines", stackgroup="one",
            name=col, line=dict(width=0),
            fillcolor=colors[i % len(colors)],
        ))
    fig.update_layout(**_LAYOUT, yaxis_title="Weight (%)", yaxis_range=[0, 100])
    return _fig_html(fig)


def build_report(results: dict, config: Config, output_dir: str = "outputs/reports"):
    """Build and save the HTML report.

    Parameters
    ----------
    results : dict from run_backtest
    config : Config object
    output_dir : folder for output HTML

    Returns
    -------
    str – path to generated report
    """
    nav = results["nav"]
    rets = results["returns"]
    weights_df = results["weights"]
    turnover = results["turnover"]
    bench_nav = results.get("benchmark_nav")

    # Stats
    stats = compute_performance_stats(rets, config.annualization, config.risk_free_rate)

    # Benchmark stats
    bench_stats = None
    if bench_nav is not None and len(bench_nav) > 10:
        bench_rets = bench_nav.pct_change().dropna()
        bench_stats = compute_performance_stats(bench_rets, config.annualization, config.risk_free_rate)

    # Charts
    nav_chart = _nav_chart(nav, bench_nav)
    dd_chart = _dd_chart(nav)
    vol_chart = _vol_chart(rets)
    weights_chart = _weights_chart(weights_df)

    # Latest weights
    last_w = weights_df.iloc[-1].sort_values(ascending=False)
    latest_weights = [(t, w) for t, w in last_w.items() if w > 0.001][:10]

    # Turnover summary
    avg_turnover = turnover.mean()
    total_tc_bps = (turnover * config.tc_bps).sum()
    n_rebalances = len(turnover)

    # Render template
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("report_template.html")

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = template.render(
        run_date=run_date,
        config=config,
        stats=stats,
        bench_stats=bench_stats,
        nav_chart=nav_chart,
        dd_chart=dd_chart,
        vol_chart=vol_chart,
        weights_chart=weights_chart,
        latest_weights=latest_weights,
        avg_turnover=avg_turnover,
        total_tc_bps=total_tc_bps,
        n_rebalances=n_rebalances,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report saved → %s", path)
    return path
