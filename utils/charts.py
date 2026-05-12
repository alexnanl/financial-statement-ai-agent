"""
Chart plotting module - interactive Plotly charts.
"""
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
from utils.ratios import is_percentage_metric


COLORS = {
    "primary": "#1f4e79",
    "accent": "#c00000",
    "good": "#2e7d32",
    "warn": "#ed6c02",
    "bad": "#c62828",
    "neutral": "#6b7280",
}

PALETTE = ["#1f4e79", "#c00000", "#2e7d32", "#ed6c02", "#7b1fa2", "#0277bd", "#5d4037"]


def plot_trend(trend_df: pd.DataFrame, metrics: List[str],
                title: str = "Trend Analysis") -> go.Figure:
    """
    Multi-metric trend chart.
    trend_df: rows = metrics, columns = years.
    """
    fig = go.Figure()

    # Extract year labels (force strings for discrete category axis)
    year_labels = []
    for c in trend_df.columns:
        if hasattr(c, "year"):
            year_labels.append(str(c.year))
        else:
            year_labels.append(str(c))

    for i, m in enumerate(metrics):
        if m not in trend_df.index:
            continue
        series = trend_df.loc[m]
        is_pct = is_percentage_metric(m)
        values = [v * 100 if (v is not None and is_pct) else v for v in series.values]

        fig.add_trace(go.Scatter(
            x=year_labels,
            y=values,
            mode="lines+markers",
            name=m,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"{m}<br>%{{x}}: %{{y:.2f}}{'%' if is_pct else ''}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["primary"])),
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    # Force X axis as category to avoid Plotly auto-inserting fractional years
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=year_labels,
        ticktext=year_labels,
        showgrid=False,
        showline=True,
        linecolor="#e5e7eb",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb")
    return fig


def plot_peer_comparison(compare_df: pd.DataFrame, metric: str) -> go.Figure:
    """Single-metric peer comparison bar chart."""
    if metric not in compare_df.index:
        return go.Figure()

    series = compare_df.loc[metric].dropna()
    is_pct = is_percentage_metric(metric)
    values = [v * 100 if is_pct else v for v in series.values]

    fig = go.Figure(go.Bar(
        x=list(series.index),
        y=values,
        marker=dict(
            color=values,
            colorscale=[[0, COLORS["bad"]], [0.5, COLORS["warn"]], [1, COLORS["good"]]],
            line=dict(color="white", width=1),
        ),
        text=[f"{v:.2f}{'%' if is_pct else ''}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"Peer Comparison: {metric}",
                    font=dict(size=16, color=COLORS["primary"])),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=40, r=40, t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")
    return fig


def plot_dupont_decomposition(dupont_history: Dict[int, Dict]) -> go.Figure:
    """
    DuPont three-factor breakdown across years.
    dupont_history: {year: {Net Margin, Asset Turnover, Equity Multiplier, ROE}}
    """
    years_raw = sorted(dupont_history.keys())
    years = [str(y) for y in years_raw]
    nm = [dupont_history[y].get("Net Margin") for y in years_raw]
    ato = [dupont_history[y].get("Asset Turnover") for y in years_raw]
    em = [dupont_history[y].get("Equity Multiplier") for y in years_raw]
    roe = [dupont_history[y].get("ROE (DuPont)") for y in years_raw]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=[v * 100 if v else None for v in nm],
                              name="Net Margin (%)", yaxis="y1",
                              line=dict(color=PALETTE[0], width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=years, y=ato,
                              name="Asset Turnover", yaxis="y2",
                              line=dict(color=PALETTE[1], width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=years, y=em,
                              name="Equity Multiplier", yaxis="y2",
                              line=dict(color=PALETTE[2], width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=years, y=[v * 100 if v else None for v in roe],
                              name="ROE (%)", yaxis="y1",
                              line=dict(color=PALETTE[3], width=3, dash="dash"),
                              mode="lines+markers"))

    fig.update_layout(
        title=dict(text="DuPont Analysis: ROE Three-Factor Decomposition",
                    font=dict(size=18, color=COLORS["primary"])),
        yaxis=dict(title="Percentage (%)", side="left", showgrid=True, gridcolor="#f3f4f6"),
        yaxis2=dict(title="Turnover / Multiplier", side="right", overlaying="y", showgrid=False),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=years,
        ticktext=years,
        title="Year",
    )
    return fig


def plot_dupont_waterfall(dupont: Dict, year: int) -> go.Figure:
    """Single-year DuPont three-factor waterfall."""
    nm = dupont.get("Net Margin") or 0
    ato = dupont.get("Asset Turnover") or 0
    em = dupont.get("Equity Multiplier") or 0
    roe = dupont.get("ROE (DuPont)") or 0

    fig = go.Figure()
    categories = [
        f"Net Margin<br>{nm*100:.2f}%",
        f"x Asset Turnover<br>{ato:.2f}",
        f"x Equity Multiplier<br>{em:.2f}",
        f"= ROE<br>{roe*100:.2f}%",
    ]
    values = [nm * 100, ato, em, roe * 100]
    colors_seq = [PALETTE[0], PALETTE[1], PALETTE[2], COLORS["accent"]]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(color=colors_seq),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"{year} DuPont Decomposition",
                    font=dict(size=16, color=COLORS["primary"])),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=60),
    )
    return fig


def plot_radar(compare_df: pd.DataFrame, metrics: List[str]) -> go.Figure:
    """Radar chart: target company vs. peer average."""
    available = [m for m in metrics if m in compare_df.index]
    if not available:
        return go.Figure()

    target_col = compare_df.columns[0]
    target_values = []
    peer_avg = []

    for m in available:
        v = compare_df.loc[m, target_col]
        target_values.append(v if pd.notna(v) else 0)
        peers = compare_df.loc[m].drop(target_col).dropna()
        peer_avg.append(peers.mean() if len(peers) > 0 else 0)

    all_per_metric = [list(compare_df.loc[m].dropna()) for m in available]

    target_norm = []
    peer_norm = []
    for i, m in enumerate(available):
        max_v = max(abs(v) for v in all_per_metric[i]) if all_per_metric[i] else 1
        target_norm.append(target_values[i] / max_v if max_v else 0)
        peer_norm.append(peer_avg[i] / max_v if max_v else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=target_norm + [target_norm[0]],
        theta=available + [available[0]],
        fill="toself",
        name=target_col.split(" ")[0],
        line=dict(color=COLORS["accent"], width=2),
        fillcolor="rgba(192,0,0,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=peer_norm + [peer_norm[0]],
        theta=available + [available[0]],
        fill="toself",
        name="Peer Average",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(31,78,121,0.15)",
    ))
    fig.update_layout(
        title=dict(text="Performance Radar (Normalized)",
                    font=dict(size=16, color=COLORS["primary"])),
        polar=dict(radialaxis=dict(visible=True, range=[-0.2, 1.1])),
        height=440,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
