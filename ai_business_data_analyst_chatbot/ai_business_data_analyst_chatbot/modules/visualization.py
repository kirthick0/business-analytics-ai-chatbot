"""Plotly chart constructors used by the dashboard and analyst pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = ["#20c997", "#5271ff", "#ffb84d", "#ff6b81", "#8b7bff", "#35a7ff"]


def make_chart(frame: pd.DataFrame, chart_type: str, x: str, y: str | None = None):
    if frame.empty or x not in frame.columns:
        return None
    if chart_type == "Pie chart":
        values = y if y in frame.columns else None
        return px.pie(frame, names=x, values=values, hole=0.48, color_discrete_sequence=PALETTE)
    if chart_type == "Line chart":
        return px.line(frame, x=x, y=y, markers=True, color_discrete_sequence=PALETTE)
    if chart_type == "Scatter plot":
        return px.scatter(frame, x=x, y=y, color_discrete_sequence=PALETTE, trendline="ols" if len(frame) >= 3 else None)
    if chart_type == "Histogram":
        return px.histogram(frame, x=x, color_discrete_sequence=PALETTE, marginal="box")
    if chart_type == "Bar chart":
        return px.bar(frame, x=x, y=y, color=x, color_discrete_sequence=PALETTE)
    return None


def correlation_heatmap(correlation: pd.DataFrame):
    if correlation.empty:
        return None
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(), x=correlation.columns.astype(str), y=correlation.index.astype(str),
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True, text=correlation.round(2).to_numpy(),
            texttemplate="%{text}", hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title="Numeric feature correlation", height=500)
    return figure