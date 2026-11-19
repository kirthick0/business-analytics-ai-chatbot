"""A deterministic natural-language analyst that computes answers from data."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


METRIC_WORDS = {
    "revenue": ("revenue", "sales", "sale", "turnover", "income", "average order value"),
    "profit": ("profit", "margin", "earnings"),
    "quantity": ("quantity", "units", "volume", "items sold"),
    "orders": ("orders", "order count", "transactions", "records"),
}
GROUP_WORDS = ("product", "category", "region", "customer", "segment", "city", "state", "country", "channel", "month", "year")


def _normal(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def _metric_column(question: str, frame: pd.DataFrame) -> str | None:
    q = _normal(question)
    numeric = _numeric_columns(frame)
    for words in METRIC_WORDS.values():
        if any(word in q for word in words):
            candidates = [c for c in numeric if any(word in _normal(c) for word in words)]
            if candidates:
                return candidates[0]
            if words is METRIC_WORDS["orders"]:
                return None
            return None
    for column in numeric:
        normalized = _normal(column)
        if normalized and normalized in q:
            return column
    return None


def _group_column(question: str, frame: pd.DataFrame) -> str | None:
    q = _normal(question)
    categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    for word in GROUP_WORDS:
        if word in q:
            match = next((c for c in categorical if word in _normal(c)), None)
            if match:
                return match
    for column in categorical:
        normalized = _normal(column)
        if normalized and normalized in q:
            return column
    if any(word in q for word in ("which", "top", "bottom", "highest", "lowest", "best", "worst")):
        return next((c for c in categorical if frame[c].nunique(dropna=True) > 1), None)
    return None


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return column
    for column in frame.columns:
        if any(token in _normal(column) for token in ("date", "time", "month")):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().sum() >= max(2, int(len(frame) * 0.5)):
                return column
    return None


def _format_number(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:,.2f}" if number % 1 else f"{number:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def answer_question(question: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Answer supported questions with computed results; otherwise explain the gap."""
    q = question.strip()
    normalized = _normal(q)
    if frame.empty:
        return {"answer": "The dataset has no rows to analyse.", "table": None, "title": "No data"}

    if any(word in normalized for word in ("missing", "null", "blank", "incomplete")):
        missing = frame.isna().sum().rename_axis("Column").reset_index(name="Missing rows")
        total = int(missing["Missing rows"].sum())
        return {"answer": f"I counted {total:,} missing cell(s) across {len(frame):,} rows. See the table for each column.",
                "table": missing[missing["Missing rows"] > 0], "title": "Missing-value analysis"}

    if any(word in normalized for word in ("correlation", "correlate", "related")):
        corr = frame.select_dtypes(include="number").corr(numeric_only=True)
        if corr.shape[1] < 2:
            return {"answer": "Correlation needs at least two numeric columns; this dataset does not have enough.", "table": None, "title": "Correlation"}
        return {"answer": "Here is the Pearson correlation matrix calculated from the numeric columns.",
                "table": corr.reset_index(names="Column"), "title": "Correlation matrix", "kind": "correlation"}

    date_column = _date_column(frame)
    if date_column and any(word in normalized for word in ("monthly", "month", "growth", "trend", "over time", "by date")):
        metric = _metric_column(q, frame)
        if metric is None:
            return {"answer": "I found a date column but could not identify a matching numeric measure. Ask for a trend of a numeric column, such as revenue or profit.", "table": None, "title": "Trend analysis"}
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        values = pd.DataFrame({"Period": dates.dt.to_period("M").astype(str), metric: frame[metric]})
        trend = values.dropna().groupby("Period", as_index=False)[metric].sum().sort_values("Period")
        trend["Growth %"] = (trend[metric].pct_change() * 100).round(2)
        latest = trend.iloc[-1]
        return {"answer": f"Monthly {metric} for the latest period ({latest['Period']}) was {_format_number(latest[metric])}. Growth is calculated against the previous month; the full series is shown below.",
                "table": trend, "title": f"Monthly {metric} trend", "kind": "trend", "x": "Period", "y": metric}

    count_request = any(word in normalized for word in ("how many", "number of", "count", "total records", "total rows"))
    metric = _metric_column(q, frame)
    if count_request and metric is None:
        id_col = next((c for c in frame.columns if any(t in _normal(c) for t in ("order id", "transaction id", "customer id"))), None)
        wants_unique = "unique" in normalized
        if "customer" in normalized:
            customer_col = next((c for c in frame.columns if "customer" in _normal(c)), id_col)
        else:
            customer_col = id_col
        if wants_unique or "customer" in normalized:
            if customer_col:
                count = int(frame[customer_col].nunique(dropna=True))
                label = f"unique {customer_col}"
            else:
                count = len(frame)
                label = "records"
        else:
            count = len(frame)
            label = "records"
        return {"answer": f"The dataset contains {count:,} {label}.", "table": None, "title": "Record count"}

    if metric is None and any(word in normalized for word in ("top", "highest", "best", "worst", "lowest", "underperform")):
        metric = next(
            (column for words in (("revenue", "sales", "turnover"), ("profit", "earnings"))
             for column in _numeric_columns(frame) if any(word in _normal(column) for word in words)),
            None,
        )

    if metric is None:
        if any(word in normalized for word in ("summary", "describe", "overview", "major trends", "business trends")):
            numeric = frame.select_dtypes(include="number")
            if numeric.empty:
                return {"answer": "I can provide a data overview, but there are no numeric columns for business measures.", "table": frame.describe(include="all").transpose().reset_index(names="Column"), "title": "Dataset overview"}
            rows = [{"Measure": c, "Total": numeric[c].sum(), "Average": numeric[c].mean(), "Minimum": numeric[c].min(), "Maximum": numeric[c].max()} for c in numeric]
            return {"answer": "These summaries are calculated directly from each numeric column. Choose a specific metric and grouping for a focused business answer.", "table": pd.DataFrame(rows), "title": "Numeric business summary"}
        available = ", ".join(map(str, _numeric_columns(frame))) or "none"
        return {"answer": f"I could not identify a numeric measure in your question. Available numeric columns: {available}. Ask for a specific measure (for example, 'total Revenue by Product').", "table": None, "title": "Question needs a measure"}

    group = _group_column(q, frame)
    ranking = any(word in normalized for word in ("highest", "lowest", "top", "bottom", "best", "worst", "underperform"))
    operation = "mean" if any(w in normalized for w in ("average", "mean")) else "sum"
    if group:
        summary = frame.groupby(group, dropna=False)[metric].agg(operation).reset_index(name=f"{operation.title()} {metric}")
        if ranking or any(word in normalized for word in ("show", "list", "compare", "by")):
            summary = summary.sort_values(summary.columns[-1], ascending=any(w in normalized for w in ("lowest", "bottom", "worst", "underperform")))
            limit_match = re.search(r"\btop\s+(\d+)\b", normalized)
            limit = min(int(limit_match.group(1)), 100) if limit_match else (10 if "top" in normalized else len(summary))
            summary = summary.head(limit).reset_index(drop=True)
            if not summary.empty and ranking:
                best = summary.iloc[0]
                direction = "lowest" if any(w in normalized for w in ("lowest", "bottom", "worst", "underperform")) else "highest"
                text = f"{best[group]} has the {direction} {operation} {metric}: {_format_number(best.iloc[-1])}. The ranked results are below."
            else:
                text = f"Here is {operation} {metric} grouped by {group}, calculated from {len(frame):,} source rows."
        else:
            text = f"Here is {operation} {metric} grouped by {group}, calculated from the uploaded dataset."
        return {"answer": text, "table": summary, "title": f"{operation.title()} {metric} by {group}", "kind": "bar", "x": group, "y": summary.columns[-1]}

    value = frame[metric].mean() if operation == "mean" else frame[metric].sum()
    operation_label = "average" if operation == "mean" else "total"
    if pd.isna(value):
        return {"answer": f"Column {metric} has no numeric values to calculate.", "table": None, "title": "No numeric values"}
    return {"answer": f"The {operation_label} {metric} is {_format_number(value)}, calculated from {frame[metric].notna().sum():,} non-missing values.",
            "table": pd.DataFrame([{f"{operation_label.title()} {metric}": value}]), "title": f"{operation_label.title()} {metric}"}