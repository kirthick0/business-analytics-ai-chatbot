"""Reusable descriptive, correlation, outlier, and SQLite analysis functions."""

from __future__ import annotations

import sqlite3

import pandas as pd


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Include numeric and categorical columns in a readable summary."""
    if frame.empty:
        return pd.DataFrame()
    return frame.describe(include="all").transpose().reset_index(names="Column")


def numeric_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    return numeric.corr(numeric_only=True) if numeric.shape[1] >= 2 else pd.DataFrame()


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count IQR outliers in each numeric column."""
    records: list[dict[str, object]] = []
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        count = int(((values < low) | (values > high)).sum())
        records.append(
            {"Column": column, "Outlier rows": count, "Outlier %": round(count / len(values) * 100, 2),
             "Lower bound": low, "Upper bound": high}
        )
    return pd.DataFrame(records)


def create_sqlite_connection(frame: pd.DataFrame) -> sqlite3.Connection:
    """Create a session-local SQLite database containing the current dataset."""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    frame.to_sql("uploaded_data", connection, if_exists="replace", index=False)
    return connection


def sqlite_dataset_preview(connection: sqlite3.Connection, limit: int = 10) -> pd.DataFrame:
    """Read a bounded preview from the session's SQLite dataset table."""
    safe_limit = max(1, min(int(limit), 100))
    return pd.read_sql_query(f"SELECT * FROM uploaded_data LIMIT {safe_limit}", connection)