"""Dataset quality summaries and explicit, opt-in cleaning operations."""

from __future__ import annotations

import pandas as pd


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    """Return dataset quality facts without changing the source frame."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "memory_mb": float(frame.memory_usage(deep=True).sum() / (1024**2)),
        "column_details": pd.DataFrame(
            {
                "Column": frame.columns.astype(str),
                "Data type": [str(dtype) for dtype in frame.dtypes],
                "Missing": frame.isna().sum().values,
                "Missing %": (frame.isna().mean().values * 100).round(2),
                "Unique values": frame.nunique(dropna=True).values,
            }
        ),
    }


def clean_data(
    frame: pd.DataFrame,
    *,
    remove_duplicates: bool = False,
    missing_strategy: str = "Keep missing values",
) -> pd.DataFrame:
    """Return a cleaned copy using only operations explicitly selected by the user."""
    cleaned = frame.copy()
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if missing_strategy == "Drop rows with missing values":
        cleaned = cleaned.dropna()
    elif missing_strategy == "Fill numeric with median; text with mode":
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                median = cleaned[column].median()
                if pd.notna(median):
                    cleaned[column] = cleaned[column].fillna(median)
            else:
                modes = cleaned[column].mode(dropna=True)
                if not modes.empty:
                    cleaned[column] = cleaned[column].fillna(modes.iloc[0])
    elif missing_strategy != "Keep missing values":
        raise ValueError(f"Unsupported missing-value strategy: {missing_strategy}")
    return cleaned