"""Business summaries and recommendations based strictly on observed values."""

from __future__ import annotations

import pandas as pd


def _column_by_words(frame: pd.DataFrame, words: tuple[str, ...]) -> str | None:
    for column in frame.select_dtypes(include="number").columns:
        if any(word in str(column).lower() for word in words):
            return column
    return None


def generate_business_insights(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No business findings are available because the dataset has no rows."]
    findings = [f"The dataset contains {len(frame):,} records across {frame.shape[1]:,} columns."]
    revenue = _column_by_words(frame, ("revenue", "sales", "turnover"))
    profit = _column_by_words(frame, ("profit", "margin", "earnings"))
    if revenue:
        total = frame[revenue].sum()
        findings.append(f"Observed total {revenue}: {total:,.2f} across {frame[revenue].notna().sum():,} non-missing rows.")
        category = next(
            (
                c for c in frame.columns
                if not pd.api.types.is_numeric_dtype(frame[c])
                and frame[c].nunique(dropna=True) > 1
                and not any(token in str(c).lower() for token in ("date", "time", "id", "number"))
            ),
            None,
        )
        if category:
            ranked = frame.groupby(category, dropna=False)[revenue].sum().sort_values(ascending=False)
            if not ranked.empty:
                findings.append(f"{category} with the highest summed {revenue}: {ranked.index[0]} ({ranked.iloc[0]:,.2f}).")
                findings.append(f"Recommendation: review the lowest-performing {category} groups and compare their product mix, pricing, and volume before reallocating resources.")
    if profit:
        findings.append(f"Observed total {profit}: {frame[profit].sum():,.2f}; average per populated row: {frame[profit].mean():,.2f}.")
    missing = int(frame.isna().sum().sum())
    if missing:
        findings.append(f"Data quality: {missing:,} missing cells remain. Validate whether these are genuinely unknown or can be safely filled before modelling.")
    else:
        findings.append("Data quality: no missing cells were detected in the uploaded dataset.")
    duplicates = int(frame.duplicated().sum())
    findings.append(f"Data quality: {duplicates:,} exact duplicate row(s) detected; inspect before deleting because repeated transactions may be valid.")
    return findings