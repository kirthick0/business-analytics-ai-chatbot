"""CSV and Excel loading helpers with conservative column cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pandas.errors


def _unique_column_names(columns: list[object]) -> list[str]:
    """Return trimmed, non-empty, unique string column names."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(columns, start=1):
        name = str(value).strip() or f"Column_{index}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count + 1}")
    return names


def load_dataset(source: BinaryIO | str | Path, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV/XLS/XLSX file, rejecting empty or unsupported datasets."""
    name = filename or getattr(source, "name", str(source))
    extension = Path(name).suffix.lower()
    try:
        if extension == ".csv":
            try:
                frame = pd.read_csv(source, encoding="utf-8-sig")
            except UnicodeDecodeError:
                if hasattr(source, "seek"):
                    source.seek(0)
                frame = pd.read_csv(source, encoding="latin-1")
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(source)
        else:
            raise ValueError("Please upload a CSV, XLS, or XLSX file.")
    except (ValueError, OSError, ImportError, pandas.errors.ParserError) as exc:
        raise ValueError(f"Could not read {name}: {exc}") from exc

    frame.columns = _unique_column_names(list(frame.columns))
    unnamed = [column for column in frame if column.lower().startswith("unnamed:")]
    frame = frame.drop(columns=[c for c in unnamed if frame[c].isna().all()])
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The uploaded file contains no usable rows or columns.")
    return frame


def load_sample_dataset() -> pd.DataFrame:
    """Load the repository's included sales example."""
    sample_path = Path(__file__).resolve().parents[1] / "sample_sales.csv"
    if not sample_path.exists():
        raise FileNotFoundError("sample_sales.csv is missing from the project folder.")
    return load_dataset(sample_path)