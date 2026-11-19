"""Guarded, optional regression, classification, and customer-clustering models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _features(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    candidates = [c for c in frame.columns if c != target]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in candidates if not pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique(dropna=True) <= 100]
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                       ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def train_regression(frame: pd.DataFrame, target: str, model_name: str = "Random Forest") -> dict[str, Any]:
    if target not in frame or not pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError("Choose a numeric target column for regression.")
    if model_name not in {"Random Forest", "Linear Regression"}:
        raise ValueError("Choose Random Forest or Linear Regression.")
    numeric, categorical = _features(frame, target)
    if not numeric and not categorical:
        raise ValueError("At least one usable feature column is required.")
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if len(data) < 8:
        raise ValueError("At least 8 rows with a target value are needed for a train/test evaluation.")
    X, y = data[numeric + categorical], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    estimator = LinearRegression() if model_name == "Linear Regression" else RandomForestRegressor(n_estimators=150, random_state=42)
    pipeline = Pipeline([("preprocessor", _preprocessor(numeric, categorical)), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return {"model": pipeline, "metrics": {"R²": float(r2_score(y_test, predictions)),
              "MAE": float(mean_absolute_error(y_test, predictions)), "RMSE": float(rmse)},
            "predictions": pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions}), "features": numeric + categorical}


def train_classification(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    if target not in frame or pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError("Choose a categorical target column for classification.")
    numeric, categorical = _features(frame, target)
    if not numeric and not categorical:
        raise ValueError("At least one usable feature column is required.")
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if len(data) < 10 or data[target].nunique() < 2:
        raise ValueError("Classification needs at least 10 labeled rows and two target classes.")
    X, y = data[numeric + categorical], data[target].astype(str)
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    pipeline = Pipeline([("preprocessor", _preprocessor(numeric, categorical)),
                         ("model", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return {"model": pipeline, "metrics": {"Accuracy": float(accuracy_score(y_test, predictions))},
            "predictions": pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions}), "features": numeric + categorical}


def segment_customers(frame: pd.DataFrame, feature_columns: list[str], clusters: int = 3) -> pd.DataFrame:
    if len(feature_columns) < 2 or any(c not in frame for c in feature_columns):
        raise ValueError("Select at least two numeric features for customer segmentation.")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in feature_columns):
        raise ValueError("Select only numeric features for customer segmentation.")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna(how="all")
    if len(usable) < max(4, clusters):
        raise ValueError(f"Need at least {max(4, clusters)} rows with numeric features.")
    if not 2 <= clusters <= min(10, len(usable) - 1):
        raise ValueError("Choose a cluster count from 2 through the smaller of 10 or rows minus 1.")
    prepared = SimpleImputer(strategy="median").fit_transform(usable)
    if not np.isfinite(prepared).all():
        raise ValueError("The selected features contain no usable numeric values.")
    scaled = StandardScaler().fit_transform(prepared)
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(scaled)
    result = frame.loc[usable.index].copy()
    result["Customer Segment"] = labels + 1
    return result