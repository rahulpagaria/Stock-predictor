from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET_COLUMN


MODEL_NAMES = ("linear", "ridge", "lasso", "xgboost")


def create_regressor(model_name: str) -> Any:
    name = model_name.lower()
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "lasso":
        return make_pipeline(
            StandardScaler(), Lasso(alpha=0.001, max_iter=10_000)
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is optional. Install it with "
                "`pip install -e '.[xgboost]'` to use this model."
            ) from error
        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(
        f"Unknown model {model_name!r}; choose from: {', '.join(MODEL_NAMES)}."
    )


def _validate_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in (*FEATURE_COLUMNS, TARGET_COLUMN) if column not in data]
    if missing:
        raise ValueError("Training data is missing columns: " + ", ".join(missing))
    if len(data) < 2:
        raise ValueError("At least two labeled trading sessions are required.")
    return data.loc[:, list(FEATURE_COLUMNS)], data[TARGET_COLUMN]


def fit_regressor(model_name: str, data: pd.DataFrame) -> Any:
    features, target = _validate_training_data(data)
    estimator = create_regressor(model_name)
    estimator.fit(features, target)
    return estimator


def evaluate_models(
    data: pd.DataFrame,
    model_names: Iterable[str],
    n_splits: int = 5,
) -> dict[str, dict[str, float | int]]:
    features, target = _validate_training_data(data)
    if n_splits < 2:
        raise ValueError("Walk-forward evaluation requires at least two splits.")
    if len(data) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} labeled sessions for walk-forward evaluation; "
            f"got {len(data)}."
        )

    split_metrics: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in model_names
    }
    if not split_metrics:
        raise ValueError("Select at least one model to evaluate.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_indices, test_indices in splitter.split(features):
        x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
        y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]
        for model_name in split_metrics:
            estimator = create_regressor(model_name)
            estimator.fit(x_train, y_train)
            predicted = np.asarray(estimator.predict(x_test), dtype=float).reshape(-1)
            actual = y_test.to_numpy(dtype=float)
            split_metrics[model_name].append(
                (
                    float(mean_absolute_error(actual, predicted)),
                    float(np.sqrt(mean_squared_error(actual, predicted))),
                    float(np.mean(np.sign(actual) == np.sign(predicted))),
                )
            )

    results: dict[str, dict[str, float | int]] = {}
    naive_predictions = np.zeros(len(target), dtype=float)
    naive_metrics: list[tuple[float, float, float]] = []
    for _, test_indices in TimeSeriesSplit(n_splits=n_splits).split(features):
        actual = target.iloc[test_indices].to_numpy(dtype=float)
        predicted = naive_predictions[test_indices]
        naive_metrics.append(
            (
                float(mean_absolute_error(actual, predicted)),
                float(np.sqrt(mean_squared_error(actual, predicted))),
                float(np.mean(np.sign(actual) == np.sign(predicted))),
            )
        )
    split_metrics["zero_return_baseline"] = naive_metrics

    for name, metrics in split_metrics.items():
        values = np.asarray(metrics, dtype=float)
        results[name] = {
            "mae": float(values[:, 0].mean()),
            "rmse": float(values[:, 1].mean()),
            "directional_accuracy": float(values[:, 2].mean()),
            "folds": len(metrics),
            "samples": len(data),
        }
    return results
