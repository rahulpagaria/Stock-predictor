from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET_COLUMN
from .signals import classify_signal, validate_threshold


MODEL_NAMES = (
    "linear",
    "ridge",
    "lasso",
    "random_forest",
    "knn",
    "mlp",
    "xgboost",
)
MODEL_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "mlp": "Neural Network (MLP)",
    "xgboost": "XGBoost",
}


class SignalClassificationError(ValueError):
    """Raised when the available data cannot train a multi-class signal model."""


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


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
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=1,
        )
    if name == "knn":
        return make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                max_iter=500,
                random_state=42,
            ),
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


def _fit_regressor(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> Any:
    estimator = create_regressor(model_name)
    if model_name.lower() == "knn":
        estimator.set_params(
            kneighborsregressor__n_neighbors=min(5, len(features))
        )
    estimator.fit(features, target)
    return estimator


def fit_regressor(model_name: str, data: pd.DataFrame) -> Any:
    features, target = _validate_training_data(data)
    return _fit_regressor(model_name, features, target)


def _signal_targets(target: pd.Series, threshold: float) -> np.ndarray:
    return np.asarray(
        [classify_signal(float(value), threshold) for value in target], dtype=str
    )


def create_signal_classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42),
    )


def fit_signal_classifier(data: pd.DataFrame, threshold: float = 0.01) -> Any:
    _, target = _validate_training_data(data)
    validate_threshold(threshold)
    labels = _signal_targets(target, threshold)
    if np.unique(labels).size < 2:
        raise SignalClassificationError(
            "Logistic signal classification needs at least two observed signal "
            "classes at the selected threshold."
        )
    classifier = create_signal_classifier()
    classifier.fit(data.loc[:, list(FEATURE_COLUMNS)], labels)
    return classifier


def evaluate_signal_classifier(
    data: pd.DataFrame,
    n_splits: int = 5,
    threshold: float = 0.01,
) -> dict[str, float | int]:
    features, target = _validate_training_data(data)
    validate_threshold(threshold)
    if n_splits < 2:
        raise ValueError("Walk-forward evaluation requires at least two splits.")
    if len(data) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} labeled sessions for walk-forward evaluation; "
            f"got {len(data)}."
        )

    labels = _signal_targets(target, threshold)
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    for train_indices, test_indices in TimeSeriesSplit(n_splits=n_splits).split(features):
        train_labels = labels[train_indices]
        if np.unique(train_labels).size < 2:
            continue
        classifier = create_signal_classifier()
        classifier.fit(features.iloc[train_indices], train_labels)
        actual_parts.append(labels[test_indices])
        predicted_parts.append(classifier.predict(features.iloc[test_indices]))

    if not actual_parts:
        raise SignalClassificationError(
            "Walk-forward logistic classification is unavailable: no training "
            "fold contains at least two signal classes at the selected threshold."
        )

    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "folds": len(actual_parts),
        "samples": len(actual),
    }


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
            estimator = _fit_regressor(model_name, x_train, y_train)
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
