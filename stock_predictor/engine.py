from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from .data import PriceDataProvider
from .features import (
    FEATURE_COLUMNS,
    build_supervised_data,
    latest_feature_row,
)
from .modeling import (
    MODEL_NAMES,
    SignalClassificationError,
    evaluate_models,
    evaluate_signal_classifier,
    fit_regressor,
    fit_signal_classifier,
)
from .signals import classify_signal, validate_threshold


@dataclass(frozen=True)
class Forecast:
    ticker: str
    as_of: str
    current_close: float
    predicted_next_close: float
    predicted_return_pct: float
    signal: str
    model: str
    threshold_pct: float
    evaluation: dict[str, dict[str, float | int]] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelForecast:
    model: str
    predicted_next_close: float
    predicted_return_pct: float
    signal: str
    mae_vs_baseline: str
    directional_accuracy_inference: str


@dataclass(frozen=True)
class SignalClassifierForecast:
    signal: str
    confidence_pct: float
    evaluation: dict[str, float | int]


@dataclass(frozen=True)
class ModelReport:
    ticker: str
    as_of: str
    current_close: float
    threshold_pct: float
    training_samples: int
    forecasts: tuple[ModelForecast, ...]
    evaluation: dict[str, dict[str, float | int]]
    signal_classifier: SignalClassifierForecast | None = None
    signal_classifier_error: str | None = None


def create_forecast(
    ticker: str,
    history: pd.DataFrame,
    model_name: str = "ridge",
    threshold: float = 0.01,
    n_splits: int = 5,
    compare_models: bool = False,
) -> Forecast:
    validate_threshold(threshold)

    training_data = build_supervised_data(history)
    if training_data.empty:
        raise ValueError(
            "Not enough complete history to train; at least 51 trading sessions "
            "are required."
        )

    evaluation = None
    if compare_models:
        evaluation = evaluate_models(training_data, MODEL_NAMES, n_splits=n_splits)
    else:
        evaluation = evaluate_models(training_data, (model_name,), n_splits=n_splits)

    estimator = fit_regressor(model_name, training_data)
    feature_row = latest_feature_row(history)
    predicted_return = float(
        estimator.predict(feature_row.loc[:, list(FEATURE_COLUMNS)])[0]
    )
    current_close = float(history["Close"].iloc[-1])
    predicted_close = current_close * (1 + predicted_return)

    return Forecast(
        ticker=ticker,
        as_of=pd.Timestamp(history.index[-1]).date().isoformat(),
        current_close=current_close,
        predicted_next_close=predicted_close,
        predicted_return_pct=predicted_return * 100,
        signal=classify_signal(predicted_return, threshold),
        model=model_name,
        threshold_pct=threshold * 100,
        evaluation=evaluation,
    )


def create_model_report(
    ticker: str,
    history: pd.DataFrame,
    model_names: tuple[str, ...],
    threshold: float = 0.01,
    n_splits: int = 5,
    include_signal_classifier: bool = False,
) -> ModelReport:
    validate_threshold(threshold)
    selected_models = tuple(dict.fromkeys(model_names))
    if not selected_models:
        raise ValueError("Select at least one model to create a report.")

    training_data = build_supervised_data(history)
    if training_data.empty:
        raise ValueError(
            "Not enough complete history to train; at least 51 trading sessions "
            "are required."
        )
    evaluation = evaluate_models(training_data, selected_models, n_splits=n_splits)
    baseline_mae = evaluation["zero_return_baseline"]["mae"]
    feature_row = latest_feature_row(history).loc[:, list(FEATURE_COLUMNS)]
    current_close = float(history["Close"].iloc[-1])

    forecasts: list[ModelForecast] = []
    for model_name in selected_models:
        estimator = fit_regressor(model_name, training_data)
        predicted_return = float(estimator.predict(feature_row)[0])
        if not math.isfinite(predicted_return):
            raise ValueError(f"{model_name} produced a non-finite forecast.")

        model_metrics = evaluation[model_name]
        mae = model_metrics["mae"]
        if mae < baseline_mae:
            mae_inference = "Walk-forward MAE was lower than the zero-return baseline."
        elif mae > baseline_mae:
            mae_inference = "Walk-forward MAE was higher than the zero-return baseline."
        else:
            mae_inference = "Walk-forward MAE matched the zero-return baseline."

        accuracy = model_metrics["directional_accuracy"]
        if accuracy > 0.5:
            accuracy_inference = "Directional accuracy exceeded 50% in these folds."
        elif accuracy < 0.5:
            accuracy_inference = "Directional accuracy was below 50% in these folds."
        else:
            accuracy_inference = "Directional accuracy was 50% in these folds."

        forecasts.append(
            ModelForecast(
                model=model_name,
                predicted_next_close=current_close * (1 + predicted_return),
                predicted_return_pct=predicted_return * 100,
                signal=classify_signal(predicted_return, threshold),
                mae_vs_baseline=mae_inference,
                directional_accuracy_inference=accuracy_inference,
            )
        )

    signal_classifier = None
    signal_classifier_error = None
    if include_signal_classifier:
        try:
            classifier_evaluation = evaluate_signal_classifier(
                training_data, n_splits=n_splits, threshold=threshold
            )
            classifier = fit_signal_classifier(training_data, threshold=threshold)
            probabilities = classifier.predict_proba(feature_row)[0]
            class_index = int(probabilities.argmax())
            signal_classifier = SignalClassifierForecast(
                signal=str(classifier.classes_[class_index]),
                confidence_pct=float(probabilities[class_index] * 100),
                evaluation=classifier_evaluation,
            )
        except SignalClassificationError as error:
            signal_classifier_error = str(error)

    return ModelReport(
        ticker=ticker,
        as_of=pd.Timestamp(history.index[-1]).date().isoformat(),
        current_close=current_close,
        threshold_pct=threshold * 100,
        training_samples=len(training_data),
        forecasts=tuple(forecasts),
        evaluation=evaluation,
        signal_classifier=signal_classifier,
        signal_classifier_error=signal_classifier_error,
    )


def fetch_and_forecast(
    ticker: str,
    provider: PriceDataProvider,
    start: str | None = None,
    end: str | None = None,
    model_name: str = "ridge",
    threshold: float = 0.01,
    n_splits: int = 5,
    compare_models: bool = False,
) -> Forecast:
    history = provider.fetch_history(ticker, start=start, end=end)
    return create_forecast(
        ticker=ticker,
        history=history,
        model_name=model_name,
        threshold=threshold,
        n_splits=n_splits,
        compare_models=compare_models,
    )
