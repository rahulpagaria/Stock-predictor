import unittest

import numpy as np
import pandas as pd

from stock_predictor.features import (
    FEATURE_COLUMNS,
    build_supervised_data,
)
from stock_predictor.modeling import (
    SignalClassificationError,
    evaluate_models,
    evaluate_signal_classifier,
    fit_regressor,
)
from tests.test_features import sample_history


def volatile_history(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.02, rows)
    close = 100 * np.cumprod(1 + returns)
    index = pd.bdate_range("2023-01-02", periods=rows)
    close_series = pd.Series(close, index=index)
    return pd.DataFrame(
        {
            "Open": close_series * 0.999,
            "High": close_series * 1.01,
            "Low": close_series * 0.99,
            "Close": close_series,
            "Volume": rng.integers(10_000, 30_000, rows),
        },
        index=index,
    )


class ModelingTests(unittest.TestCase):
    def test_fit_and_walk_forward_evaluation(self) -> None:
        training_data = build_supervised_data(sample_history(140))
        estimator = fit_regressor("ridge", training_data)
        predictions = estimator.predict(
            training_data.iloc[-1:][list(FEATURE_COLUMNS)]
        )
        self.assertTrue(np.isfinite(predictions).all())

        metrics = evaluate_models(training_data, ("ridge",), n_splits=3)
        self.assertIn("zero_return_baseline", metrics)
        self.assertEqual(metrics["ridge"]["folds"], 3)
        self.assertGreater(metrics["ridge"]["samples"], 0)

    def test_nonlinear_regressors_fit_and_evaluate(self) -> None:
        training_data = build_supervised_data(sample_history(140))
        for model_name in ("random_forest", "knn", "mlp"):
            with self.subTest(model=model_name):
                estimator = fit_regressor(model_name, training_data)
                prediction = estimator.predict(
                    training_data.iloc[-1:][list(FEATURE_COLUMNS)]
                )
                self.assertTrue(np.isfinite(prediction).all())

                metrics = evaluate_models(training_data, (model_name,), n_splits=3)
                self.assertEqual(metrics[model_name]["folds"], 3)

    def test_knn_handles_small_walk_forward_training_folds(self) -> None:
        training_data = build_supervised_data(sample_history(60))

        metrics = evaluate_models(training_data, ("knn",), n_splits=5)

        self.assertEqual(metrics["knn"]["folds"], 5)

    def test_logistic_signal_classifier_is_evaluated_separately(self) -> None:
        training_data = build_supervised_data(volatile_history())

        metrics = evaluate_signal_classifier(
            training_data, n_splits=3, threshold=0.005
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertGreaterEqual(metrics["folds"], 1)

    def test_logistic_signal_classifier_requires_multiple_classes(self) -> None:
        training_data = build_supervised_data(sample_history(140))

        with self.assertRaisesRegex(SignalClassificationError, "at least two"):
            evaluate_signal_classifier(training_data, n_splits=3, threshold=0.01)

    def test_evaluation_rejects_too_few_samples(self) -> None:
        training_data = build_supervised_data(sample_history(60))
        with self.assertRaisesRegex(ValueError, "Need more than"):
            evaluate_models(training_data, ("ridge",), n_splits=len(training_data))


if __name__ == "__main__":
    unittest.main()
