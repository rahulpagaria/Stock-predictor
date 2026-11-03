import unittest

import numpy as np

from stock_predictor.features import FEATURE_COLUMNS, build_supervised_data
from stock_predictor.modeling import evaluate_models, fit_regressor
from tests.test_features import sample_history


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

    def test_evaluation_rejects_too_few_samples(self) -> None:
        training_data = build_supervised_data(sample_history(60))
        with self.assertRaisesRegex(ValueError, "Need more than"):
            evaluate_models(training_data, ("ridge",), n_splits=len(training_data))


if __name__ == "__main__":
    unittest.main()
