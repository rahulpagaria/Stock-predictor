import unittest

from stock_predictor.engine import create_forecast, create_model_report
from tests.test_features import sample_history


class ForecastWorkflowTests(unittest.TestCase):
    def test_forecast_includes_price_return_signal_and_evaluation(self) -> None:
        history = sample_history(140)

        forecast = create_forecast(
            ticker="TEST",
            history=history,
            model_name="ridge",
            threshold=0.005,
            n_splits=3,
        )
        output = forecast.to_dict()

        self.assertEqual(output["ticker"], "TEST")
        self.assertAlmostEqual(
            forecast.predicted_next_close,
            forecast.current_close * (1 + forecast.predicted_return_pct / 100),
        )
        self.assertIn(forecast.signal, {"BUY", "SELL", "FLAT"})
        self.assertIn("ridge", forecast.evaluation)

    def test_interactive_report_compares_each_selected_model(self) -> None:
        history = sample_history(140)

        report = create_model_report(
            ticker="TEST",
            history=history,
            model_names=("linear", "ridge"),
            threshold=0.005,
            n_splits=3,
        )

        self.assertEqual([item.model for item in report.forecasts], ["linear", "ridge"])
        self.assertIn("zero_return_baseline", report.evaluation)
        for item in report.forecasts:
            self.assertIn(item.signal, {"BUY", "SELL", "FLAT"})
            self.assertTrue(item.mae_vs_baseline)
            self.assertTrue(item.directional_accuracy_inference)


if __name__ == "__main__":
    unittest.main()
