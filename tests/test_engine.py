import unittest

from stock_predictor.engine import create_forecast, create_model_report
from stock_predictor.report_pdf import create_pdf_report, pdf_filename
from tests.test_features import sample_history
from tests.test_modeling import volatile_history


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

    def test_pdf_report_generates_pdf_with_safe_filename(self) -> None:
        history = sample_history(140)
        report = create_model_report(
            ticker="GC=F",
            history=history,
            model_names=("linear", "ridge"),
            threshold=0.005,
            n_splits=3,
        )

        pdf = create_pdf_report(report, history)

        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn(b"%%EOF", pdf[-1024:])
        self.assertEqual(pdf_filename("GC=F", "2024-07-12"), "ticker-predictor-GC-F-2024-07-12.pdf")

    def test_report_includes_separate_logistic_signal_classifier(self) -> None:
        report = create_model_report(
            ticker="TEST",
            history=volatile_history(),
            model_names=("random_forest",),
            threshold=0.005,
            n_splits=3,
            include_signal_classifier=True,
        )

        self.assertIsNotNone(report.signal_classifier)
        assert report.signal_classifier is not None
        self.assertIn(report.signal_classifier.signal, {"BUY", "SELL", "FLAT"})
        self.assertGreaterEqual(report.signal_classifier.confidence_pct, 0)
        self.assertTrue(create_pdf_report(report, volatile_history()).startswith(b"%PDF-"))

    def test_report_explains_when_logistic_classes_are_unavailable(self) -> None:
        report = create_model_report(
            ticker="TEST",
            history=sample_history(140),
            model_names=("ridge",),
            threshold=0.01,
            n_splits=3,
            include_signal_classifier=True,
        )

        self.assertIsNone(report.signal_classifier)
        self.assertIn("at least two signal classes", report.signal_classifier_error)


if __name__ == "__main__":
    unittest.main()
