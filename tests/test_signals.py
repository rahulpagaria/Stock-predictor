import unittest

from stock_predictor.signals import classify_signal


class SignalTests(unittest.TestCase):
    def test_threshold_classifies_buy_sell_and_flat(self) -> None:
        self.assertEqual(classify_signal(0.02, 0.01), "BUY")
        self.assertEqual(classify_signal(-0.02, 0.01), "SELL")
        self.assertEqual(classify_signal(0.01, 0.01), "FLAT")

    def test_negative_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            classify_signal(0.1, -0.01)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            classify_signal(0.1, float("nan"))
        with self.assertRaisesRegex(ValueError, "Predicted return must be finite"):
            classify_signal(float("inf"), 0.01)


if __name__ == "__main__":
    unittest.main()
