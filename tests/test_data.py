import unittest

import pandas as pd

from stock_predictor.data import (
    DataFetchError,
    normalize_history,
    resolve_market_ticker,
)


class DataTests(unittest.TestCase):
    def test_resolves_indian_exchange_symbols(self) -> None:
        self.assertEqual(resolve_market_ticker("RELIANCE", "NSE India"), "RELIANCE.NS")
        self.assertEqual(resolve_market_ticker("RELIANCE", "BSE India"), "RELIANCE.BO")
        self.assertEqual(
            resolve_market_ticker("reliance.ns", "NSE India"), "RELIANCE.NS"
        )
        self.assertEqual(resolve_market_ticker("GC=F", "Global & futures"), "GC=F")

    def test_rejects_mismatched_indian_exchange_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "already a NS symbol"):
            resolve_market_ticker("RELIANCE.NS", "BSE India")
        with self.assertRaisesRegex(ValueError, "already a BO symbol"):
            resolve_market_ticker("RELIANCE.BO", "NSE India")

    def test_rejects_empty_market_ticker(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ticker cannot be empty"):
            resolve_market_ticker("  ", "NSE India")

    def test_normalizes_and_sorts_history(self) -> None:
        history = pd.DataFrame(
            {
                "open": [11, 10],
                "high": [12, 11],
                "low": [10, 9],
                "close": [11.5, 10.5],
                "volume": [100, 90],
            },
            index=["2024-01-03", "2024-01-02"],
        )

        normalized = normalize_history(history)

        self.assertEqual(list(normalized.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(normalized.index.is_monotonic_increasing)

    def test_rejects_empty_or_incomplete_data(self) -> None:
        with self.assertRaises(DataFetchError):
            normalize_history(pd.DataFrame())
        with self.assertRaisesRegex(DataFetchError, "missing required columns"):
            normalize_history(pd.DataFrame({"Close": [10]}))

    def test_discards_non_finite_rows(self) -> None:
        history = pd.DataFrame(
            {
                "Open": [10, 11],
                "High": [11, 12],
                "Low": [9, 10],
                "Close": [10.5, float("inf")],
                "Volume": [100, 110],
            },
            index=["2024-01-02", "2024-01-03"],
        )

        normalized = normalize_history(history)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized["Close"].iloc[0], 10.5)


if __name__ == "__main__":
    unittest.main()
