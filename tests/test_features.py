import unittest

import numpy as np
import pandas as pd

from stock_predictor.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_technical_features,
    build_supervised_data,
    latest_feature_row,
)


def sample_history(rows: int = 90) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.Series(100 + np.arange(rows) * 0.25, index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1_000 + np.arange(rows) * 3,
        },
        index=index,
    )


class FeatureTests(unittest.TestCase):
    def test_features_use_only_current_and_prior_sessions(self) -> None:
        history = sample_history()
        features = add_technical_features(history)
        first_valid_date = history.index[49]

        self.assertEqual(
            features.loc[first_valid_date, "sma_50"],
            history["Close"].iloc[:50].mean(),
        )
        self.assertEqual(features.loc[first_valid_date, "rsi_14"], 100.0)
        self.assertEqual(features.loc[first_valid_date, "sma_50"], features["sma_50"].iloc[49])

    def test_supervised_target_is_next_session_return(self) -> None:
        history = sample_history()
        supervised = build_supervised_data(history)
        target_date = supervised.index[0]
        next_date = history.index[history.index.get_loc(target_date) + 1]

        self.assertEqual(
            supervised.loc[target_date, TARGET_COLUMN],
            history.loc[next_date, "Close"] / history.loc[target_date, "Close"] - 1,
        )
        self.assertNotIn(history.index[-1], supervised.index)
        self.assertEqual(
            tuple(column for column in FEATURE_COLUMNS if column in supervised),
            FEATURE_COLUMNS,
        )

    def test_latest_feature_row_includes_latest_close(self) -> None:
        history = sample_history()
        latest = latest_feature_row(history)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.index[0], history.index[-1])
        self.assertEqual(latest.iloc[0]["Close"], history["Close"].iloc[-1])

    def test_short_history_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 50 trading sessions"):
            latest_feature_row(sample_history(40))


if __name__ == "__main__":
    unittest.main()
