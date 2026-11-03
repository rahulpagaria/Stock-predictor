from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "Close",
    "daily_return",
    "sma_5",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "volume_sma_20",
    "volume_ratio_20",
    "volatility_20",
)
TARGET_COLUMN = "target_next_return"


def add_technical_features(history: pd.DataFrame) -> pd.DataFrame:
    required = {"Close", "Volume"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError("History is missing required columns: " + ", ".join(missing))

    frame = history.copy()
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=14, min_periods=14).mean()
    average_loss = losses.rolling(window=14, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)

    frame["daily_return"] = close.pct_change()
    frame["sma_5"] = close.rolling(window=5, min_periods=5).mean()
    frame["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    frame["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    frame["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    frame["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["rsi_14"] = 100 - (100 / (1 + relative_strength))
    frame.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100.0
    frame.loc[(average_loss == 0) & (average_gain == 0), "rsi_14"] = 50.0

    volume_average = volume.rolling(window=20, min_periods=20).mean()
    frame["volume_sma_20"] = volume_average
    frame["volume_ratio_20"] = volume / volume_average.replace(0, np.nan)
    frame["volatility_20"] = frame["daily_return"].rolling(
        window=20, min_periods=20
    ).std()
    return frame.replace([np.inf, -np.inf], np.nan)


def build_supervised_data(history: pd.DataFrame) -> pd.DataFrame:
    frame = add_technical_features(history)
    frame[TARGET_COLUMN] = frame["Close"].shift(-1) / frame["Close"] - 1
    return frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])


def latest_feature_row(history: pd.DataFrame) -> pd.DataFrame:
    frame = add_technical_features(history)
    usable = frame.dropna(subset=list(FEATURE_COLUMNS))
    if usable.empty:
        raise ValueError(
            "Not enough history to compute features; at least 50 trading sessions "
            "are required."
        )
    return usable.loc[:, list(FEATURE_COLUMNS)].iloc[[-1]]
