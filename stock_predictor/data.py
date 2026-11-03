from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class PriceDataProvider(Protocol):
    def fetch_history(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return daily OHLCV history for a ticker."""


class DataFetchError(RuntimeError):
    """Raised when a market-data source cannot provide usable history."""


def normalize_history(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise DataFetchError("The data provider returned no historical prices.")

    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).strip().title() for column in frame.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataFetchError(
            "Historical data is missing required columns: " + ", ".join(missing)
        )

    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    try:
        frame.index = pd.to_datetime(frame.index)
    except (TypeError, ValueError) as error:
        raise DataFetchError("Historical data has an invalid date index.") from error

    if frame.index.has_duplicates:
        raise DataFetchError("Historical data contains duplicate trading dates.")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_rows = np.isfinite(frame.loc[:, list(REQUIRED_COLUMNS)]).all(axis=1)
    frame = frame.loc[finite_rows].dropna(subset=list(REQUIRED_COLUMNS)).sort_index()
    if frame.empty:
        raise DataFetchError("Historical data has no complete OHLCV rows.")
    if (frame["Close"] <= 0).any():
        raise DataFetchError("Historical closing prices must be greater than zero.")
    if (frame["Volume"] < 0).any():
        raise DataFetchError("Historical volume cannot be negative.")
    return frame


class YahooFinanceProvider:
    def fetch_history(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        symbol = ticker.strip()
        if not symbol:
            raise ValueError("Ticker cannot be empty.")
        if (start is None) != (end is None):
            raise ValueError("Provide both start and end dates, or neither.")
        if start is not None and end is not None and start >= end:
            raise ValueError("Start date must be earlier than end date.")

        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError(
                "yfinance is required to download market history; install project "
                "dependencies with `pip install -e .`."
            ) from error

        options: dict[str, object] = {
            "tickers": symbol,
            "interval": "1d",
            "auto_adjust": True,
            "progress": False,
            "threads": False,
        }
        if start is None:
            options["period"] = "5y"
        else:
            options["start"] = start
            options["end"] = end

        history = yf.download(**options)
        try:
            return normalize_history(history)
        except DataFetchError as error:
            raise DataFetchError(
                f"Unable to load history for {symbol}: {error}"
            ) from error
