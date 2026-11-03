from __future__ import annotations

import argparse
import json
import math
import sys

from .agent_api import HttpAgentApiAdapter
from .data import YahooFinanceProvider
from .engine import fetch_and_forecast
from .modeling import MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast a ticker's next trading-session closing price."
    )
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL or MSFT")
    parser.add_argument("--start", help="History start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        help="History end date (YYYY-MM-DD; yfinance treats this as exclusive)",
    )
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default="ridge", help="Forecast estimator"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Absolute predicted-return threshold for BUY/SELL (default: 0.01)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of chronological walk-forward evaluation splits",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Evaluate linear, ridge, lasso, and XGBoost before fitting the selected model",
    )
    parser.add_argument(
        "--agent-api-url",
        help="Optional HTTP JSON endpoint to receive the forecast (explicit opt-in)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("Provide both --start and --end, or neither.")
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be a finite, non-negative value.")
    if args.splits < 2:
        parser.error("--splits must be at least 2.")

    try:
        forecast = fetch_and_forecast(
            ticker=args.ticker,
            provider=YahooFinanceProvider(),
            start=args.start,
            end=args.end,
            model_name=args.model,
            threshold=args.threshold,
            n_splits=args.splits,
            compare_models=args.compare_models,
        )
        output = forecast.to_dict()
        if args.agent_api_url:
            adapter = HttpAgentApiAdapter(args.agent_api_url)
            output["agent_api_response"] = dict(
                adapter.send_prediction(forecast.to_dict())
            )
        print(json.dumps(output, indent=2))
        return 0
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
