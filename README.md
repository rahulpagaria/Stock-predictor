# Ticker Predictor

A small Python pipeline that fetches daily historical prices for a caller-provided
ticker, calculates technical features, and forecasts the next trading session's
closing price and close-to-close percentage move.

The forecast is an estimate, not investment advice or a guarantee. A `BUY`, `SELL`,
or `FLAT` label is derived from the configurable predicted-return threshold; it is
not a position-sizing or execution recommendation.

## Setup

Requires Python 3.10 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The linear, Ridge, and Lasso regressors are included in the base install. XGBoost
is optional:

```sh
python -m pip install -e '.[xgboost]'
```

On macOS, the XGBoost native library may also require the OpenMP runtime; with
Homebrew, install it using `brew install libomp`.

## Run a forecast

Fetches five years of daily history by default:

```sh
stock-predictor AAPL
```

Choose a model, custom date range, or signal threshold:

```sh
stock-predictor MSFT --start 2019-01-01 --end 2025-01-01 --model ridge --threshold 0.015
```

The end date follows yfinance semantics and is exclusive. Supply both dates or
neither. Forecast output is JSON and includes the as-of date, current close,
estimated next close, estimated percentage return, signal, and walk-forward
metrics (return MAE, RMSE, directional accuracy) compared with a zero-return
baseline. Metric values are expressed as decimal return fractions. The estimated
next session's calendar date is intentionally not guessed; at least 56 usable
trading sessions are required for the default five-split evaluation.

Evaluate the linear, Ridge, Lasso, and XGBoost candidates with chronological
walk-forward splits, then fit the chosen `--model` on all available labeled rows:

```sh
stock-predictor AAPL --compare-models --model xgboost
```

This requires the optional XGBoost extra. Evaluation data is never randomly
shuffled. Technical indicators are computed using information available at each
session close; the label is the following observed session's close-to-close return.
The predicted price is derived from the return forecast and the latest close so the
two reported outputs stay consistent.

## Agent API integration

An optional HTTP JSON endpoint can receive the completed forecast:

```sh
stock-predictor AAPL --agent-api-url https://agent.example/api/predictions
```

This sends a POST request with a JSON forecast body and expects a JSON object in
response. No API is called unless this option is supplied. The `AgentApiAdapter`
protocol is the replacement seam for a provider-specific client; adapt request and
response handling when the external agent's API contract is known.

## Structure

- `stock_predictor/data.py`: data-provider protocol, Yahoo Finance source, and OHLCV validation.
- `stock_predictor/features.py`: technical indicators, aligned next-session label, and latest features.
- `stock_predictor/modeling.py`: regression estimators and time-series cross-validation.
- `stock_predictor/engine.py`: forecast workflow and result shape.
- `stock_predictor/signals.py`: configurable buy/sell/flat threshold policy.
- `stock_predictor/agent_api.py`: generic opt-in HTTP adapter and protocol.
- `stock_predictor/cli.py`: command-line interface.

The forecast result is a plain dataclass/JSON payload suitable for a later service
or native iOS client. Keep customer authentication, storage, and mobile
distribution outside this Python-first prototype until those requirements are
defined.

## Interactive web report

Install the optional dashboard dependencies and start the local app:

```sh
python -m pip install -e '.[web]'
python -m streamlit run app.py
```

Search the ticker autocomplete in the welcome panel to choose a suggested symbol,
or type any Yahoo Finance ticker and press Enter. Then select a history window and one or more statistical techniques,
then generate a report. Ticker Predictor is presented by Rahul Pagaria and Vruksh Enterprises.
The dashboard plots the adjusted close with selectable
moving-average overlays, RSI, and volume; compares each selected model's next
close/return/signal; and charts walk-forward MAE and directional accuracy against
the zero-return baseline. Each model includes a short inference based on its
forecast and held-out metrics. XGBoost can be selected when its optional dependency
is installed; on macOS it may additionally require `brew install libomp`.

### Free hosting with Streamlit Community Cloud

The app is configured to run from the repository root with `app.py` and
`requirements.txt`. To publish a hosted dashboard:

1. Push this project to a GitHub repository under your account.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and
   authorize access to the repository.
3. Choose **Create app**, select the repository and `main` branch, and set the
   app file to `app.py`.
4. Deploy. Community Cloud will install packages from `requirements.txt`.

The deployed dashboard is accessible to anyone with its URL. The source
repository can remain private if the hosting account has access. No API keys or
secrets are required by the current app. Review provider terms and free-tier
limits before sharing broadly; ticker lookups fetch market data from Yahoo
Finance.

## Tests

```sh
python -m pip install -e '.[test]'
python -m pytest
```
