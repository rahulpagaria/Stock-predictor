from __future__ import annotations

import base64
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stock_predictor.data import YahooFinanceProvider
from stock_predictor.engine import ModelReport, create_model_report
from stock_predictor.features import add_technical_features
from stock_predictor.modeling import MODEL_NAMES


TICKER_SUGGESTIONS = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "BTC-USD": "Bitcoin",
    "GC=F": "Gold futures",
    "GOOGL": "Alphabet",
    "INFY.NS": "Infosys",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "RELIANCE.NS": "Reliance Industries",
    "SI=F": "Silver futures",
    "SPY": "S&P 500 ETF",
    "TCS.NS": "Tata Consultancy Services",
    "TSLA": "Tesla",
}


st.set_page_config(
    page_title="Ticker Predictor",
    page_icon="assets/ticker-predictor-logo.svg",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        --brand-forest: #103d35;
        --brand-forest-light: #1c594a;
        --brand-gold: #d8b66a;
        --brand-ink: #1c2a26;
        --brand-muted: #65756e;
        --brand-paper: #f5f7f4;
    }
    [data-testid="stAppViewContainer"] { background: var(--brand-paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] > div { padding-top: 1.2rem; }
    .block-container { max-width: 1440px; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--brand-ink); letter-spacing: -0.035em; }
    [data-testid="stCaptionContainer"] { color: var(--brand-muted); }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e4e9e4; border-radius: 16px;
        padding: 1rem 1.15rem; box-shadow: 0 5px 18px rgba(16, 61, 53, .045);
    }
    [data-testid="stSidebar"] { background: #edf2ed; }
    [data-testid="stSidebar"] h2 { color: var(--brand-forest); }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px; font-weight: 650; min-height: 2.8rem;
        background: var(--brand-forest); border-color: var(--brand-forest);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: var(--brand-forest-light); border-color: var(--brand-forest-light);
        color: white;
    }
    [data-testid="stTextInput"] input {
        min-height: 3rem; border-radius: 10px; border-color: #cbd6ce;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8e2; border-radius: 22px; background: white;
        box-shadow: 0 12px 36px rgba(16, 61, 53, .06);
    }
    .brand-header { display: flex; align-items: center; gap: .9rem; margin-bottom: 1rem; }
    .brand-title { color: var(--brand-forest); font-size: 1.45rem; line-height: 1.1; font-weight: 750; }
    .brand-owner { color: var(--brand-muted); font-size: .79rem; margin-top: .3rem; }
    .hero-eyebrow {
        color: #b28e42; text-transform: uppercase; letter-spacing: .16em;
        font-size: .74rem; font-weight: 750; margin: .6rem 0 .7rem;
    }
    .hero-title {
        color: var(--brand-forest); font-size: clamp(2.2rem, 4vw, 3.35rem);
        line-height: 1.04; letter-spacing: -.055em; font-weight: 760; margin: 0 0 .75rem;
    }
    .hero-copy { color: #56665f; font-size: 1.03rem; line-height: 1.7; max-width: 37rem; }
    .section-kicker {
        color: #7d8c83; text-transform: uppercase; letter-spacing: .12em;
        font-size: .73rem; font-weight: 700;
    }
    .report-footer {
        border-top: 1px solid #e1e7e1; margin-top: 2.6rem; padding-top: 1.2rem;
        color: #748179; font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_path = Path(__file__).parent / "assets" / "ticker-predictor-logo.svg"
logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <div class="brand-header">
      <img src="data:image/svg+xml;base64,{logo_data}" alt="Ticker Predictor logo"
           width="54" height="54" style="border-radius:14px">
      <div>
        <div class="brand-title">Ticker Predictor</div>
        <div class="brand-owner">An independent market research tool by Rahul Pagaria · Vruksh Enterprises</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Report settings")
    history_years = st.selectbox("History window", options=(1, 2, 5, 10), index=2)
    selected_models = st.multiselect(
        "Statistical techniques",
        options=MODEL_NAMES,
        default=["linear", "ridge", "lasso"],
        help="XGBoost is optional and may require the OpenMP runtime on macOS.",
    )
    threshold_percent = st.number_input(
        "Flat-signal threshold (%)",
        min_value=0.0,
        max_value=25.0,
        value=1.0,
        step=0.25,
    )
    folds = st.slider("Walk-forward evaluation folds", min_value=2, max_value=10, value=5)

with st.container(border=True):
    hero_copy, hero_art = st.columns([1.08, 0.92], vertical_alignment="center")
    with hero_copy:
        st.markdown(
            """
            <div class="hero-eyebrow">Clear signals. Better context.</div>
            <div class="hero-title">What market are<br>you watching today?</div>
            <div class="hero-copy">
              Enter a ticker to explore its price history, compare statistical
              forecasts, and see what the validation data says.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("ticker_search", border=False):
            ticker = st.selectbox(
                "Enter a ticker",
                options=tuple(TICKER_SUGGESTIONS),
                index=None,
                placeholder="Search a company or ticker — or type any ticker",
                help=(
                    "Type to filter suggestions, or enter any Yahoo Finance ticker "
                    "and press Enter."
                ),
                format_func=lambda symbol: (
                    f"{symbol} · {TICKER_SUGGESTIONS[symbol]}"
                    if symbol in TICKER_SUGGESTIONS
                    else symbol
                ),
                accept_new_options=True,
                filter_mode="fuzzy",
            )
            run_report = st.form_submit_button(
                "Explore this ticker  →", type="primary", width="stretch"
            )
    with hero_art:
        hero_path = Path(__file__).parent / "assets" / "market-hero.svg"
        hero_data = base64.b64encode(hero_path.read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <img src="data:image/svg+xml;base64,{hero_data}"
                 alt="Market momentum chart illustration"
                 style="display:block;width:40%;height:auto;margin:0 auto;opacity:.5">
            """,
            unsafe_allow_html=True,
        )

if run_report:
    st.session_state.pop("stock_report", None)
    st.session_state.pop("stock_history", None)
    symbol = (ticker or "").strip()
    if not symbol:
        st.error("Enter a ticker symbol.")
    elif not selected_models:
        st.error("Select at least one statistical technique.")
    else:
        today = date.today()
        start = today - timedelta(days=365 * history_years)
        end = today + timedelta(days=1)
        with st.spinner(f"Downloading {symbol} history and evaluating models..."):
            try:
                prices = YahooFinanceProvider().fetch_history(
                    symbol, start=start.isoformat(), end=end.isoformat()
                )
                report = create_model_report(
                    ticker=symbol,
                    history=prices,
                    model_names=tuple(selected_models),
                    threshold=threshold_percent / 100,
                    n_splits=folds,
                )
                st.session_state["stock_report"] = report
                st.session_state["stock_history"] = prices
            except (ValueError, RuntimeError) as error:
                st.error(str(error))

report = st.session_state.get("stock_report")
history = st.session_state.get("stock_history")
if not isinstance(report, ModelReport) or not isinstance(history, pd.DataFrame):
    if not run_report:
        st.markdown(
            """
            <div class="report-footer">
              Pick a ticker to begin. Your report includes interactive charts,
              model comparisons and plain-language validation notes.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

st.markdown(
    f'<div class="section-kicker">Ticker report &nbsp; / &nbsp; {report.as_of}</div>',
    unsafe_allow_html=True,
)
st.subheader(report.ticker)
st.caption(
    f"{report.training_samples:,} labeled sessions · "
    f"{len(report.evaluation) - 1} models · "
    f"BUY/SELL threshold ±{report.threshold_pct:.2f}%"
)

price_columns = st.columns(3)
price_columns[0].metric("Latest close", f"{report.current_close:,.2f}")
best_model = min(report.evaluation, key=lambda name: report.evaluation[name]["mae"])
price_columns[1].metric("Lowest walk-forward MAE", best_model)
price_columns[2].metric(
    "Zero-return baseline MAE",
    f"{report.evaluation['zero_return_baseline']['mae']:.4f}",
)

features = add_technical_features(history)
st.subheader("Historical price and indicators")
indicator_options = ["SMA 5", "SMA 20", "SMA 50", "EMA 12", "EMA 26"]
indicators = st.multiselect(
    "Overlay moving averages", indicator_options, default=["SMA 20", "SMA 50"]
)
indicator_columns = {
    "SMA 5": "sma_5",
    "SMA 20": "sma_20",
    "SMA 50": "sma_50",
    "EMA 12": "ema_12",
    "EMA 26": "ema_26",
}
price_chart = go.Figure()
price_chart.add_trace(
    go.Scatter(x=features.index, y=features["Close"], name="Close", line={"width": 2})
)
for label in indicators:
    column = indicator_columns[label]
    price_chart.add_trace(
        go.Scatter(x=features.index, y=features[column], name=label, line={"width": 1})
    )
price_chart.update_layout(
    height=480,
    xaxis_title="Trading session",
    yaxis_title="Price",
    hovermode="x unified",
    legend={"orientation": "h"},
    margin={"t": 20, "b": 10},
)
st.plotly_chart(price_chart, width="stretch")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("#### RSI (14)")
    rsi_chart = go.Figure()
    rsi_chart.add_trace(go.Scatter(x=features.index, y=features["rsi_14"], name="RSI 14"))
    rsi_chart.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70")
    rsi_chart.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="30")
    rsi_chart.update_yaxes(range=[0, 100], title="RSI")
    rsi_chart.update_layout(height=320, hovermode="x unified", margin={"t": 20, "b": 10})
    st.plotly_chart(rsi_chart, width="stretch")

with chart_right:
    st.markdown("#### Volume")
    volume_chart = px.bar(features, x=features.index, y="Volume", labels={"x": "Trading session"})
    volume_chart.update_layout(height=320, yaxis_title="Volume", showlegend=False)
    st.plotly_chart(volume_chart, width="stretch")

st.subheader("Next-session estimates by technique")
forecast_rows = [
    {
        "Technique": forecast.model,
        "Predicted close": forecast.predicted_next_close,
        "Predicted return (%)": forecast.predicted_return_pct,
        "Signal": forecast.signal,
    }
    for forecast in report.forecasts
]
forecast_frame = pd.DataFrame(forecast_rows)
forecast_left, forecast_right = st.columns(2)
with forecast_left:
    st.dataframe(
        forecast_frame.style.format(
            {"Predicted close": "{:,.2f}", "Predicted return (%)": "{:+.3f}%"}
        ),
        width="stretch",
        hide_index=True,
    )
with forecast_right:
    return_chart = px.bar(
        forecast_frame,
        x="Technique",
        y="Predicted return (%)",
        color="Signal",
        color_discrete_map={"BUY": "#2ca02c", "SELL": "#d62728", "FLAT": "#7f7f7f"},
        title="Predicted next-session return",
    )
    return_chart.add_hline(y=report.threshold_pct, line_dash="dash", annotation_text="Buy threshold")
    return_chart.add_hline(y=-report.threshold_pct, line_dash="dash", annotation_text="Sell threshold")
    return_chart.update_layout(height=360, margin={"t": 45, "b": 10})
    st.plotly_chart(return_chart, width="stretch")

st.subheader("Walk-forward model comparison")
st.caption(
    "Scores are averages across chronological held-out folds. MAE/RMSE are in "
    "decimal return units; lower is better. Direction accuracy is the share of "
    "held-out sessions with the correct return sign. Directional accuracy is "
    "shown for fitted models only; a zero-return baseline has no directional call."
)
metric_rows = [
    {"Technique": model, **metrics}
    for model, metrics in report.evaluation.items()
]
metrics_frame = pd.DataFrame(metric_rows).set_index("Technique")
fitted_metrics_frame = metrics_frame.drop(index="zero_return_baseline")
metric_left, metric_right = st.columns(2)
with metric_left:
    st.markdown("#### Mean absolute error")
    mae_frame = metrics_frame.reset_index()
    mae_chart = px.bar(mae_frame, x="Technique", y="mae", title="Lower is better")
    mae_chart.update_layout(height=340, showlegend=False)
    st.plotly_chart(mae_chart, width="stretch")
with metric_right:
    st.markdown("#### Directional accuracy")
    direction_frame = fitted_metrics_frame.reset_index()
    direction_chart = px.bar(
        direction_frame,
        x="Technique",
        y="directional_accuracy",
        title="Held-out return direction",
    )
    direction_chart.add_hline(y=0.5, line_dash="dash", annotation_text="50% reference")
    direction_chart.update_yaxes(tickformat=".0%")
    direction_chart.update_layout(height=340, showlegend=False)
    st.plotly_chart(direction_chart, width="stretch")
st.dataframe(
    metrics_frame.copy().assign(
        directional_accuracy=metrics_frame["directional_accuracy"].where(
            metrics_frame.index != "zero_return_baseline"
        )
    ).style.format(
        {"mae": "{:.5f}", "rmse": "{:.5f}", "directional_accuracy": "{:.1%}"},
        na_rep="n/a",
    ),
    width="stretch",
)

st.subheader("Model inferences")
for forecast in report.forecasts:
    metrics = report.evaluation[forecast.model]
    with st.expander(f"{forecast.model.title()} · {forecast.signal}"):
        st.write(
            f"Forecast close: **{forecast.predicted_next_close:,.2f}** "
            f"({forecast.predicted_return_pct:+.3f}%)."
        )
        st.write(
            f"Walk-forward MAE: **{metrics['mae']:.5f}**; RMSE: "
            f"**{metrics['rmse']:.5f}**; directional accuracy: "
            f"**{metrics['directional_accuracy']:.1%}**."
        )
        st.write(forecast.mae_vs_baseline)
        st.write(forecast.directional_accuracy_inference)
        st.caption(
            "These are historical validation results, not guarantees of future "
            "performance. The zero-return baseline is not a directional trading rule."
        )

st.caption(
    "Prices and forecasts are shown in the ticker's quoted currency. "
    "The next trading date is not inferred because market holidays vary."
)
st.markdown(
    """
    <div class="report-footer">
      Ticker Predictor &nbsp;·&nbsp; Rahul Pagaria &nbsp;·&nbsp; Vruksh Enterprises
      <br>Forecasts are statistical estimates for research and education, not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
