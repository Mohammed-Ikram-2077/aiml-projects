"""Multi-tab Streamlit dashboard for financial sentiment research."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import DEFAULT_CONFIG, MODEL_NAME, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ROOT_DIR
from src.sentiment import finbert_scores, vader_scores


@st.cache_data(show_spinner=False)
def _read_csv(path: str, modified_at: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv(path: Path) -> pd.DataFrame:
    """Load data and invalidate the cache if a pipeline run rewrote the file."""
    return _read_csv(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_json(path: str, modified_at: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return _read_json(str(path), path.stat().st_mtime)


def ticker_file_key(ticker: str) -> str:
    """Map exchange symbols such as ``RELIANCE.NS`` to their cache filename."""
    return ticker.replace(".", "_")


def run_live_pipeline(ticker: str) -> subprocess.CompletedProcess[str]:
    """Run the existing collection/scoring pipeline for one selected ticker."""
    return subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--tickers",
            ticker,
            "--months",
            str(DEFAULT_CONFIG.months),
            "--force-collect",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def collect_live_data(ticker: str) -> None:
    """Run live collection, show actionable logs on failure, and reload on success."""
    try:
        with st.spinner(f"Collecting live data and scoring {ticker}. This may take a few minutes…"):
            result = run_live_pipeline(ticker)
    except subprocess.TimeoutExpired:
        st.error("The pipeline exceeded its 10-minute time limit. Try again later or run it locally for this ticker.")
        return
    except Exception as exc:
        st.error(f"Could not start the collection pipeline: {exc}")
        return

    if result.returncode == 0:
        try:
            run_summary = json.loads((PROCESSED_DIR / "pipeline_summary.json").read_text(encoding="utf-8"))
            counts = run_summary.get("tickers", {}).get(ticker, {})
            st.success(
                f"Finished {ticker}: {counts.get('news_rows', 0)} news headlines, "
                f"{counts.get('social_rows', 0)} Bluesky posts, "
                f"{counts.get('price_rows', 0)} price rows."
            )
        except (OSError, ValueError):
            st.success(f"Finished collecting and scoring {ticker}.")
        st.rerun()
    else:
        st.error("The pipeline did not complete. Check the logs below for the source error.")
        logs = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        with st.expander("Pipeline logs", expanded=True):
            st.code(logs[-12000:] or "No logs were returned.")


st.set_page_config(page_title="Financial Sentiment Lab", layout="wide")
st.title("AI-Based Financial Market Sentiment Analysis")
st.caption("Research dashboard: sentiment is noisy, correlations are not causal, and results are not financial advice.")

sentiment_paths = sorted(PROCESSED_DIR.glob("sentiment_*.csv"))
summary_path = PROCESSED_DIR / "pipeline_summary.json"
summary = load_json(summary_path) if summary_path.exists() else {}
latest_tickers = set(summary.get("tickers", {}))
available_tickers = [path.stem.removeprefix("sentiment_").replace("_", ".") for path in sentiment_paths]
tickers = [ticker for ticker in available_tickers if not latest_tickers or ticker in latest_tickers]
if not tickers:
    st.info("No pipeline outputs are stored in this deployment yet. Collect live data below to build the dashboard.")
    ticker_to_collect = st.selectbox("Ticker", DEFAULT_CONFIG.tickers)
    if st.button("Collect live data", type="primary"):
        collect_live_data(ticker_to_collect)
    st.stop()

with st.sidebar:
    refresh_requested = False
    selected_ticker = st.selectbox("Ticker", tickers)
    ticker_key = ticker_file_key(selected_ticker)
    sentiment_path = PROCESSED_DIR / f"sentiment_{ticker_key}.csv"
    data = load_csv(sentiment_path)
    source_choices = ["Both", "News", "Bluesky"]
    selected_source = st.selectbox("Source", source_choices)
    score_candidates = [col for col in ("finbert_score", "vader_score", "tfidf_score") if col in data and data[col].notna().any()]
    score_options = {"FinBERT": "finbert_score", "VADER": "vader_score", "TF-IDF baseline": "tfidf_score"}
    available_names = [name for name, col in score_options.items() if col in score_candidates]
    selected_model = st.selectbox("Sentiment model", available_names or ["VADER"])
    selected_score = score_options[selected_model]
    refresh_requested = st.button("Refresh live data", help="Fetch current headlines, Bluesky posts, and daily prices for this ticker.")

if refresh_requested:
    collect_live_data(selected_ticker)

if "timestamp_utc" in data:
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    sentiment_dates = data["timestamp_utc"].dropna().dt.date
    min_date = sentiment_dates.min() if not sentiment_dates.empty else pd.NaT
    max_date = sentiment_dates.max() if not sentiment_dates.empty else pd.NaT
else:
    min_date = max_date = pd.NaT

price_path = RAW_DIR / f"prices_{ticker_file_key(selected_ticker)}.csv"
if price_path.exists():
    prices = load_csv(price_path)
else:
    prices = pd.DataFrame()
if not prices.empty and "date" in prices:
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    min_date = min(min_date, prices["date"].min()) if not pd.isna(min_date) else prices["date"].min()
    max_date = max(max_date, prices["date"].max()) if not pd.isna(max_date) else prices["date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    date_range = (date.today(), date.today())
else:
    date_range = (min_date, max_date)
date_selection = st.sidebar.date_input("Date range", value=date_range, min_value=date_range[0], max_value=date_range[1])
if isinstance(date_selection, tuple) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    start_date = end_date = date_selection

ticker_summary = summary.get("tickers", {}).get(selected_ticker, {})
is_demo = bool(ticker_summary.get("demo_data", False)) or ("is_demo" in data and data["is_demo"].fillna(False).astype(bool).any())
if is_demo:
    st.warning("DEMO DATA — sentiment texts and price history are synthetic assignments for pipeline demonstration, not live market observations.")

if not data.empty:
    data["date"] = data["timestamp_utc"].dt.date
    data[selected_score] = pd.to_numeric(data.get(selected_score), errors="coerce")
    if selected_source == "News":
        data = data[data["source"].astype(str).str.lower().str.contains("news")]
    elif selected_source == "Bluesky":
        data = data[data["source"].astype(str).str.lower().eq("bluesky")]
    data = data[data["date"].between(start_date, end_date)]
    data = data.dropna(subset=[selected_score])

if not prices.empty:
    prices = prices[prices["date"].between(start_date, end_date)].dropna(subset=["close"])

overview_tab, headlines_tab, correlation_tab, models_tab, analyzer_tab = st.tabs(
    ["Overview", "Headlines", "Correlation", "Model & Backtest", "Live Analyzer"]
)

with overview_tab:
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Scored items", f"{len(data):,}")
    metric_b.metric("Mean sentiment", f"{data[selected_score].mean():.3f}" if not data.empty else "n/a")
    metric_c.metric("Price sessions", f"{len(prices):,}")
    st.subheader("Daily sentiment and adjusted close")
    if data.empty and prices.empty:
        st.info("No data in this date/source selection. Run `python run_pipeline.py --demo` or collect live sources.")
    else:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        if not prices.empty:
            figure.add_trace(go.Scatter(x=prices["date"], y=prices["close"], name="Adjusted close", mode="lines"), secondary_y=False)
        if not data.empty:
            daily = data.groupby("date", as_index=False)[selected_score].mean()
            figure.add_trace(go.Scatter(x=daily["date"], y=daily[selected_score], name=f"Daily {selected_model}", mode="lines+markers"), secondary_y=True)
        figure.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=True)
        figure.update_yaxes(title_text="Adjusted close", secondary_y=False)
        figure.update_yaxes(title_text="Sentiment score", range=[-1, 1], secondary_y=True)
        figure.update_xaxes(title_text="Date")
        figure.update_layout(height=480, legend_title="Series")
        st.plotly_chart(figure, use_container_width=True)

with headlines_tab:
    st.subheader("Highest and lowest sentiment items")
    if data.empty:
        st.info("No scored text for the selected filters.")
    else:
        columns = [column for column in ("source", "timestamp_utc", "text", selected_score, "url") if column in data]
        positive, negative = st.columns(2)
        with positive:
            st.markdown("**Most positive**")
            st.dataframe(data.nlargest(10, selected_score)[columns], use_container_width=True, hide_index=True, column_config={"url": st.column_config.LinkColumn("Source link")})
        with negative:
            st.markdown("**Most negative**")
            st.dataframe(data.nsmallest(10, selected_score)[columns], use_container_width=True, hide_index=True, column_config={"url": st.column_config.LinkColumn("Source link")})

with correlation_tab:
    corr_path = PROCESSED_DIR / f"correlations_{ticker_file_key(selected_ticker)}.csv"
    if corr_path.exists():
        correlations = load_csv(corr_path)
        if not correlations.empty:
            st.subheader("Sentiment versus future trading-session return")
            chart = px.bar(correlations, x="lag_sessions", y="correlation", error_y=None, hover_data=["n", "p_value"], labels={"lag_sessions": "Sessions ahead", "correlation": "Pearson r"})
            chart.add_hline(y=0, line_dash="dot")
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(correlations, use_container_width=True, hide_index=True)
            lag = st.selectbox("Scatter lag (sessions)", sorted(correlations["lag_sessions"].dropna().unique().tolist()), key="scatter_lag")
            market_path = PROCESSED_DIR / f"market_{ticker_file_key(selected_ticker)}.csv"
            if market_path.exists():
                market = load_csv(market_path)
                market["return_future"] = pd.to_numeric(market["return_t"], errors="coerce").shift(-int(lag))
                scatter = market.dropna(subset=["mean_sentiment", "return_future"])
                st.plotly_chart(px.scatter(scatter, x="mean_sentiment", y="return_future", hover_data=["date"], labels={"mean_sentiment": "Daily sentiment", "return_future": "Future return"}), use_container_width=True)
            st.caption("P-values and correlations are descriptive. Small samples and synthetic demo observations are not evidence of predictive power.")
        else:
            st.info("Not enough paired observations to calculate correlations.")
    else:
        st.info("Run the pipeline to generate lagged correlations.")

with models_tab:
    analysis_path = PROCESSED_DIR / f"analysis_{ticker_file_key(selected_ticker)}.json"
    if analysis_path.exists():
        analysis = load_json(analysis_path)
        direction = analysis.get("direction_model", {})
        st.subheader("Next-session direction model")
        if direction.get("status") == "ok":
            st.dataframe(pd.DataFrame([direction]).drop(columns=["feature_importance"], errors="ignore"), use_container_width=True, hide_index=True)
            st.json(direction.get("feature_importance", {}))
        else:
            st.info(direction.get("reason", direction.get("status", "Model unavailable")))
        st.subheader("Sentiment strategy versus buy-and-hold")
        backtest = analysis.get("backtest", {})
        st.caption(f"Transaction cost: {backtest.get('transaction_cost_bps_per_position_change', 'n/a')} bps per position change")
        if backtest.get("by_ticker"):
            rows = []
            for strategy_name, metrics in backtest["by_ticker"].get(selected_ticker, {}).items():
                rows.append({"strategy": strategy_name, **metrics})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            equity_path = PROCESSED_DIR / f"backtest_{ticker_file_key(selected_ticker)}.csv"
            if equity_path.exists():
                equity = load_csv(equity_path)
                if not equity.empty:
                    equity["strategy_equity"] = (1 + pd.to_numeric(equity["strategy_return"], errors="coerce").fillna(0)).cumprod()
                    equity["buy_hold_equity"] = (1 + pd.to_numeric(equity["buy_hold_return"], errors="coerce").fillna(0)).cumprod()
                    st.plotly_chart(px.line(equity, x="date", y=["strategy_equity", "buy_hold_equity"], labels={"value": "Growth of $1", "variable": "Strategy"}), use_container_width=True)
    else:
        st.info("Run the pipeline to train the time-split model and create the backtest.")
    evaluation_path = REPORTS_DIR / "sentiment_evaluation.csv"
    if evaluation_path.exists():
        st.subheader("Labeled-set sentiment evaluation")
        st.dataframe(load_csv(evaluation_path), use_container_width=True, hide_index=True)

with analyzer_tab:
    st.subheader("Analyze a headline or post")
    user_text = st.text_area("Text", placeholder="Paste a financial headline or social post…", height=120)
    if st.button("Analyze", type="primary"):
        if not user_text.strip():
            st.warning("Enter some text first.")
        else:
            vader = vader_scores([user_text]).iloc[0]
            st.metric("VADER", f"{vader['vader_label']} ({vader['vader_score']:.3f})")
            try:
                prediction = finbert_scores([user_text], local_files_only=True).iloc[0]
                st.metric("FinBERT", f"{prediction['finbert_label']} ({prediction['finbert_score']:.3f})")
                st.json({name: float(prediction[name]) for name in ("finbert_positive_probability", "finbert_neutral_probability", "finbert_negative_probability")})
            except Exception:
                st.info(f"FinBERT weights are not cached locally. Run `python run_pipeline.py --demo --download-model` once with internet access to fetch {MODEL_NAME}.")
