"""Align daily sentiment with market returns and report correlations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collect import PROJECT_ROOT, RAW_DATA_DIR, _safe_ticker

LOGGER = logging.getLogger(__name__)
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def aggregate_daily_sentiment(
    sentiment: pd.DataFrame,
    *,
    score_col: str = "finbert_score",
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Average valid record scores by ticker and UTC calendar date."""
    needed = {score_col, timestamp_col}
    missing = needed - set(sentiment.columns)
    if missing:
        raise ValueError(f"sentiment data is missing columns: {sorted(missing)}")
    frame = sentiment.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=[timestamp_col, score_col])
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "date", "daily_sentiment", "sentiment_count"])
    frame["date"] = frame[timestamp_col].dt.strftime("%Y-%m-%d")
    if "ticker" not in frame:
        frame["ticker"] = "UNKNOWN"
    return (
        frame.groupby(["ticker", "date"], as_index=False)
        .agg(daily_sentiment=(score_col, "mean"), sentiment_count=(score_col, "count"))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def align_sentiment_with_returns(
    sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    score_col: str = "finbert_score",
) -> pd.DataFrame:
    """Join same-calendar-day sentiment to prices and add next-session returns.

    ``return_t`` is the close-to-close return ending on date t. ``return_t_plus_1``
    is the next observed trading session's return and is the causal lag comparison:
    daily sentiment is paired with a return that occurs after that calendar day.
    The same-day comparison is descriptive and can include after-close information.
    """
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("prices must contain 'date' and 'close' columns")
    daily = aggregate_daily_sentiment(sentiment, score_col=score_col)
    market = prices.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["date", "close"]).sort_values("date")
    market = market.drop_duplicates(subset=["date"], keep="last")
    if "ticker" not in market:
        market["ticker"] = "UNKNOWN"
    market["return_t"] = market.groupby("ticker")["close"].pct_change()
    market["return_t_plus_1"] = market.groupby("ticker")["return_t"].shift(-1)
    joined = market.merge(daily, on=["ticker", "date"], how="left")
    return joined.sort_values(["ticker", "date"]).reset_index(drop=True)


def correlation_report(aligned: pd.DataFrame) -> dict[str, Any]:
    """Compute Pearson correlations for same-day and next-session observations."""
    report: dict[str, Any] = {}
    for label, return_col in (("same_day", "return_t"), ("next_session", "return_t_plus_1")):
        valid = aligned[["daily_sentiment", return_col]].dropna()
        report[label] = {
            "n": int(len(valid)),
            "pearson_correlation": (
                float(valid["daily_sentiment"].corr(valid[return_col]))
                if len(valid) >= 2 and valid["daily_sentiment"].nunique() > 1 and valid[return_col].nunique() > 1
                else None
            ),
        }
    return report


def lagged_correlations(aligned: pd.DataFrame, max_lag: int = 5) -> dict[int, float | None]:
    """Correlate sentiment at session t with returns at t+lag for each ticker."""
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    result: dict[int, float | None] = {}
    ordered = aligned.sort_values(["ticker", "date"])
    for lag in range(1, max_lag + 1):
        pairs: list[pd.DataFrame] = []
        for _, group in ordered.groupby("ticker"):
            paired = group[["daily_sentiment"]].copy()
            paired["future_return"] = group["return_t"].shift(-lag).to_numpy()
            pairs.append(paired)
        valid = pd.concat(pairs, ignore_index=True).dropna() if pairs else pd.DataFrame()
        result[lag] = (
            float(valid["daily_sentiment"].corr(valid["future_return"]))
            if len(valid) >= 2 and valid["daily_sentiment"].nunique() > 1 and valid["future_return"].nunique() > 1
            else None
        )
    return result


def granger_test(aligned: pd.DataFrame, max_lag: int = 2) -> dict[str, Any]:
    """Optionally test whether past sentiment adds predictive information for returns.

    This is an exploratory statistical test, not evidence of causal impact. It needs
    enough consecutive observations and uses only rows with both series present.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as exc:
        raise RuntimeError("Install statsmodels to run the optional Granger test") from exc
    frame = aligned[["return_t", "daily_sentiment"]].dropna()
    if len(frame) <= max_lag + 2:
        return {"status": "insufficient observations", "n": int(len(frame))}
    values = grangercausalitytests(frame[["return_t", "daily_sentiment"]], maxlag=max_lag, verbose=False)
    return {
        "status": "ok",
        "n": int(len(frame)),
        "ssr_ftest_p_values": {str(lag): float(result[0]["ssr_ftest"][1]) for lag, result in values.items()},
    }


def main() -> None:
    """Align one ticker's scores/prices, write rows, and print correlation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", choices=("vader", "finbert"), default="finbert")
    parser.add_argument("--sentiment-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--price-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--granger", action="store_true", help="Also run the optional exploratory test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ticker = _safe_ticker(args.ticker)
    sentiment_path = args.sentiment_dir / f"{ticker}_sentiment.csv"
    price_path = args.price_dir / f"{ticker}_prices.csv"
    sentiment = pd.read_csv(sentiment_path)
    prices = pd.read_csv(price_path)
    aligned = align_sentiment_with_returns(sentiment, prices, score_col=f"{args.model}_score")
    summary: dict[str, Any] = {
        "ticker": ticker,
        "sentiment_model": args.model,
        "correlations": correlation_report(aligned),
        "lagged_correlations": lagged_correlations(aligned),
    }
    if args.granger:
        summary["granger"] = granger_test(aligned)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.output_dir / f"{ticker}_market_analysis.csv", index=False)
    (args.output_dir / f"{ticker}_market_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
