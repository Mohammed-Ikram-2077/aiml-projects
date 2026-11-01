"""Tests for ticker mapping, sentiment, aggregation, and time-aware modeling."""

import numpy as np
import pandas as pd

from src.aggregate import aggregate_daily
from src.analysis import align_daily_to_market, train_direction_model
from src.preprocess import clean_text, map_tickers
from src.sentiment import vader_scores


def test_alias_mapping_and_social_emoji_retention() -> None:
    assert map_tickers("Apple shares rise $AAPL 🚀", ["AAPL"]) == ["AAPL"]
    assert "🚀" in clean_text("$AAPL to the moon 🚀", keep_emojis=True)


def test_sentiment_scores_are_signed_and_bounded() -> None:
    result = vader_scores(["Revenue surged and profits beat estimates.", "The company reported a severe loss."])
    assert result.loc[0, "vader_score"] > 0
    assert result.loc[1, "vader_score"] < 0
    assert result["vader_score"].between(-1, 1).all()


def test_aggregation_keeps_news_and_social_and_weights_social() -> None:
    records = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "source": ["news", "reddit", "reddit_comment"],
            "timestamp_utc": ["2026-01-02T12:00:00Z"] * 3,
            "finbert_score": [0.2, 0.8, -0.2],
            "vader_score": [0.2, 0.8, -0.2],
            "score": [None, 9, 1],
        }
    )
    daily = aggregate_daily(records).iloc[0]
    assert daily["news_mean"] == 0.2
    assert daily["social_count"] == 2
    assert daily["social_weighted_mean"] > daily["social_mean"]


def test_weekend_sentiment_moves_forward_to_next_session() -> None:
    prices = pd.DataFrame({"ticker": ["AAPL"] * 3, "date": ["2026-01-02", "2026-01-05", "2026-01-06"], "close": [100, 101, 99]})
    daily = pd.DataFrame({"ticker": ["AAPL"], "date": ["2026-01-03"], "mean_sentiment": [0.7], "count": [1], "news_count": [1], "social_count": [0], "is_demo": [False]})
    aligned = align_daily_to_market(daily, prices)
    monday = aligned.loc[aligned["date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
    assert monday["mean_sentiment"] == 0.7
    assert monday["sentiment_assigned"]


def test_market_model_uses_chronological_holdout() -> None:
    dates = pd.bdate_range("2025-01-01", periods=90)
    returns = np.sin(np.arange(len(dates)) * 1.7) * 0.015 + 0.001
    close = 100 * np.cumprod(1 + returns)
    aligned = pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "close": close,
            "return_t": returns,
            "return_t_plus_1": np.r_[returns[1:], np.nan],
            "mean_sentiment": np.cos(np.arange(len(dates)) * 0.8) * 0.5,
        }
    )
    report = train_direction_model(aligned)
    assert report["status"] == "ok"
    assert report["train_end"] < report["test_start"]
