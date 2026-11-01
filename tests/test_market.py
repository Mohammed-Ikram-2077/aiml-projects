"""Unit tests for daily aggregation and forward-return alignment."""

import pandas as pd

from src.market import aggregate_daily_sentiment, align_sentiment_with_returns


def test_aggregate_sentiment_by_utc_date() -> None:
    sentiment = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "timestamp_utc": ["2024-01-02T10:00:00Z", "2024-01-02T15:00:00Z"],
            "finbert_score": [0.4, -0.2],
        }
    )
    result = aggregate_daily_sentiment(sentiment)
    assert result.loc[0, "date"] == "2024-01-02"
    assert result.loc[0, "daily_sentiment"] == 0.1
    assert result.loc[0, "sentiment_count"] == 2


def test_next_session_return_is_aligned_to_prior_date() -> None:
    sentiment = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "timestamp_utc": ["2024-01-02T18:00:00Z", "2024-01-03T18:00:00Z"],
            "finbert_score": [0.8, -0.5],
        }
    )
    prices = pd.DataFrame(
        {"ticker": ["AAPL", "AAPL", "AAPL"], "date": ["2024-01-02", "2024-01-03", "2024-01-04"], "close": [100, 110, 99]}
    )
    result = align_sentiment_with_returns(sentiment, prices)
    assert result.loc[0, "daily_sentiment"] == 0.8
    assert abs(result.loc[0, "return_t_plus_1"] - 0.1) < 1e-9
    assert abs(result.loc[1, "return_t_plus_1"] + 0.1) < 1e-9
    assert pd.isna(result.loc[2, "return_t_plus_1"])
