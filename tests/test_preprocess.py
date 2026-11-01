"""Unit tests for text cleanup, ticker detection, and timestamp normalization."""

import pandas as pd

from src.preprocess import clean_text, detect_tickers, preprocess_frame


def test_clean_text_removes_noise_and_keeps_cashtag() -> None:
    cleaned = clean_text("$AAPL 🚀 @trader https://example.com  beats   estimates!")
    assert cleaned == "$AAPL beats estimates!"


def test_detect_tickers_handles_cashtags_and_whole_words() -> None:
    assert detect_tickers("$AAPL rises; MSFT follows. NOTAAPL", ["AAPL", "MSFT"]) == ["AAPL", "MSFT"]


def test_preprocess_normalizes_time_and_deduplicates_text() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "title": ["$AAPL beats estimates", "$AAPL beats estimates"],
            "description": [None, None],
            "published_at": ["2024-01-02T10:00:00-05:00", "2024-01-02T15:00:00Z"],
            "url": [None, None],
        }
    )
    result = preprocess_frame(raw, ticker="AAPL", source="news")
    assert len(result) == 1
    assert str(result.loc[0, "timestamp_utc"]) == "2024-01-02 15:00:00+00:00"
    assert result.loc[0, "detected_tickers"] == ["AAPL"]
