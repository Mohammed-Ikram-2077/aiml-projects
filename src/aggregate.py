"""Aggregate scored records into daily ticker and source sentiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE_GROUPS = {
    "news": "news",
    "news_rss": "news",
    "gdelt_news": "news",
    "reddit": "social",  # Preserve compatibility with older cached datasets.
    "reddit_comment": "social",
    "bluesky": "social",
}


def aggregate_daily(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sentiment, weighted social sentiment, counts, and polarity."""
    columns = ["ticker", "date", "news_mean", "social_mean", "social_weighted_mean", "mean_sentiment", "count", "news_count", "social_count", "positive_pct", "negative_pct", "is_demo"]
    if sentiment.empty:
        return pd.DataFrame(columns=columns)
    frame = sentiment.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.get("timestamp_utc", frame.get("published_at")), utc=True, errors="coerce")
    if "published_at" in frame and "timestamp_utc" not in sentiment:
        frame["timestamp_utc"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["date"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%d")
    frame["sentiment_score"] = pd.to_numeric(frame.get("finbert_score"), errors="coerce")
    if "vader_score" in frame:
        frame["sentiment_score"] = frame["sentiment_score"].fillna(pd.to_numeric(frame["vader_score"], errors="coerce"))
    frame = frame.dropna(subset=["date", "sentiment_score"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "source" not in frame:
        frame["source"] = "news"
    frame["source_group"] = frame["source"].map(SOURCE_GROUPS).fillna("news")
    raw_weights = frame["score"] if "score" in frame else pd.Series(1, index=frame.index)
    frame["score_weight"] = pd.to_numeric(raw_weights, errors="coerce").fillna(1).clip(lower=1)
    if "is_demo" not in frame:
        frame["is_demo"] = False
    else:
        frame["is_demo"] = frame["is_demo"].fillna(False).astype(bool)

    group_cols = ["ticker", "date"]
    base = frame.groupby(group_cols).agg(mean_sentiment=("sentiment_score", "mean"), count=("sentiment_score", "count"), positive_pct=("sentiment_score", lambda values: float((values > 0.05).mean())), negative_pct=("sentiment_score", lambda values: float((values < -0.05).mean())), is_demo=("is_demo", "max")).reset_index()
    parts = []
    for source_group in ("news", "social"):
        subset = frame[frame["source_group"].eq(source_group)]
        if source_group == "social":
            grouped = subset.groupby(group_cols).apply(lambda group: pd.Series({"social_mean": group["sentiment_score"].mean(), "social_weighted_mean": np.average(group["sentiment_score"], weights=group["score_weight"]), "social_count": len(group)}), include_groups=False).reset_index()
        else:
            grouped = subset.groupby(group_cols).agg(news_mean=("sentiment_score", "mean"), news_count=("sentiment_score", "count")).reset_index()
        parts.append(grouped)
    result = base
    for part in parts:
        result = result.merge(part, on=group_cols, how="left")
    for col in ("news_count", "social_count"):
        result[col] = result[col].fillna(0).astype(int)
    return result[columns].sort_values(["ticker", "date"]).reset_index(drop=True)
