"""Leakage-aware alignment, correlation, and next-session direction model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def align_daily_to_market(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Map off-session/weekend sentiment forward to the next trading session."""
    if prices.empty or "date" not in prices or "close" not in prices:
        return pd.DataFrame()
    market = prices.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    market["return_t"] = market.groupby("ticker")["close"].pct_change()
    market["return_t_plus_1"] = market.groupby("ticker")["return_t"].shift(-1)
    if daily.empty:
        return market.assign(sentiment_date=pd.NaT, sentiment_assigned=False)
    sentiment = daily.copy()
    sentiment["calendar_date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.normalize()
    assigned: list[pd.DataFrame] = []
    for ticker, group in sentiment.groupby("ticker"):
        ticker_prices = market.loc[market["ticker"].eq(ticker), ["date"]].drop_duplicates().sort_values("date")
        if ticker_prices.empty:
            continue
        trade_dates = ticker_prices["date"].to_numpy(dtype="datetime64[ns]")
        indexes = np.searchsorted(trade_dates, group["calendar_date"].to_numpy(dtype="datetime64[ns]"), side="left")
        valid = indexes < len(trade_dates)
        mapped = group.loc[valid].copy()
        mapped["session_date"] = trade_dates[indexes[valid]]
        mapped["sentiment_date"] = mapped["calendar_date"]
        assigned.append(mapped)
    if not assigned:
        return market.assign(sentiment_date=pd.NaT, sentiment_assigned=False)
    mapped = pd.concat(assigned, ignore_index=True)
    numeric = [col for col in ("news_mean", "social_mean", "social_weighted_mean", "mean_sentiment", "positive_pct", "negative_pct") if col in mapped]
    sums = mapped.groupby(["ticker", "session_date"], as_index=False).agg({**{col: "mean" for col in numeric}, "count": "sum", "news_count": "sum", "social_count": "sum", "is_demo": "max", "sentiment_date": "min"})
    joined = market.merge(sums, left_on=["ticker", "date"], right_on=["ticker", "session_date"], how="left")
    joined["sentiment_assigned"] = joined["mean_sentiment"].notna()
    return joined.drop(columns=["session_date"], errors="ignore").reset_index(drop=True)


def correlation_table(aligned: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """Pearson r and p-values for same-day and 1..max_lag future returns."""
    rows: list[dict[str, Any]] = []
    if aligned.empty or "mean_sentiment" not in aligned:
        return pd.DataFrame(columns=["ticker", "lag_sessions", "n", "correlation", "p_value"])
    for ticker, group in aligned.sort_values("date").groupby("ticker"):
        for lag in range(0, max_lag + 1):
            future = group["return_t"] if lag == 0 else group["return_t"].shift(-lag)
            pairs = pd.DataFrame({"sentiment": group["mean_sentiment"], "future_return": future}).dropna()
            corr, pvalue = (np.nan, np.nan)
            if len(pairs) >= 3 and pairs.nunique().min() > 1:
                corr, pvalue = pearsonr(pairs["sentiment"], pairs["future_return"])
            rows.append({"ticker": ticker, "lag_sessions": lag, "n": int(len(pairs)), "correlation": float(corr) if np.isfinite(corr) else np.nan, "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan})
    return pd.DataFrame(rows)


def add_technical_features(aligned: pd.DataFrame) -> pd.DataFrame:
    """Create features known at close t for predicting direction at t+1."""
    pieces = []
    for _, group in aligned.sort_values("date").groupby("ticker"):
        part = group.copy()
        close = pd.to_numeric(part["close"], errors="coerce")
        ret = pd.to_numeric(part["return_t"], errors="coerce")
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = -delta.clip(upper=0).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        part["rsi_14"] = 100 - (100 / (1 + rs))
        part["volatility_20"] = ret.rolling(20, min_periods=5).std()
        part["ma_gap_20"] = close / close.rolling(20, min_periods=5).mean() - 1
        future_return = pd.to_numeric(part["return_t_plus_1"], errors="coerce")
        part["target_up_next"] = np.where(future_return.notna(), (future_return > 0).astype(float), np.nan)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def train_direction_model(aligned: pd.DataFrame) -> dict[str, Any]:
    """Fit a random forest with chronological 80/20 split and naive-majority baseline."""
    feature_names = ["mean_sentiment", "return_t", "volatility_20", "ma_gap_20", "rsi_14"]
    featured = add_technical_features(aligned)
    if featured.empty or not set(feature_names + ["target_up_next"]).issubset(featured):
        return {"status": "insufficient data", "reason": "price/sentiment features are unavailable"}
    valid = featured.dropna(subset=feature_names + ["target_up_next"]).reset_index(drop=True)
    if len(valid) < 30 or valid["target_up_next"].nunique() < 2:
        return {"status": "insufficient data", "n": int(len(valid)), "reason": "need at least 30 usable rows and both target classes"}
    split = int(len(valid) * 0.8)
    if split < 10 or len(valid) - split < 5:
        return {"status": "insufficient data", "n": int(len(valid)), "reason": "time split is too small"}
    train, test = valid.iloc[:split], valid.iloc[split:]
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=42)
    model.fit(train[feature_names], train["target_up_next"])
    prediction = model.predict(test[feature_names])
    baseline = np.full(len(test), int(train["target_up_next"].mean() >= 0.5))
    return {
        "status": "ok",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_end": str(train["date"].iloc[-1]),
        "test_start": str(test["date"].iloc[0]),
        "model_accuracy": float(accuracy_score(test["target_up_next"], prediction)),
        "model_precision": float(precision_score(test["target_up_next"], prediction, zero_division=0)),
        "model_recall": float(recall_score(test["target_up_next"], prediction, zero_division=0)),
        "model_f1": float(f1_score(test["target_up_next"], prediction, zero_division=0)),
        "naive_accuracy": float(accuracy_score(test["target_up_next"], baseline)),
        "feature_importance": dict(zip(feature_names, map(float, model.feature_importances_))),
    }
