"""Download daily adjusted OHLCV data with a CSV cache."""

from __future__ import annotations

import logging
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import RAW_DIR

LOGGER = logging.getLogger(__name__)
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "is_demo"]


def collect_prices(ticker: str, start: date, end: date, *, output_dir: Path = RAW_DIR, demo: bool = False, force: bool = False) -> pd.DataFrame:
    """Fetch or synthesize ticker OHLCV; saved results are reused when covering dates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / f"prices_{ticker.replace('.', '_')}.csv"
    metadata_path = cache.with_suffix(".meta.json")
    if cache.exists() and not force:
        cached = pd.read_csv(cache)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("start") and metadata.get("end") and date.fromisoformat(metadata["start"]) <= start and date.fromisoformat(metadata["end"]) >= end:
            LOGGER.info("Using cached price history for %s (%d rows)", ticker, len(cached))
            return cached
    if demo:
        frame = synthetic_prices(ticker, start, end)
    else:
        try:
            raw = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d", auto_adjust=True, progress=False, threads=False, multi_level_index=False)
            if raw is None or raw.empty:
                LOGGER.warning("yfinance returned no rows for %s", ticker)
                frame = pd.DataFrame(columns=PRICE_COLUMNS)
            else:
                raw = raw.reset_index()
                raw.columns = [str(column).lower().replace(" ", "_") for column in raw.columns]
                date_column = "date" if "date" in raw else "datetime"
                frame = pd.DataFrame({"date": pd.to_datetime(raw[date_column]).dt.strftime("%Y-%m-%d"), "ticker": ticker, "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"], "volume": raw["volume"], "is_demo": False})
        except Exception as exc:
            LOGGER.error("yfinance download failed for %s: %s", ticker, exc)
            frame = pd.DataFrame(columns=PRICE_COLUMNS)
    frame[PRICE_COLUMNS].to_csv(cache, index=False)
    metadata_path.write_text(json.dumps({"start": start.isoformat(), "end": end.isoformat(), "records": len(frame)}), encoding="utf-8")
    return frame[PRICE_COLUMNS]


def synthetic_prices(ticker: str, start: date, end: date, *, seed: int = 42) -> pd.DataFrame:
    """Create visibly synthetic business-day prices for demo-only analysis."""
    dates = pd.bdate_range(start=start, end=end)
    if dates.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    offset_seed = seed + sum(ord(char) for char in ticker)
    rng = np.random.default_rng(offset_seed)
    start_price = 50.0 + (offset_seed % 350)
    returns = rng.normal(0.0003, 0.018, len(dates))
    close = start_price * np.cumprod(1.0 + returns)
    open_price = close * (1.0 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.015, len(dates)))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.015, len(dates)))
    volume = rng.integers(1_000_000, 20_000_000, len(dates))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": ticker, "open": open_price, "high": high, "low": low, "close": close, "volume": volume, "is_demo": True})
