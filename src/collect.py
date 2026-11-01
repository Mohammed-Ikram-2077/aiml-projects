"""Compatibility entry point for the active GDELT, Bluesky, and yfinance sources."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date, datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config import RAW_DIR, ROOT_DIR
from src.collect_bluesky import collect_bluesky
from src.collect_news import collect_news
from src.collect_prices import collect_prices

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = ROOT_DIR
RAW_DATA_DIR = RAW_DIR


def _as_date(value: date | datetime | str) -> date:
    """Convert a date, datetime, or ISO date string to a calendar date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _safe_ticker(ticker: str) -> str:
    """Return a normalized ticker token for display and file paths."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.strip().upper())
    if not token:
        raise ValueError("ticker must contain at least one letter or digit")
    return token


def collect_all(
    ticker: str,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    output_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Collect GDELT news, Bluesky posts, and yfinance prices for one ticker."""
    first, last = _as_date(start), _as_date(end)
    if first > last:
        raise ValueError("start date must be on or before end date")
    return {
        "news": collect_news(ticker, first, last, output_dir=output_dir, force=force),
        "bluesky": collect_bluesky(ticker, first, last, output_dir=output_dir, force=force),
        "prices": collect_prices(ticker, first, last, output_dir=output_dir, force=force),
    }


def main() -> None:
    """Run the three live collectors directly from the command line."""
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Ticker, e.g. NVDA or RELIANCE.NS")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Ignore cached collection files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = collect_all(args.ticker, args.start, args.end, output_dir=args.output_dir, force=args.force)
    for source, frame in results.items():
        LOGGER.info("%s %s: %d records", args.ticker, source, len(frame))


if __name__ == "__main__":
    main()
