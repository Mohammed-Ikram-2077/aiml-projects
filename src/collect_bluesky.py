"""Collect public Bluesky posts through the unauthenticated AppView API."""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import RAW_DIR, TICKER_ALIASES

LOGGER = logging.getLogger(__name__)
BLUESKY_SEARCH_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
BLUESKY_COLUMNS = ["ticker", "source", "text", "title", "summary", "published_at", "url", "score", "subreddit", "is_demo"]
PAGE_SIZE = 100


def collect_bluesky(
    ticker: str,
    start: date,
    end: date,
    *,
    output_dir: Path = RAW_DIR,
    force: bool = False,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch public posts mentioning the ticker/cashtag or known company aliases.

    Public search reads do not need an app password. The endpoint is a search
    index, so historical coverage and result completeness depend on its index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / f"bluesky_{ticker.replace('.', '_')}.csv"
    metadata_path = cache.with_suffix(".meta.json")
    if cache.exists() and not force:
        cached = pd.read_csv(cache)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("provider") == "bluesky" and metadata.get("success") and metadata.get("start") and metadata.get("end"):
            if date.fromisoformat(metadata["start"]) <= start and date.fromisoformat(metadata["end"]) >= end:
                LOGGER.info("Using cached Bluesky posts for %s (%d records)", ticker, len(cached))
                return cached.reindex(columns=BLUESKY_COLUMNS)

    succeeded = False
    try:
        rows = _search_posts(ticker, start, end, timeout=timeout)
        succeeded = True
    except Exception as exc:
        LOGGER.error("Bluesky collection failed for %s: %s", ticker, exc)
        rows = []
    frame = pd.DataFrame(rows, columns=BLUESKY_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["url"], keep="first")
    frame.to_csv(cache, index=False)
    metadata_path.write_text(
        json.dumps({"provider": "bluesky", "success": succeeded, "start": start.isoformat(), "end": end.isoformat(), "records": len(frame)}),
        encoding="utf-8",
    )
    LOGGER.info("Collected %d Bluesky posts for %s", len(frame), ticker)
    return frame


def _search_posts(ticker: str, start: date, end: date, *, timeout: int) -> list[dict[str, Any]]:
    """Search separate terms, date-filter results, and deduplicate by post URI."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    terms = list(dict.fromkeys([f"${ticker}", ticker, *TICKER_ALIASES.get(ticker, ())[:2]]))
    session = requests.Session()
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_exclusive = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    rows_by_uri: dict[str, dict[str, Any]] = {}
    successful_searches = 0

    for term in terms:
        params: dict[str, Any] = {
            "q": term,
            "sort": "latest",
            "since": start_dt.isoformat().replace("+00:00", "Z"),
            "until": end_exclusive.isoformat().replace("+00:00", "Z"),
            "limit": PAGE_SIZE,
        }
        try:
            response = session.get(BLUESKY_SEARCH_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Bluesky search for %s query %r failed: %s", ticker, term, exc)
            continue
        successful_searches += 1
        for view in payload.get("posts", []):
            post = _post_row(ticker, view, start_dt, end_exclusive)
            if post:
                rows_by_uri.setdefault(post["_uri"], post)
        time.sleep(0.15)

    if not successful_searches:
        raise RuntimeError("All Bluesky search terms failed")
    rows: list[dict[str, Any]] = []
    for row in rows_by_uri.values():
        row.pop("_uri", None)
        rows.append(row)
    return rows


def _post_row(
    ticker: str,
    view: dict[str, Any],
    start: datetime,
    end_exclusive: datetime,
) -> dict[str, Any] | None:
    """Convert a Bluesky post view into the pipeline's shared raw schema."""
    record = view.get("record") or {}
    text = str(record.get("text") or "").strip()
    uri = str(view.get("uri") or "")
    created_raw = record.get("createdAt") or view.get("indexedAt")
    created = pd.to_datetime(created_raw, utc=True, errors="coerce")
    if not text or not uri or pd.isna(created) or not pd.Timestamp(start) <= created < pd.Timestamp(end_exclusive):
        return None
    author = view.get("author") or {}
    handle = str(author.get("handle") or author.get("did") or "")
    rkey = uri.rsplit("/", 1)[-1]
    url = f"https://bsky.app/profile/{handle}/post/{rkey}" if handle else uri
    engagement = sum(int(view.get(key) or 0) for key in ("likeCount", "repostCount", "quoteCount"))
    return {
        "_uri": uri,
        "ticker": ticker,
        "source": "bluesky",
        "text": text,
        "title": "",
        "summary": "",
        "published_at": created.isoformat(),
        "url": url,
        "score": engagement,
        "subreddit": None,
        "is_demo": False,
    }
