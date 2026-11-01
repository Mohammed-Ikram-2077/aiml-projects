"""Collect headline records from the public GDELT DOC API."""

from __future__ import annotations

import logging
import json
import time as time_module
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import RAW_DIR, TICKER_ALIASES

LOGGER = logging.getLogger(__name__)
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_COLUMNS = ["ticker", "source", "text", "title", "summary", "published_at", "url", "score", "subreddit", "is_demo"]
MAX_ARTICLES = 250
MIN_REQUEST_INTERVAL_SECONDS = 5.5
_LAST_REQUEST_AT = time_module.monotonic()


def collect_news(
    ticker: str,
    start: date,
    end: date,
    *,
    output_dir: Path = RAW_DIR,
    force: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch GDELT headlines for a ticker and cache the normalized records.

    GDELT's DOC search covers a rolling period of roughly three months. Results
    are headline metadata; the collector does not scrape publisher article bodies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / f"news_{ticker.replace('.', '_')}.csv"
    metadata_path = cache.with_suffix(".meta.json")
    if cache.exists() and not force:
        cached = pd.read_csv(cache)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("provider") == "gdelt" and metadata.get("success") and metadata.get("start") and metadata.get("end"):
            if date.fromisoformat(metadata["start"]) <= start and date.fromisoformat(metadata["end"]) >= end:
                LOGGER.info("Using cached GDELT headlines for %s (%d records)", ticker, len(cached))
                return cached.reindex(columns=NEWS_COLUMNS)

    rows: list[dict[str, Any]] = []
    succeeded = False
    try:
        rows = _gdelt(ticker, start, end, timeout=timeout)
        succeeded = True
    except Exception as exc:
        LOGGER.error("GDELT collection failed for %s: %s", ticker, exc)
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    frame = frame.drop_duplicates(subset=["url"], keep="first") if not frame.empty else frame
    frame.to_csv(cache, index=False)
    metadata_path.write_text(
        json.dumps({"provider": "gdelt", "success": succeeded, "start": start.isoformat(), "end": end.isoformat(), "records": len(frame)}),
        encoding="utf-8",
    )
    LOGGER.info("Collected %d GDELT news records for %s", len(frame), ticker)
    return frame


def _gdelt(ticker: str, start: date, end: date, *, timeout: int) -> list[dict[str, Any]]:
    """Call DOC Article List mode and convert GDELT's result schema."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    search_terms = [ticker, *TICKER_ALIASES.get(ticker, ())[:3]]
    query = "(" + " OR ".join(f'"{term.replace(chr(34), "")}"' for term in search_terms) + ")"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": MAX_ARTICLES,
        "startdatetime": datetime.combine(start, time.min, tzinfo=timezone.utc).strftime("%Y%m%d%H%M%S"),
        "enddatetime": datetime.combine(end, time.max, tzinfo=timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    global _LAST_REQUEST_AT
    wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - (time_module.monotonic() - _LAST_REQUEST_AT)
    if wait_seconds > 0:
        time_module.sleep(wait_seconds)
    response: requests.Response | None = None
    for attempt in range(3):
        response = requests.get(GDELT_DOC_URL, params=params, headers={"User-Agent": "financial-sentiment-student-project/1.0"}, timeout=timeout)
        _LAST_REQUEST_AT = time_module.monotonic()
        if response.status_code != 429 or attempt == 2:
            break
        retry_after = response.headers.get("Retry-After", "")
        try:
            delay = max(MIN_REQUEST_INTERVAL_SECONDS, float(retry_after))
        except ValueError:
            delay = MIN_REQUEST_INTERVAL_SECONDS
        LOGGER.warning("GDELT rate-limited %s; retrying after %.1f seconds", ticker, delay)
        time_module.sleep(delay)
    assert response is not None
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        detail = response.text[:300].replace("\n", " ").strip()
        raise RuntimeError(f"GDELT returned non-JSON HTTP {response.status_code}: {detail}") from exc
    if payload == {}:
        return []
    if not isinstance(payload, dict) or "articles" not in payload:
        raise RuntimeError(f"GDELT response did not contain an articles list: {payload!r}")

    rows: list[dict[str, Any]] = []
    for article in payload.get("articles") or []:
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title or not url:
            continue
        published_at = _parse_gdelt_date(article.get("seendate"))
        if published_at:
            published_date = pd.Timestamp(published_at).date()
            if not start <= published_date <= end:
                continue
        rows.append({
            "ticker": ticker,
            "source": "gdelt_news",
            "text": title,
            "title": title,
            "summary": "",
            "published_at": published_at,
            "url": url,
            "score": None,
            "subreddit": None,
            "is_demo": False,
        })
    return rows


def _parse_gdelt_date(value: object) -> str | None:
    """Normalize GDELT timestamps (normally YYYYMMDDTHHMMSSZ) to UTC ISO."""
    if value is None or not str(value).strip():
        return None
    parsed = pd.to_datetime(str(value), utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.isoformat()
