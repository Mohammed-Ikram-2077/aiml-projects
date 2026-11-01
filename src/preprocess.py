"""Normalize collected text records for sentiment modeling."""

from __future__ import annotations

import argparse
import difflib
import html
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, TICKER_ALIASES
from src.collect import PROJECT_ROOT, RAW_DATA_DIR, _safe_ticker

LOGGER = logging.getLogger(__name__)
PROCESSED_DATA_DIR = PROCESSED_DIR
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")
CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9.\-]{0,9})")


def clean_text(text: object, *, keep_emojis: bool = False) -> str:
    """Remove URLs, social handles, emoji/symbol noise, and excess whitespace.

    Cashtags such as ``$AAPL`` are retained. Unicode letters, numbers, punctuation,
    and currency symbol ``$`` remain available to the sentiment models.
    """
    if text is None or pd.isna(text):
        return ""
    value = html.unescape(URL_RE.sub(" ", str(text)))
    value = re.sub(r"<[^>]+>", " ", value)
    value = MENTION_RE.sub(" ", value)
    retained: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        # Most emoji are symbols. Keep $ so cashtags survive the symbol filter.
        if category.startswith("S") and char != "$" and not keep_emojis:
            continue
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            continue
        retained.append(char)
    return WHITESPACE_RE.sub(" ", "".join(retained)).strip()


def detect_tickers(text: object, candidate_tickers: list[str] | tuple[str, ...]) -> list[str]:
    """Return candidate tickers mentioned as cashtags or standalone words.

    Cashtags are preferred for precision, while whole-word matching also detects
    ordinary forms like ``AAPL``. The function returns unique tickers in the order
    supplied by ``candidate_tickers``.
    """
    value = "" if text is None or pd.isna(text) else str(text)
    cashtags = {match.group(1).upper() for match in CASHTAG_RE.finditer(value)}
    found: list[str] = []
    for candidate in candidate_tickers:
        ticker = _safe_ticker(candidate)
        cashtag_match = ticker in cashtags
        bare_match = re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", value, re.I)
        if cashtag_match or bare_match:
            found.append(ticker)
    return found


def _choose_text(row: pd.Series) -> str:
    """Combine available headline/body fields into one modeling text."""
    body = row.get("text")
    if body is not None and not pd.isna(body) and str(body).strip():
        body_text = str(body).strip()
        title = row.get("title")
        if title is not None and not pd.isna(title) and str(title).strip() and str(title).strip().casefold() not in body_text.casefold():
            body_text = f"{str(title).strip()}. {body_text}"
        return body_text
    parts: list[str] = []
    for column in ("title", "description", "summary"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            parts.append(str(value).strip())
    return " ".join(parts)


def _timestamp_column(frame: pd.DataFrame) -> str | None:
    """Find a supported timestamp column from collector or downstream data."""
    for column in ("published_at", "created_at", "timestamp", "date"):
        if column in frame.columns:
            return column
    return None


def preprocess_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    source: str,
    candidate_tickers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Convert one raw source dataframe to a common, deduplicated schema.

    Timestamps are parsed as UTC. Naive timestamps are interpreted as UTC because
    both collector APIs provide UTC timestamps; invalid/missing times become NaT.
    """
    columns = ["ticker", "source", "timestamp_utc", "text", "title", "summary", "url", "score", "subreddit", "is_demo", "detected_tickers"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    candidates = candidate_tickers or [ticker]
    timestamp_col = _timestamp_column(frame)
    records = pd.DataFrame(index=frame.index)
    records["ticker"] = frame.get("ticker", pd.Series(ticker, index=frame.index)).fillna(ticker).astype(str).str.upper()
    records["source"] = source
    if timestamp_col:
        records["timestamp_utc"] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    else:
        records["timestamp_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    source_column = frame.get("source", pd.Series(source, index=frame.index)).fillna(source).astype(str)
    records["source"] = source_column
    records["text"] = [
        clean_text(_choose_text(row), keep_emojis=any(name in str(source_column.loc[index]).lower() for name in ("reddit", "bluesky")))
        for index, row in frame.iterrows()
    ]
    records["url"] = frame["url"] if "url" in frame else None
    for optional_column in ("title", "summary", "score", "subreddit"):
        records[optional_column] = frame[optional_column] if optional_column in frame else None
    records["is_demo"] = frame["is_demo"].fillna(False) if "is_demo" in frame else False
    records["detected_tickers"] = records["text"].map(lambda value: map_tickers(value, candidates))
    records = records[records["text"].str.len().ge(10)].copy()
    # Drop obvious low-information spam while retaining currency signs/cashtags.
    spam_mask = records["text"].map(lambda value: bool(re.fullmatch(r"(.)\1{5,}", str(value))))
    records = records[~spam_mask]
    records = records[records["detected_tickers"].map(bool)]

    # Prefer stable source identifiers when available; otherwise dedupe normalized
    # text within a source/ticker so repeated collection runs do not inflate counts.
    identifier = next((name for name in ("post_id", "url") if name in frame), None)
    if identifier == "post_id":
        records["_dedupe_key"] = frame.loc[records.index, identifier].fillna("").astype(str)
        empty_key = records["_dedupe_key"].eq("")
        records.loc[empty_key, "_dedupe_key"] = records.loc[empty_key, "text"]
    elif identifier == "url":
        records["_dedupe_key"] = frame.loc[records.index, identifier].fillna("").astype(str)
        empty_key = records["_dedupe_key"].eq("")
        records.loc[empty_key, "_dedupe_key"] = records.loc[empty_key, "text"]
    else:
        records["_dedupe_key"] = records["text"].str.casefold()
    records = records.drop_duplicates(subset=["ticker", "_dedupe_key"], keep="first")
    records = _drop_near_duplicates(records)
    return records[columns].reset_index(drop=True)


def map_tickers(text: object, candidates: list[str] | tuple[str, ...]) -> list[str]:
    """Detect a ticker from cashtags, symbols, and company aliases."""
    found = detect_tickers(text, candidates)
    value = "" if text is None or pd.isna(text) else str(text)
    for ticker in candidates:
        aliases = TICKER_ALIASES.get(_safe_ticker(ticker), ())
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", value, flags=re.IGNORECASE) for alias in aliases):
            normalized = _safe_ticker(ticker)
            if normalized not in found:
                found.append(normalized)
    return found


def _drop_near_duplicates(frame: pd.DataFrame, threshold: float = 0.96) -> pd.DataFrame:
    """Remove highly similar texts per ticker using a conservative ratio cutoff."""
    keep: list[int] = []
    by_ticker: dict[str, list[str]] = {}
    for index, row in frame.iterrows():
        text = str(row["text"]).casefold()
        prior = by_ticker.setdefault(str(row["ticker"]), [])
        if bool(row.get("is_demo", False)):
            # Demo fixtures deliberately reuse labeled sentences on different
            # synthetic days; retain each timestamped record for time-series demos.
            prior.append(text)
            keep.append(index)
            continue
        if any(difflib.SequenceMatcher(None, text, existing).ratio() >= threshold for existing in prior[-250:]):
            continue
        prior.append(text)
        keep.append(index)
    return frame.loc[keep].copy()


def preprocess_ticker(
    ticker: str,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    candidate_tickers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read source CSVs for a ticker, normalize records, and save processed CSV."""
    token = _safe_ticker(ticker)
    frames: list[pd.DataFrame] = []
    for source in ("news", "bluesky"):
        path = raw_dir / f"{token}_{source}.csv"
        if not path.exists():
            LOGGER.info("No %s raw file found for %s: %s", source, token, path)
            continue
        raw = pd.read_csv(path)
        frames.append(
            preprocess_frame(
                raw,
                ticker=token,
                source=source,
                candidate_tickers=candidate_tickers or [token],
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "source", "timestamp_utc", "text", "title", "summary", "url", "score", "subreddit", "is_demo", "detected_tickers"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{token}_processed.csv"
    combined.to_csv(output_path, index=False)
    LOGGER.info("Saved %d processed records to %s", len(combined), output_path)
    return combined


def main() -> None:
    """Run preprocessing for ticker symbols passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", nargs="+", required=True, help="Ticker symbol(s), e.g. AAPL MSFT")
    parser.add_argument("--candidate-tickers", nargs="*", help="Symbols to detect within each record")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for symbol in args.ticker:
        preprocess_ticker(
            symbol,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            candidate_tickers=args.candidate_tickers,
        )


if __name__ == "__main__":
    main()
