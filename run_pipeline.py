"""Single entry point for collection/demo generation through analysis and backtest."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import DEFAULT_CONFIG, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ROOT_DIR, ProjectConfig
from src.aggregate import aggregate_daily
from src.analysis import align_daily_to_market, correlation_table, train_direction_model
from src.backtest import run_backtest
from src.collect_news import NEWS_COLUMNS, collect_news
from src.collect_prices import PRICE_COLUMNS, collect_prices
from src.collect_bluesky import BLUESKY_COLUMNS, collect_bluesky
from src.evaluate import evaluate_labeled_text, save_evaluation
from src.load_datasets import load_financial_phrasebank
from src.preprocess import preprocess_frame
from src.sentiment import score_frame

LOGGER = logging.getLogger("pipeline")


def _demo_records(ticker: str, start: date, end: date, labeled: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign labeled financial sentences to synthetic dates/tickers for demo only."""
    if labeled.empty:
        raise ValueError("The demo requires at least one labeled financial text example")
    days = pd.bdate_range(start=start, end=end)
    if days.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    rng = np.random.default_rng(seed + sum(ord(char) for char in ticker))
    rows: list[dict[str, Any]] = []
    tick = 0
    for day in days:
        for slot in range(3):
            sample = labeled.iloc[tick % len(labeled)]
            tick += 1
            source = "news" if slot == 0 else "bluesky"
            text = f"${ticker} demo item {day.strftime('%Y%m%d')}-{slot}: {sample['text']}"
            published = day.to_pydatetime().replace(hour=14 if source == "news" else 19, minute=slot * 7, tzinfo=__import__("datetime").timezone.utc)
            rows.append({
                "ticker": ticker,
                "source": source,
                "text": text,
                "title": str(sample["text"])[:120] if source == "news" else "",
                "summary": str(sample["text"]),
                "published_at": published.isoformat(),
                "url": f"demo://{ticker}/{day.strftime('%Y%m%d')}/{slot}",
                "score": int(rng.integers(1, 250)) if source == "bluesky" else None,
                "subreddit": None,
                "is_demo": True,
                "demo_gold_label": sample["label"],
            })
    return pd.DataFrame(rows)


def _load_cached_raw(ticker: str) -> pd.DataFrame:
    """Read the current collector cache files for one ticker."""
    pieces = []
    for prefix, schema in (("news", NEWS_COLUMNS), ("bluesky", BLUESKY_COLUMNS)):
        path = RAW_DIR / f"{prefix}_{ticker.replace('.', '_')}.csv"
        pieces.append(pd.read_csv(path) if path.exists() else pd.DataFrame(columns=schema))
    return pd.concat(pieces, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run all project stages and persist reusable intermediate outputs."""
    load_dotenv(ROOT_DIR / ".env")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    start, end = ProjectConfig(months=args.months).date_range(today) if not args.start else (date.fromisoformat(args.start), date.fromisoformat(args.end))
    tickers = tuple(args.tickers or DEFAULT_CONFIG.tickers)
    labeled = load_financial_phrasebank(csv_path=args.dataset_csv, allow_download=args.download_dataset)
    LOGGER.info("Using %s labeled examples for the baseline/evaluation", len(labeled))
    ticker_summary: dict[str, Any] = {}

    for ticker in tickers:
        LOGGER.info("Starting %s from %s through %s%s", ticker, start, end, " (DEMO DATA)" if args.demo else "")
        if args.demo:
            raw = _demo_records(ticker, start, end, labeled, DEFAULT_CONFIG.random_seed)
            raw_dir = RAW_DIR
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw[raw["source"].eq("news")].to_csv(raw_dir / f"news_{ticker.replace('.', '_')}.csv", index=False)
            raw[raw["source"].eq("bluesky")].to_csv(raw_dir / f"bluesky_{ticker.replace('.', '_')}.csv", index=False)
        elif args.skip_collect:
            raw = _load_cached_raw(ticker)
        else:
            news = collect_news(ticker, start, end, force=args.force_collect)
            bluesky = collect_bluesky(ticker, start, end, force=args.force_collect)
            raw = pd.concat([news, bluesky], ignore_index=True)

        prices = collect_prices(ticker, start, end, demo=args.demo, force=args.force_collect)
        preprocessed = preprocess_frame(raw, ticker=ticker, source="combined", candidate_tickers=tickers)
        preprocessed["is_demo"] = True if args.demo else preprocessed.get("is_demo", False)
        preprocessed.to_csv(PROCESSED_DIR / f"clean_{ticker.replace('.', '_')}.csv", index=False)

        scored = score_frame(
            preprocessed,
            labeled_training=labeled,
            include_finbert=True,
            finbert_local_only=args.demo and not args.download_model,
            batch_size=args.batch_size,
        )
        scored.to_csv(PROCESSED_DIR / f"sentiment_{ticker.replace('.', '_')}.csv", index=False)
        daily = aggregate_daily(scored)
        daily.to_csv(PROCESSED_DIR / f"daily_{ticker.replace('.', '_')}.csv", index=False)

        aligned = align_daily_to_market(daily, prices)
        correlations = correlation_table(aligned)
        direction_model = train_direction_model(aligned)
        backtest_frame, backtest_summary = run_backtest(
            aligned,
            threshold=DEFAULT_CONFIG.sentiment_threshold,
            transaction_cost_bps=DEFAULT_CONFIG.transaction_cost_bps,
        )
        aligned.to_csv(PROCESSED_DIR / f"market_{ticker.replace('.', '_')}.csv", index=False)
        correlations.to_csv(PROCESSED_DIR / f"correlations_{ticker.replace('.', '_')}.csv", index=False)
        backtest_frame.to_csv(PROCESSED_DIR / f"backtest_{ticker.replace('.', '_')}.csv", index=False)
        report = {"ticker": ticker, "is_demo": bool(args.demo), "direction_model": direction_model, "backtest": backtest_summary}
        (PROCESSED_DIR / f"analysis_{ticker.replace('.', '_')}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        ticker_summary[ticker] = {
            "news_rows": int(raw[raw.get("source", pd.Series(index=raw.index, dtype=str)).astype(str).str.contains("news")].shape[0]),
            "social_rows": int(raw[raw.get("source", pd.Series(index=raw.index, dtype=str)).astype(str).eq("bluesky")].shape[0]),
            "scored_rows": int(len(scored)),
            "price_rows": int(len(prices)),
            "demo_data": bool(args.demo),
            "finbert_status": str(scored["finbert_status"].iloc[0]) if not scored.empty else "no input rows",
            "direction_model": direction_model,
            "backtest": backtest_summary,
        }
        LOGGER.info("%s: GDELT news/Bluesky=%d/%d, prices=%d", ticker, ticker_summary[ticker]["news_rows"], ticker_summary[ticker]["social_rows"], len(prices))

    evaluation = evaluate_labeled_text(labeled, local_finbert_only=args.demo and not args.download_model)
    save_evaluation(evaluation, REPORTS_DIR)
    summary = {"start": start.isoformat(), "end": end.isoformat(), "demo_data": bool(args.demo), "dataset": str(labeled["dataset"].iloc[0]) if not labeled.empty else "none", "tickers": ticker_summary, "evaluation": evaluation}
    (PROCESSED_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use visibly labeled synthetic ticker/date assignments; no API calls")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing raw CSV caches")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--months", type=int, default=DEFAULT_CONFIG.months)
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--download-dataset", action="store_true", help="Try downloading Financial PhraseBank; offline fixture is used if unavailable")
    parser.add_argument("--download-model", action="store_true", help="Allow downloading FinBERT weights (demo defaults to cached weights only)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--force-collect", action="store_true", help="Ignore collected CSV caches")
    args = parser.parse_args()
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be supplied together")
    if args.months < 1 or args.batch_size < 1:
        parser.error("--months and --batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
