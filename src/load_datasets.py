"""Load labeled financial text or provide a clearly synthetic offline fixture."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Compact, hand-written financial text fixtures keep --demo fully offline. They are
# not presented as real posts or as a replacement for a licensed public benchmark.
OFFLINE_FIXTURE: tuple[tuple[str, str], ...] = (
    ("The company reported record revenue and raised its full-year outlook.", "positive"),
    ("Shares surged after earnings comfortably beat analyst expectations.", "positive"),
    ("Strong demand drove higher sales and improved operating margins.", "positive"),
    ("The board approved a larger dividend after robust cash generation.", "positive"),
    ("The firm secured a major contract that is expected to lift growth.", "positive"),
    ("Management announced a successful product launch and expanding orders.", "positive"),
    ("Quarterly profit increased as costs fell and revenue accelerated.", "positive"),
    ("Analysts upgraded the stock following better-than-expected guidance.", "positive"),
    ("The company completed a share buyback funded by healthy free cash flow.", "positive"),
    ("Customer growth reached a new high during the quarter.", "positive"),
    ("Revenue declined and management lowered its annual forecast.", "negative"),
    ("The stock plunged after the company missed earnings estimates.", "negative"),
    ("Higher costs squeezed margins and reduced quarterly profit.", "negative"),
    ("The regulator opened an investigation into the company's accounting.", "negative"),
    ("Weak demand led to falling orders and excess inventory.", "negative"),
    ("The firm announced layoffs as cash losses continued to widen.", "negative"),
    ("Analysts cut their price target after disappointing guidance.", "negative"),
    ("Debt increased while operating cash flow weakened.", "negative"),
    ("A product recall is expected to weigh on sales this year.", "negative"),
    ("The company warned that supply disruptions could delay deliveries.", "negative"),
    ("The company reported quarterly revenue of $12 billion.", "neutral"),
    ("The board will meet next week to discuss the proposed transaction.", "neutral"),
    ("Shares closed little changed ahead of the earnings release.", "neutral"),
    ("The firm appointed a new chief financial officer on Monday.", "neutral"),
    ("The annual shareholder meeting is scheduled for June.", "neutral"),
    ("The company filed its quarterly report with the exchange.", "neutral"),
    ("Trading volume was close to its 30-day average.", "neutral"),
    ("Management said it would provide an update at the investor conference.", "neutral"),
    ("The company operates across three regional business segments.", "neutral"),
    ("The stock ended the session at $145.20 per share.", "neutral"),
)


def load_financial_phrasebank(
    *,
    csv_path: Path | None = None,
    allow_download: bool = False,
    config: str = "sentences_allagree",
) -> pd.DataFrame:
    """Load PhraseBank from CSV/Hugging Face, falling back to labeled offline fixtures.

    Hugging Face downloads are opt-in. Integer class labels in PhraseBank map to
    negative=0, neutral=1, positive=2. ``is_demo`` distinguishes fallback rows.
    """
    if csv_path and csv_path.exists():
        frame = pd.read_csv(csv_path)
        sentence_col = "sentence" if "sentence" in frame else "text"
        if sentence_col not in frame or "label" not in frame:
            raise ValueError("Dataset CSV must contain sentence/text and label columns")
        result = frame[[sentence_col, "label"]].rename(columns={sentence_col: "text"})
        result["label"] = result["label"].map(_normalize_label)
        result["is_demo"] = False
        result["dataset"] = "local_csv"
        return result.dropna(subset=["text", "label"]).reset_index(drop=True)

    if allow_download:
        try:
            from datasets import load_dataset

            dataset = load_dataset("takala/financial_phrasebank", config, split="train")
            frame = dataset.to_pandas()[["sentence", "label"]].rename(columns={"sentence": "text"})
            frame["label"] = frame["label"].map(_normalize_label)
            frame["is_demo"] = False
            frame["dataset"] = "Financial PhraseBank"
            return frame.dropna(subset=["text", "label"]).reset_index(drop=True)
        except Exception as exc:
            LOGGER.warning("PhraseBank unavailable; using offline demo fixtures: %s", exc)

    return pd.DataFrame(
        [{"text": text, "label": label, "is_demo": True, "dataset": "synthetic offline fixture"} for text, label in OFFLINE_FIXTURE]
    )


def _normalize_label(value: Any) -> str | None:
    """Normalize PhraseBank's labels to lowercase sentiment names."""
    numeric = {0: "negative", 1: "neutral", 2: "positive"}
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return numeric.get(int(value))
    label = str(value).strip().lower()
    return label if label in {"positive", "neutral", "negative"} else None
