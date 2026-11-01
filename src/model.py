"""Score financial text with a VADER baseline and ProsusAI FinBERT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.collect import PROJECT_ROOT, _safe_ticker
from src.preprocess import PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "neutral", "negative")
SCORED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _label_from_score(score: float, threshold: float = 0.05) -> str:
    """Map a signed sentiment score to positive, neutral, or negative."""
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def score_vader(texts: list[str]) -> list[dict[str, Any]]:
    """Score texts with VADER; compound score is already in [-1, 1]."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    results: list[dict[str, Any]] = []
    for text in texts:
        score = float(analyzer.polarity_scores(text or "")["compound"])
        results.append({"label": _label_from_score(score), "score": score})
    return results


def _resolve_device(torch: Any, requested: str) -> Any:
    """Resolve auto/cpu/cuda/mps to a torch device with clear availability errors."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this device")
    return device


def score_finbert(
    texts: list[str],
    *,
    batch_size: int = 16,
    model_name: str = MODEL_NAME,
    device: str = "auto",
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Batch-score text with a Hugging Face sequence classification model.

    FinBERT's signed score is ``P(positive) - P(negative)``, bounded in [-1, 1].
    Model weights are downloaded/cached by Transformers on the first call.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if not texts:
        return []

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch_device = _resolve_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(torch_device)
    model.eval()

    # Read class names from model config instead of assuming logit ordering.
    id2label = getattr(model.config, "id2label", {}) or {}
    label_by_index: dict[int, str] = {}
    for index, raw_label in id2label.items():
        normalized = str(raw_label).strip().lower()
        if normalized in LABELS:
            label_by_index[int(index)] = normalized
    # ProsusAI/finbert uses negative, neutral, positive output classes in this order.
    if set(label_by_index.values()) != set(LABELS):
        if model_name == MODEL_NAME:
            label_by_index = {0: "negative", 1: "neutral", 2: "positive"}
        else:
            raise ValueError(f"Could not identify positive/neutral/negative model labels: {id2label}")

    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            batch = [(text or "") for text in texts[offset : offset + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(torch_device) for name, tensor in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()
            for row in probabilities:
                by_label = {label_by_index[index]: float(row[index]) for index in range(len(row))}
                signed_score = by_label["positive"] - by_label["negative"]
                results.append(
                    {
                        "label": max(by_label, key=by_label.get),
                        "score": signed_score,
                        "positive_probability": by_label["positive"],
                        "neutral_probability": by_label["neutral"],
                        "negative_probability": by_label["negative"],
                    }
                )
    return results


def score_dataframe(
    frame: pd.DataFrame,
    *,
    method: Literal["vader", "finbert", "both"] = "both",
    batch_size: int = 16,
    model_name: str = MODEL_NAME,
    device: str = "auto",
) -> pd.DataFrame:
    """Append selected sentiment labels/scores to a processed dataframe."""
    if "text" not in frame.columns:
        raise ValueError("input dataframe must contain a 'text' column")
    output = frame.copy()
    texts = output["text"].fillna("").astype(str).tolist()
    if method in ("vader", "both"):
        vader = score_vader(texts)
        output["vader_label"] = [result["label"] for result in vader]
        output["vader_score"] = [result["score"] for result in vader]
    if method in ("finbert", "both"):
        finbert = score_finbert(texts, batch_size=batch_size, model_name=model_name, device=device)
        output["finbert_label"] = [result["label"] for result in finbert]
        output["finbert_score"] = [result["score"] for result in finbert]
        for probability in ("positive_probability", "neutral_probability", "negative_probability"):
            output[f"finbert_{probability}"] = [result[probability] for result in finbert]
    return output


def main() -> None:
    """Score a ticker's processed records and save a sentiment CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--model", choices=("vader", "finbert", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCORED_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ticker = _safe_ticker(args.ticker)
    input_path = args.input_dir / f"{ticker}_processed.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input not found: {input_path}; run src.preprocess first")
    frame = pd.read_csv(input_path)
    LOGGER.info("Scoring %d records for %s with %s", len(frame), ticker, args.model)
    scored = score_dataframe(
        frame,
        method=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{ticker}_sentiment.csv"
    scored.to_csv(output_path, index=False)
    LOGGER.info("Saved sentiment output to %s", output_path)


if __name__ == "__main__":
    main()
