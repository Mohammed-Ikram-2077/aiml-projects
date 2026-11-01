"""FinBERT, VADER, and TF-IDF/logistic-regression sentiment models."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DEFAULT_CONFIG, MODEL_NAME

LOGGER = logging.getLogger(__name__)
LABELS = ("negative", "neutral", "positive")
_VADER = SentimentIntensityAnalyzer()


def vader_scores(texts: list[str]) -> pd.DataFrame:
    """Return VADER labels and compound scores bounded to [-1, 1]."""
    rows = []
    for text in texts:
        score = float(_VADER.polarity_scores(text or "")["compound"])
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        rows.append({"vader_label": label, "vader_score": score})
    return pd.DataFrame(rows, columns=["vader_label", "vader_score"])


def finbert_scores(
    texts: list[str],
    *,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    model_name: str = MODEL_NAME,
    local_files_only: bool = False,
) -> pd.DataFrame:
    """Batched FinBERT probabilities, labels, and positive-minus-negative score.

    Uses CUDA/MPS when available, otherwise CPU. The loader respects
    ``local_files_only`` for deterministic offline demo runs.
    """
    columns = ["finbert_positive_probability", "finbert_neutral_probability", "finbert_negative_probability", "finbert_label", "finbert_score"]
    if not texts:
        return pd.DataFrame(columns=columns)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer, model = _load_finbert(model_name, local_files_only, str(device))
    ids = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
    if not all(label in ids for label in LABELS):
        if model_name != MODEL_NAME:
            raise ValueError(f"Model does not expose expected sentiment labels: {ids}")
        ids = {"negative": 0, "neutral": 1, "positive": 2}

    rows: list[dict[str, Any]] = []
    for offset in tqdm(range(0, len(texts), batch_size), desc="FinBERT", unit="batch"):
        batch = [str(value or "") for value in texts[offset : offset + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        for vector in probs:
            result = {f"finbert_{label}_probability": float(vector[index]) for label, index in ids.items()}
            result["finbert_score"] = result["finbert_positive_probability"] - result["finbert_negative_probability"]
            result["finbert_label"] = max(LABELS, key=lambda label: result[f"finbert_{label}_probability"])
            rows.append(result)
    return pd.DataFrame(rows, columns=columns)


@lru_cache(maxsize=4)
def _load_finbert(model_name: str, local_files_only: bool, device_name: str) -> tuple[Any, Any]:
    """Load and cache FinBERT weights once per process/device."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def fit_tfidf_baseline(texts: list[str], labels: list[str]) -> Any:
    """Fit a TF-IDF + logistic-regression multiclass financial baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=DEFAULT_CONFIG.random_seed)),
        ]
    )
    model.fit(texts, labels)
    return model


def tfidf_scores(model: Any, texts: list[str]) -> pd.DataFrame:
    """Predict labels and signed scores from a fitted TF-IDF classifier."""
    if not texts:
        return pd.DataFrame(columns=["tfidf_label", "tfidf_score"])
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    positive_index, negative_index = classes.index("positive"), classes.index("negative")
    score = probabilities[:, positive_index] - probabilities[:, negative_index]
    predicted = model.predict(texts)
    return pd.DataFrame({"tfidf_label": predicted, "tfidf_score": score})


def score_frame(
    frame: pd.DataFrame,
    *,
    labeled_training: pd.DataFrame | None = None,
    include_finbert: bool = True,
    finbert_local_only: bool = False,
    batch_size: int = DEFAULT_CONFIG.batch_size,
) -> pd.DataFrame:
    """Append baseline and FinBERT scores, gracefully recording unavailable weights."""
    if "text" not in frame:
        raise ValueError("Input records require a text column")
    output = frame.copy().reset_index(drop=True)
    texts = output["text"].fillna("").astype(str).tolist()
    vader = vader_scores(texts)
    for col in vader:
        output[col] = vader[col].to_numpy()

    if labeled_training is not None and not labeled_training.empty:
        training = labeled_training.dropna(subset=["text", "label"])
        baseline = fit_tfidf_baseline(training["text"].astype(str).tolist(), training["label"].astype(str).tolist())
        baseline_scores = tfidf_scores(baseline, texts)
        for col in baseline_scores:
            output[col] = baseline_scores[col].to_numpy()
    else:
        output["tfidf_label"], output["tfidf_score"] = None, np.nan

    if include_finbert:
        try:
            result = finbert_scores(texts, batch_size=batch_size, local_files_only=finbert_local_only)
            for col in result:
                output[col] = result[col].to_numpy()
            output["finbert_status"] = "available"
        except Exception as exc:
            LOGGER.warning("FinBERT unavailable (%s); FinBERT columns are empty", exc)
            for col in ("finbert_positive_probability", "finbert_neutral_probability", "finbert_negative_probability", "finbert_score"):
                output[col] = np.nan
            output["finbert_label"] = None
            output["finbert_status"] = f"unavailable: {type(exc).__name__}"
    else:
        output["finbert_status"] = "skipped"
    return output
