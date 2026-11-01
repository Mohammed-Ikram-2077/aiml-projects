"""Compare FinBERT, VADER, and TF-IDF on a labeled financial text holdout."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import REPORTS_DIR
from src.load_datasets import load_financial_phrasebank
from src.sentiment import finbert_scores, fit_tfidf_baseline, vader_scores

LOGGER = logging.getLogger(__name__)
LABELS = ["negative", "neutral", "positive"]


def _metrics(gold: list[str], predicted: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "precision_macro": float(precision_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)),
    }


def evaluate_labeled_text(
    dataset: pd.DataFrame,
    *,
    local_finbert_only: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a stratified holdout and report all available classifiers."""
    if dataset.empty or dataset["label"].nunique() < 2:
        return {"status": "insufficient labeled rows", "metrics": {}}
    train, test = train_test_split(dataset, test_size=0.2, random_state=seed, stratify=dataset["label"])
    train_text = train["text"].astype(str).tolist()
    test_text = test["text"].astype(str).tolist()
    gold = test["label"].astype(str).tolist()
    results: dict[str, Any] = {"dataset": str(dataset["dataset"].iloc[0]), "dataset_rows": len(dataset), "test_rows": len(test), "models": {}}

    vader = vader_scores(test_text)
    vader_labels = vader["vader_label"].tolist()
    results["models"]["vader"] = {"status": "ok", "metrics": _metrics(gold, vader_labels), "confusion_matrix": confusion_matrix(gold, vader_labels, labels=LABELS).tolist()}

    baseline = fit_tfidf_baseline(train_text, train["label"].astype(str).tolist())
    tfidf_labels = baseline.predict(test_text).tolist()
    results["models"]["tfidf_logistic"] = {"status": "ok", "metrics": _metrics(gold, tfidf_labels), "confusion_matrix": confusion_matrix(gold, tfidf_labels, labels=LABELS).tolist()}

    try:
        prediction = finbert_scores(test_text, local_files_only=local_finbert_only)
        finbert_labels = prediction["finbert_label"].tolist()
        results["models"]["finbert"] = {"status": "ok", "metrics": _metrics(gold, finbert_labels), "confusion_matrix": confusion_matrix(gold, finbert_labels, labels=LABELS).tolist()}
    except Exception as exc:
        results["models"]["finbert"] = {"status": f"unavailable: {type(exc).__name__}: {exc}", "metrics": {}, "confusion_matrix": None}
    results["status"] = "ok"
    return results


def save_evaluation(results: dict[str, Any], output_dir: Path = REPORTS_DIR) -> None:
    """Save JSON/CSV model table and labeled confusion-matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sentiment_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = []
    for name, result in results.get("models", {}).items():
        table.append({"model": name, "status": result["status"], **result.get("metrics", {})})
    pd.DataFrame(table).to_csv(output_dir / "sentiment_evaluation.csv", index=False)
    available = [(name, result) for name, result in results.get("models", {}).items() if result.get("confusion_matrix")]
    if available:
        figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
        for axis, (name, result) in zip(axes[0], available):
            matrix = result["confusion_matrix"]
            axis.imshow(matrix, cmap="Blues")
            axis.set_title(name)
            axis.set_xticks(range(3), LABELS, rotation=25)
            axis.set_yticks(range(3), LABELS)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, str(matrix[i][j]), ha="center", va="center")
        figure.tight_layout()
        figure.savefig(output_dir / "sentiment_confusion_matrices.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--demo", action="store_true", help="Do not download model weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset = load_financial_phrasebank(csv_path=args.csv, allow_download=args.download_dataset)
    results = evaluate_labeled_text(dataset, local_finbert_only=args.demo)
    save_evaluation(results, args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
