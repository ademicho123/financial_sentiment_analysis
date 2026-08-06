"""Benchmark fin_sentiment on the Financial PhraseBank dataset.

Run from the repository root:

    python benchmarks/benchmark_fin_sentiment.py --max-samples 100
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from fin_sentiment import __version__, analyze_text

DATASET_ID = "ghbacct/financial-phrasebank-all-agree-classification"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
LABEL_NAMES = ["bearish", "neutral", "bullish"]
PREDICTION_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}


def load_financial_phrasebank() -> Any:
    """Load the all-agree Financial PhraseBank split from Hugging Face."""
    return load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)


def evaluate(max_samples: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run inference and return aggregate metrics plus per-example predictions."""
    data = load_financial_phrasebank()
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples must be greater than zero")
        if max_samples < len(data):
            data = data.train_test_split(
                train_size=max_samples,
                seed=42,
                stratify_by_column="label",
            )["train"]

    gold_labels: list[int] = []
    predicted_labels: list[int] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, sample in enumerate(tqdm(data, desc="Benchmarking")):
        text = str(sample["text"])
        gold = int(sample["label"])
        prediction_started = time.perf_counter()
        result = analyze_text(text, explain=False)
        elapsed = time.perf_counter() - prediction_started
        sentiment = str(result["sentiment"]).lower()

        if sentiment not in PREDICTION_TO_ID:
            raise ValueError(f"Unexpected sentiment returned by model: {sentiment!r}")

        predicted = PREDICTION_TO_ID[sentiment]
        gold_labels.append(gold)
        predicted_labels.append(predicted)
        rows.append(
            {
                "id": index,
                "text": text,
                "gold_label": LABEL_NAMES[gold],
                "predicted_label": sentiment,
                "prediction_seconds": elapsed,
            }
        )

    total_seconds = time.perf_counter() - started
    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "samples": len(rows),
        "accuracy": accuracy_score(gold_labels, predicted_labels),
        "macro_f1": f1_score(gold_labels, predicted_labels, average="macro"),
        "weighted_f1": f1_score(gold_labels, predicted_labels, average="weighted"),
        "total_seconds": total_seconds,
        "average_seconds_per_sample": total_seconds / len(rows),
        "confusion_matrix": confusion_matrix(
            gold_labels, predicted_labels, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": report,
    }
    return metrics, rows


def save_results(
    metrics: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write machine-readable results, predictions, and a short report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = (
        "# fin_sentiment benchmark\n\n"
        f"- Dataset: `{metrics['dataset']}` (`{metrics['dataset_config']}`)\n"
        f"- Samples: {metrics['samples']}\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- Macro F1: {metrics['macro_f1']:.4f}\n"
        f"- Weighted F1: {metrics['weighted_f1']:.4f}\n"
        f"- Runtime: {metrics['total_seconds']:.2f} seconds\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fin_sentiment using Financial PhraseBank."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N samples for a quick test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "financial_phrasebank",
        help="Directory in which benchmark artifacts are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics, rows = evaluate(max_samples=args.max_samples)
        save_results(metrics, rows, args.output_dir)
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
