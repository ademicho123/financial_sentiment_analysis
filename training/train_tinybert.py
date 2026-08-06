"""Fine-tune and save TinyBERT for three-class financial sentiment."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
DATASET_ID = "ghbacct/financial-phrasebank-all-agree-classification"
ID2LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "fin_sentiment" / "model"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(tokenizer: AutoTokenizer, max_length: int) -> tuple[DataLoader, DataLoader]:
    dataset = load_dataset(DATASET_ID, "default")

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    encoded = dataset.map(tokenize, batched=True, remove_columns=["text"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return (
        DataLoader(encoded["train"], batch_size=16, shuffle=True),
        DataLoader(encoded["test"], batch_size=32),
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} ({args.threads} CPU threads)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_loader, test_loader = prepare_data(tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += outputs.loss.item()

            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )

        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / len(train_loader)
        history.append(metrics)
        print(
            f"Validation epoch {epoch}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    results: dict[str, object] = {
        "base_model": BASE_MODEL,
        "dataset": DATASET_ID,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "epochs": args.epochs,
        "batch_size": 16,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "final_metrics": history[-1],
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved trained model to: {args.output_dir}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

