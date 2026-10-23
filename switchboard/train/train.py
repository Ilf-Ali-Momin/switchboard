"""Step 5: fine tune the router classifier.

A binary classifier on top of `encoder_model`. The input is the prompt only; the output is the
probability that the small model is good enough (label small_ok). Training uses the train split.
Learning rates from `training.learning_rates` are compared on val ROC AUC and the best model is
saved to `training.output_dir`. The test split is never read here.

Input:  data/prompts.parquet, data/labels.parquet
Output: models/router/ (model and tokenizer), results/train_metrics.json

Runs on CPU, Apple MPS or a CUDA GPU (free Colab or Kaggle). notebooks/train_colab.ipynb runs
this same module.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from switchboard.config import ROOT, settings

DATA_DIR = ROOT / "data"
METRICS_PATH = ROOT / "results" / "train_metrics.json"
ID2LABEL = {0: "needs_large", 1: "small_ok"}


@dataclass
class RunResult:
    learning_rate: float
    val_roc_auc: float | None
    val_accuracy: float
    model_dir: Path


# --- data ---


def load_splits(labels_path: Path, limit: int | None, seed: int) -> tuple[pd.DataFrame, ...]:
    """Train and val rows with the prompt text and an integer label."""
    labels = pd.read_parquet(labels_path)
    prompts = pd.read_parquet(DATA_DIR / "prompts.parquet")[["id", "prompt"]]
    df = labels.merge(prompts, on="id")
    df["label"] = df["small_ok"].astype(int)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    if limit is not None:
        train = train.sample(n=min(limit, len(train)), random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def to_dataset(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["prompt", "label"]], preserve_index=False)
    return ds.map(
        lambda b: tokenizer(b["prompt"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["prompt"],
    )


# --- metrics ---


def small_ok_probability(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC, or None when val holds only one class."""
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = small_ok_probability(np.asarray(logits))
    labels = np.asarray(pred.label_ids)
    auc = safe_roc_auc(labels, probs)
    return {
        "roc_auc": float("nan") if auc is None else auc,
        "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
    }


# --- training ---


def training_args(out_dir: Path, lr: float, max_steps: int | None) -> TrainingArguments:
    cfg = settings().training
    cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,  # a float below 1 is a share of total steps
        eval_strategy="epoch" if max_steps is None else "no",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=settings().dataset.seed,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
    )


def train_one(
    lr: float,
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    work_dir: Path,
    max_steps: int | None,
) -> RunResult:
    set_seed(settings().dataset.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings().encoder_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    )
    trainer = Trainer(
        model=model,
        args=training_args(work_dir / "trainer", lr, max_steps),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    model_dir = work_dir / f"lr-{lr:g}"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    auc = metrics.get("eval_roc_auc")
    return RunResult(
        learning_rate=lr,
        val_roc_auc=None if auc is None or np.isnan(auc) else round(float(auc), 4),
        val_accuracy=round(float(metrics["eval_accuracy"]), 4),
        model_dir=model_dir,
    )


def pick_best(results: list[RunResult]) -> RunResult:
    """Highest val ROC AUC; runs without an AUC rank last; ties go to the earlier run."""
    return max(results, key=lambda r: -1.0 if r.val_roc_auc is None else r.val_roc_auc)


def class_balance(df: pd.DataFrame) -> dict[str, float | int]:
    return {"n": int(len(df)), "small_ok_share": round(float(df["label"].mean()), 3)}


def write_metrics(path: Path, best: RunResult, results: list[RunResult], train, val) -> dict:
    report = {
        "encoder_model": settings().encoder_model,
        "created": datetime.now(UTC).date().isoformat(),
        "seed": settings().dataset.seed,
        "train": class_balance(train),
        "val": class_balance(val),
        "best_learning_rate": best.learning_rate,
        "val_roc_auc": best.val_roc_auc,
        "val_accuracy": best.val_accuracy,
        "runs": [
            {
                "learning_rate": r.learning_rate,
                "val_roc_auc": r.val_roc_auc,
                "val_accuracy": r.val_accuracy,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / settings().training.output_dir)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="use only N train rows")
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    parser.add_argument("--max-length", type=int, default=settings().dataset.max_prompt_tokens)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=settings().training.learning_rates
    )
    args = parser.parse_args()

    seed = settings().dataset.seed
    train, val = load_splits(args.labels, args.limit, seed)
    print(f"train {class_balance(train)}  val {class_balance(val)}")
    tokenizer = AutoTokenizer.from_pretrained(settings().encoder_model)
    train_ds = to_dataset(train, tokenizer, args.max_length)
    val_ds = to_dataset(val, tokenizer, args.max_length)

    with tempfile.TemporaryDirectory(prefix="router-train-") as tmp:
        results = []
        for lr in args.learning_rates:
            result = train_one(lr, train_ds, val_ds, tokenizer, Path(tmp), args.max_steps)
            print(f"lr={lr:g}  val_roc_auc={result.val_roc_auc}  val_acc={result.val_accuracy}")
            results.append(result)
        best = pick_best(results)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        shutil.copytree(best.model_dir, args.output_dir)

    report = write_metrics(args.metrics_path, best, results, train, val)
    print(json.dumps({k: report[k] for k in ("best_learning_rate", "val_roc_auc")}, indent=2))
    print(f"saved model to {args.output_dir} and metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
