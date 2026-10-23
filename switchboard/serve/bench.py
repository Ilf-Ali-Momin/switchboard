"""Measure the classifier's latency overhead per request on CPU, p50 and p95.

Scores test prompts one at a time, as the proxy does, after a short warmup.
Output: results/classifier_latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from switchboard.config import ROOT, settings
from switchboard.serve.classifier import RouterClassifier

OUTPUT_PATH = ROOT / "results" / "classifier_latency.json"
WARMUP = 5


def latency_summary(ms: list[float]) -> dict[str, float]:
    arr = np.asarray(ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def measure(classifier: RouterClassifier, prompts: list[str]) -> list[float]:
    for prompt in prompts[:WARMUP]:
        classifier.score(prompt)
    times = []
    for prompt in prompts:
        start = time.perf_counter()
        classifier.score(prompt)
        times.append((time.perf_counter() - start) * 1000)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / settings().serve.model_dir)
    parser.add_argument("--limit", type=int, default=200, help="number of test prompts")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    prompts = pd.read_parquet(ROOT / "data" / "prompts.parquet")
    test = prompts[prompts["split"] == "test"]["prompt"]
    sample = test.sample(n=min(args.limit, len(test)), random_state=settings().dataset.seed)
    times = measure(RouterClassifier(args.model_dir), sample.tolist())
    report = {
        "created": datetime.now(UTC).date().isoformat(),
        "device": "cpu",
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "n": len(times),
        **latency_summary(times),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
