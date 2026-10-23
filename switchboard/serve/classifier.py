"""The routing classifier as the proxy uses it."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Protocol


class Classifier(Protocol):
    def score(self, prompt: str) -> float: ...


class RouterClassifier:
    """P(small model is good enough) from the fine tuned model, on CPU."""

    def __init__(self, model_dir: Path) -> None:
        from switchboard.router_model import RouterModel  # torch loads only when serving

        self.model = RouterModel(model_dir, device="cpu")

    def score(self, prompt: str) -> float:
        return self.model.predict([prompt])[0]


async def timed_score(classifier: Classifier, prompt: str) -> tuple[float, float]:
    """(score, milliseconds), run in a thread so the event loop keeps serving."""
    start = time.perf_counter()
    score = await asyncio.to_thread(classifier.score, prompt)
    return score, (time.perf_counter() - start) * 1000
