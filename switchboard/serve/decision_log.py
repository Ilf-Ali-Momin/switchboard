"""Append only JSONL log of routing decisions, and the summary behind GET /stats.

Each line holds a hash of the prompt, never its text.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.config import settings


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def cost_usd(role: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if role is None or input_tokens is None or output_tokens is None:
        return None
    price = getattr(settings().prices_per_million_tokens, role)
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


class DecisionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **record})
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(line + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def stats(self) -> dict[str, Any]:
        return summarize(self.records())


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Request count, share sent to small, and estimated savings against always large.

    The always-large cost reprices each request's real token usage at large model prices.
    Output length would differ on the large model, so this is an estimate.
    """
    priced = [r for r in records if r.get("cost_usd") is not None]
    actual = sum(r["cost_usd"] for r in priced)
    if_large = sum(cost_usd("large", r["input_tokens"], r["output_tokens"]) for r in priced)
    small = sum(r.get("role") == "small" for r in records)
    return {
        "requests": len(records),
        "share_small": round(small / len(records), 4) if records else None,
        "fallbacks": sum(bool(r.get("fallback")) for r in records),
        "classifier_errors": sum(bool(r.get("classifier_error")) for r in records),
        "priced_requests": len(priced),
        "cost_usd": round(actual, 6),
        "cost_if_always_large_usd": round(if_large, 6),
        "estimated_savings_usd": round(if_large - actual, 6),
        "estimated_savings_share": round(1 - actual / if_large, 4) if if_large else None,
    }
