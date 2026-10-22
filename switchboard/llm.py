"""Groq client over its OpenAI compatible API.

Every LLM call in the project goes through `LLMClient.chat`. Results are cached in an append
only JSONL file keyed by a hash of (model, messages, params), so reruns are free and a crashed
run resumes where it stopped. Rate limits and server errors are retried with exponential
backoff, and a semaphore caps concurrency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from switchboard.config import ROOT, settings

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Transient network failures only. LocalProtocolError (a malformed request) is not retried.
RETRY_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
Messages = list[dict[str, str]]
# A 429 asking us to wait longer than this is a daily quota, not a per minute limit.
DAILY_LIMIT_RETRY_AFTER_S = 300


class DailyLimitError(RuntimeError):
    """The provider's daily quota for a model is used up. Resume the run tomorrow."""


@dataclass
class Completion:
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str
    cached: bool = False


# Fields every cache entry must have. Entries written before a field existed are refetched.
CACHE_FIELDS = {f for f in Completion.__dataclass_fields__ if f != "cached"}


def cache_key(model: str, messages: Messages, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class DiskCache:
    """Append only JSONL cache. The last line for a key wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a crash
                self._entries[row["key"]] = row["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


def is_daily_limit(resp: httpx.Response) -> bool:
    if resp.status_code != 429:
        return False
    if "per day" in resp.text.lower():
        return True
    try:
        return float(resp.headers.get("retry-after", 0)) > DAILY_LIMIT_RETRY_AFTER_S
    except ValueError:
        return False


def backoff_delay(attempt: int, retry_after: str | None, base: float = 1.0) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(60.0, base * 2**attempt) + random.uniform(0, 0.5)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_path: Path | None = None,
        concurrency: int | None = None,
        max_retries: int | None = None,
        timeout_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        load_dotenv(ROOT / ".env")
        cfg = settings().groq
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env.")
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.cache = DiskCache(cache_path or ROOT / cfg.cache_path)
        self.max_retries = cfg.max_retries if max_retries is None else max_retries
        self._sem = asyncio.Semaphore(concurrency or cfg.concurrency)
        self.exhausted: set[str] = set()  # models whose daily quota is used up
        self._http = httpx.AsyncClient(
            timeout=timeout_s or cfg.timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat(self, model: str, messages: Messages, **params: Any) -> Completion:
        params.setdefault("temperature", 0)
        key = cache_key(model, messages, params)
        hit = self.cache.get(key)
        if hit is not None and hit.keys() >= CACHE_FIELDS:
            return Completion(**{**hit, "cached": True})
        async with self._sem:
            if model in self.exhausted:
                raise DailyLimitError(f"daily quota used up for {model}")
            completion = await self._post_with_retries(model, messages, params)
        self.cache.put(key, {k: v for k, v in asdict(completion).items() if k != "cached"})
        return completion

    async def _post_with_retries(
        self, model: str, messages: Messages, params: dict[str, Any]
    ) -> Completion:
        body = {"model": model, "messages": messages, **params}
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = await self._http.post(f"{self.base_url}/chat/completions", json=body)
            except RETRY_ERRORS:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(backoff_delay(attempt, None))
                continue
            latency = time.perf_counter() - start
            if is_daily_limit(resp):
                self.exhausted.add(model)
                raise DailyLimitError(f"daily quota used up for {model}: {resp.text[:200]}")
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                await asyncio.sleep(backoff_delay(attempt, resp.headers.get("retry-after")))
                continue
            resp.raise_for_status()
            return parse_completion(resp.json(), latency)
        raise RuntimeError("unreachable")


def parse_completion(data: dict[str, Any], latency_s: float) -> Completion:
    usage = data.get("usage") or {}
    choice = data["choices"][0]
    return Completion(
        model=data.get("model", ""),
        content=choice["message"].get("content") or "",
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        latency_s=round(latency_s, 4),
        finish_reason=choice.get("finish_reason") or "",
    )
