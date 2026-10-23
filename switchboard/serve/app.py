"""Step 7: OpenAI compatible routing proxy.

POST /v1/chat/completions takes the OpenAI request format.
    "model": "auto"                   the classifier picks the small or the large model
    "model": <small or large model>   passed through unchanged
    any other model                   400

The classifier reads the last user message, truncated to `dataset.max_prompt_tokens`.
`stream: true` relays the upstream SSE bytes unchanged. Responses carry X-Router-Model and
X-Router-Score. Fail safe: a classifier error routes to large, and a failed small model call is
retried once on large. Every request is appended to the decision log, which GET /stats sums up.

    uv run uvicorn switchboard.serve.app:app --port 8100
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from switchboard.config import ROOT, settings
from switchboard.serve.classifier import Classifier, timed_score
from switchboard.serve.decision_log import DecisionLog, cost_usd, prompt_hash
from switchboard.serve.upstream import Upstream, UpstreamError, relay

log = logging.getLogger("switchboard")
RESULTS_PATH = ROOT / "results" / "results.json"


# --- request helpers ---


def last_user_text(messages: Any) -> str:
    """Text of the last user message; content may be a string or a list of parts."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join(t for t in parts if t)
    raise ValueError("no user message")


def resolve_threshold(results_path: Path | None = None) -> float | None:
    """The configured threshold, else the one picked in step 6, else None (all to large)."""
    results_path = results_path or RESULTS_PATH
    configured = settings().serve.threshold
    if configured is not None:
        return configured
    if results_path.exists():
        return float(json.loads(results_path.read_text())["operating_point"]["threshold"])
    log.warning("no threshold (set serve.threshold or run step 6): auto routes to large")
    return None


def openai_error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def usage_from(obj: dict[str, Any]) -> dict[str, Any] | None:
    """OpenAI puts usage at the top level; Groq streams also send it under x_groq."""
    return obj.get("usage") or (obj.get("x_groq") or {}).get("usage")


class StreamUsage:
    """Picks the token usage out of SSE bytes as they pass by, across chunk boundaries."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            try:
                found = usage_from(json.loads(line[5:]))
            except (json.JSONDecodeError, AttributeError):
                continue
            if found:
                self.usage = found


# --- routing decision ---


@dataclass
class Decision:
    request_id: str
    prompt: str
    role: str
    mode: str  # "auto" or "passthrough"
    score: float | None = None
    threshold: float | None = None
    classifier_ms: float | None = None
    classifier_error: bool = False
    fallback: bool = False
    started: float = field(default_factory=time.perf_counter)

    def headers(self) -> dict[str, str]:
        return {
            "X-Router-Model": model_id(self.role),
            "X-Router-Score": "none" if self.score is None else f"{self.score:.4f}",
            "X-Request-ID": self.request_id,
        }

    def record(self, usage: dict[str, Any] | None, stream: bool, status: int) -> dict[str, Any]:
        tokens_in = (usage or {}).get("prompt_tokens")
        tokens_out = (usage or {}).get("completion_tokens")
        return {
            "request_id": self.request_id,
            "prompt_hash": prompt_hash(self.prompt),
            "mode": self.mode,
            "score": self.score,
            "threshold": self.threshold,
            "role": self.role,
            "model": model_id(self.role),
            "classifier_ms": None if self.classifier_ms is None else round(self.classifier_ms, 2),
            "classifier_error": self.classifier_error,
            "fallback": self.fallback,
            "stream": stream,
            "status": status,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": cost_usd(self.role if status < 400 else None, tokens_in, tokens_out),
            "latency_ms": round((time.perf_counter() - self.started) * 1000, 2),
        }


def model_id(role: str) -> str:
    return settings().small_model if role == "small" else settings().large_model


async def decide(
    requested: Any, prompt: str, classifier: Classifier | None, threshold: float | None
) -> Decision:
    request_id = uuid.uuid4().hex
    if requested == settings().small_model:
        return Decision(request_id, prompt, "small", "passthrough")
    if requested == settings().large_model:
        return Decision(request_id, prompt, "large", "passthrough")
    if requested != "auto":
        raise HTTPException(
            400,
            f"unknown model {requested!r}: use 'auto', "
            f"{settings().small_model!r} or {settings().large_model!r}",
        )
    decision = Decision(request_id, prompt, "large", "auto", threshold=threshold)
    try:
        if classifier is None or threshold is None:
            raise RuntimeError("classifier or threshold not available")
        decision.score, decision.classifier_ms = await timed_score(classifier, prompt)
        decision.role = "small" if decision.score >= threshold else "large"
    except Exception:  # fail safe: any classifier problem routes to large
        log.exception("classifier failed, routing to large")
        decision.classifier_error = True
    return decision


async def with_fallback(
    decision: Decision, body: dict[str, Any], call: Callable[[dict[str, Any]], Awaitable[Any]]
) -> Any:
    """Call upstream with the chosen model; a failed small call is retried once on large."""
    try:
        return await call({**body, "model": model_id(decision.role)})
    except UpstreamError:
        if decision.role != "small":
            raise
        log.warning("small model call failed, retrying on large")
        decision.role, decision.fallback = "large", True
        return await call({**body, "model": model_id("large")})


# --- body size limit ---


class BodyLimitMiddleware:
    """Rejects request bodies over the limit with 413, even without a Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return
        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body"):
                break

        async def replay() -> Message:
            # After the buffered body, wait on the real channel: a streaming response treats
            # http.disconnect as the client leaving and stops.
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = openai_error(413, f"request body over {self.max_bytes} bytes")
        await response(scope, receive, send)


# --- app ---


def create_app(
    classifier: Classifier | None = None,
    upstream: Upstream | None = None,
    proxy_api_key: str | None = None,
    threshold: float | None = None,
    decision_log: DecisionLog | None = None,
    load_classifier: bool = True,
) -> FastAPI:
    load_dotenv(ROOT / ".env")
    key = proxy_api_key or os.environ.get("PROXY_API_KEY", "")
    if not key:
        raise RuntimeError("PROXY_API_KEY is not set")
    cfg = settings().serve
    state: dict[str, Any] = {"classifier": classifier, "upstream": upstream}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        model_dir = ROOT / cfg.model_dir
        if state["classifier"] is None and load_classifier and not model_dir.exists():
            log.warning("no model at %s: every auto request goes to large", model_dir)
        elif state["classifier"] is None and load_classifier:
            try:
                from switchboard.serve.classifier import RouterClassifier

                state["classifier"] = RouterClassifier(model_dir)
            except Exception:
                log.exception("could not load the classifier: every auto request goes to large")
        state["upstream"] = state["upstream"] or Upstream()
        state["threshold"] = threshold if threshold is not None else resolve_threshold()
        yield
        await state["upstream"].aclose()

    app = FastAPI(title="Switchboard", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=cfg.max_body_bytes)
    decisions = decision_log or DecisionLog(ROOT / cfg.decision_log_path)

    def require_key(request: Request) -> None:
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "invalid or missing API key")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        kind = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return openai_error(exc.status_code, str(exc.detail), kind)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "classifier_loaded": state["classifier"] is not None}

    @app.get("/stats", dependencies=[Depends(require_key)])
    async def stats() -> dict[str, Any]:
        return decisions.stats()

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def chat_completions(request: Request) -> Any:
        try:
            body = json.loads(await request.body())
            prompt = last_user_text(body.get("messages"))
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            raise HTTPException(400, f"bad request: {exc}") from exc
        decision = await decide(body.get("model"), prompt, state["classifier"], state["threshold"])
        upstream: Upstream = state["upstream"]
        stream = bool(body.get("stream"))
        try:
            if stream:
                resp = await with_fallback(decision, body, upstream.open_stream)
            else:
                result = await with_fallback(decision, body, upstream.complete)
        except UpstreamError as exc:
            decisions.append(decision.record(None, stream, exc.status))
            return JSONResponse(_error_body(exc), exc.status, headers=decision.headers())

        if not stream:
            decisions.append(decision.record(usage_from(result.body), False, result.status))
            return JSONResponse(result.body, result.status, headers=decision.headers())

        async def body_iter() -> AsyncIterator[bytes]:
            usage = StreamUsage()
            try:
                async for chunk in relay(resp):
                    usage.feed(chunk)
                    yield chunk
            finally:
                decisions.append(decision.record(usage.usage, True, resp.status_code))

        return StreamingResponse(
            body_iter(),
            resp.status_code,
            headers=decision.headers(),
            media_type="text/event-stream",
        )

    return app


def _error_body(exc: UpstreamError) -> Any:
    try:
        return json.loads(exc.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "error": {"message": exc.body.decode(errors="replace")[:500], "type": "upstream_error"}
        }


def __getattr__(name: str) -> FastAPI:
    # `uvicorn switchboard.serve.app:app` builds the app on first access, so importing this
    # module (in tests, for example) needs no API key.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
