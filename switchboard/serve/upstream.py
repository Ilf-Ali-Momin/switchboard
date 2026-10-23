"""Calls to Groq's OpenAI compatible chat completions endpoint."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from switchboard.config import settings


class UpstreamError(Exception):
    """Groq answered with an error status or could not be reached."""

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"upstream returned {status}")
        self.status = status
        self.body = body


@dataclass
class UpstreamResponse:
    status: int
    body: dict[str, Any]


class Upstream:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.url = (base_url or settings().groq.base_url).rstrip("/") + "/chat/completions"
        self._http = httpx.AsyncClient(
            timeout=settings().serve.upstream_timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, body: dict[str, Any]) -> UpstreamResponse:
        try:
            resp = await self._http.post(self.url, json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(502, str(exc).encode()) from exc
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.content)
        return UpstreamResponse(resp.status_code, resp.json())

    async def open_stream(self, body: dict[str, Any]) -> httpx.Response:
        """Start a streaming call. Errors are raised before any byte reaches the client."""
        request = self._http.build_request("POST", self.url, json=body)
        try:
            resp = await self._http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise UpstreamError(502, str(exc).encode()) from exc
        if resp.status_code >= 400:
            body_bytes = await resp.aread()
            await resp.aclose()
            raise UpstreamError(resp.status_code, body_bytes)
        return resp


async def relay(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Pass the upstream SSE bytes through unchanged."""
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()
