import httpx
import pytest
import respx

from switchboard.llm import DailyLimitError, DiskCache, LLMClient, cache_key, is_daily_limit

BASE = "https://groq.test/openai/v1"
MESSAGES = [{"role": "user", "content": "hi"}]


def ok_body(content: str = "hello") -> dict:
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


@pytest.fixture
def client(tmp_path):
    return LLMClient(api_key="k", base_url=BASE, cache_path=tmp_path / "cache.jsonl", max_retries=2)


def test_cache_key_is_order_independent_and_param_sensitive():
    a = cache_key("m", MESSAGES, {"temperature": 0, "max_tokens": 5})
    b = cache_key("m", MESSAGES, {"max_tokens": 5, "temperature": 0})
    c = cache_key("m", MESSAGES, {"temperature": 0.5, "max_tokens": 5})
    assert a == b
    assert a != c
    assert a != cache_key("other", MESSAGES, {"temperature": 0, "max_tokens": 5})


def test_disk_cache_survives_reload_and_torn_line(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = DiskCache(path)
    cache.put("a", {"x": 1})
    with path.open("a") as f:
        f.write('{"key": "b", "val')  # simulated crash mid-write
    reloaded = DiskCache(path)
    assert reloaded.get("a") == {"x": 1}
    assert reloaded.get("b") is None


@respx.mock
async def test_second_call_hits_cache(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    first = await client.chat("m", MESSAGES)
    second = await client.chat("m", MESSAGES)
    assert route.call_count == 1
    assert not first.cached and second.cached
    assert second.content == "hello"
    assert (second.input_tokens, second.output_tokens) == (3, 2)
    await client.aclose()


@respx.mock
async def test_cache_persists_across_clients(tmp_path):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    path = tmp_path / "cache.jsonl"
    async with LLMClient(api_key="k", base_url=BASE, cache_path=path) as c1:
        await c1.chat("m", MESSAGES)
    async with LLMClient(api_key="k", base_url=BASE, cache_path=path) as c2:
        result = await c2.chat("m", MESSAGES)
    assert route.call_count == 1
    assert result.cached


@respx.mock
async def test_retries_on_429_then_succeeds(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json=ok_body("after retry")),
        ]
    )
    result = await client.chat("m", MESSAGES)
    assert route.call_count == 2
    assert result.content == "after retry"
    await client.aclose()


@respx.mock
async def test_gives_up_after_max_retries_and_caches_nothing(client):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(503, headers={"retry-after": "0"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat("m", MESSAGES)
    assert len(client.cache) == 0
    await client.aclose()


def test_missing_api_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr("switchboard.llm.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        LLMClient(base_url=BASE, cache_path=tmp_path / "c.jsonl")


@respx.mock
async def test_local_protocol_error_is_not_retried(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=httpx.LocalProtocolError("Illegal header value")
    )
    with pytest.raises(httpx.LocalProtocolError):
        await client.chat("m", MESSAGES)
    assert route.call_count == 1
    await client.aclose()


@respx.mock
async def test_connect_error_is_retried(client, monkeypatch):
    monkeypatch.setattr("switchboard.llm.backoff_delay", lambda *a: 0)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[httpx.ConnectError("down"), httpx.Response(200, json=ok_body())]
    )
    result = await client.chat("m", MESSAGES)
    assert route.call_count == 2
    assert result.content == "hello"
    await client.aclose()


@respx.mock
async def test_finish_reason_is_recorded_and_cached(client):
    body = ok_body("")
    body["choices"][0]["finish_reason"] = "length"
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=body))
    first = await client.chat("m", MESSAGES)
    second = await client.chat("m", MESSAGES)
    assert first.finish_reason == second.finish_reason == "length"
    assert second.cached
    await client.aclose()


@respx.mock
async def test_cache_entry_missing_a_field_is_refetched(tmp_path):
    path = tmp_path / "cache.jsonl"
    key = cache_key("m", MESSAGES, {"temperature": 0})
    old = {"model": "m", "content": "old", "input_tokens": 1, "output_tokens": 1, "latency_s": 0.1}
    DiskCache(path).put(key, old)  # written before finish_reason existed
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body("new"))
    )
    async with LLMClient(api_key="k", base_url=BASE, cache_path=path) as c:
        result = await c.chat("m", MESSAGES)
    assert route.call_count == 1
    assert result.content == "new" and not result.cached


@respx.mock
async def test_daily_limit_stops_that_model_only(tmp_path):
    def reply(request: httpx.Request) -> httpx.Response:
        if b'"exhausted"' in request.content:
            return httpx.Response(
                429, json={"error": {"message": "Rate limit reached on requests per day (RPD)"}}
            )
        return httpx.Response(200, json=ok_body())

    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=reply)
    async with LLMClient(api_key="k", base_url=BASE, cache_path=tmp_path / "c.jsonl") as c:
        with pytest.raises(DailyLimitError):
            await c.chat("exhausted", MESSAGES)
        with pytest.raises(DailyLimitError):
            await c.chat("exhausted", [{"role": "user", "content": "other"}])
        assert route.call_count == 1  # no request sent once the quota is known to be gone
        assert (await c.chat("fresh", MESSAGES)).content == "hello"


def test_long_retry_after_counts_as_daily_limit():
    assert is_daily_limit(httpx.Response(429, headers={"retry-after": "3600"}))
    assert not is_daily_limit(httpx.Response(429, headers={"retry-after": "20"}))
    assert not is_daily_limit(httpx.Response(503, headers={"retry-after": "3600"}))
