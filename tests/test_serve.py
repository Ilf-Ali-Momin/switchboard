import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from switchboard.config import settings
from switchboard.serve import app as serve
from switchboard.serve.decision_log import DecisionLog, summarize
from switchboard.serve.upstream import Upstream

BASE = "https://groq.test/openai/v1"
URL = f"{BASE}/chat/completions"
KEY = "test-proxy-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
SMALL, LARGE = settings().small_model, settings().large_model
THRESHOLD = 0.5


class FixedClassifier:
    def __init__(self, score: float) -> None:
        self.value = score
        self.seen: list[str] = []

    def score(self, prompt: str) -> float:
        self.seen.append(prompt)
        return self.value


class BrokenClassifier:
    def score(self, prompt: str) -> float:
        raise RuntimeError("model exploded")


def completion(request: httpx.Request) -> httpx.Response:
    model = json.loads(request.content)["model"]
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"hi from {model}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )


def body(model="auto", text="What is 2+2?", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": text}],
        **extra,
    }


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(classifier=None, load_classifier=False):
        log = DecisionLog(tmp_path / "decisions.jsonl")
        app = serve.create_app(
            classifier=classifier,
            upstream=Upstream(api_key="groq", base_url=BASE),
            proxy_api_key=KEY,
            threshold=THRESHOLD,
            decision_log=log,
            load_classifier=load_classifier,
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, log

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def sent_models(route) -> list[str]:
    return [json.loads(c.request.content)["model"] for c in route.calls]


# --- routing ---


@respx.mock
def test_auto_routes_to_small_above_threshold(make_client):
    route = respx.post(URL).mock(side_effect=completion)
    clf = FixedClassifier(0.9)
    client, log = make_client(clf)
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Router-Model"] == SMALL
    assert r.headers["X-Router-Score"] == "0.9000"
    assert sent_models(route) == [SMALL]
    assert clf.seen == ["What is 2+2?"]  # the last user message, not the system prompt
    assert r.json()["choices"][0]["message"]["content"] == f"hi from {SMALL}"


@respx.mock
def test_auto_routes_to_large_below_threshold(make_client):
    route = respx.post(URL).mock(side_effect=completion)
    client, _ = make_client(FixedClassifier(0.2))
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.headers["X-Router-Model"] == LARGE
    assert sent_models(route) == [LARGE]


@respx.mock
@pytest.mark.parametrize("model", [SMALL, LARGE])
def test_explicit_model_passes_through_without_classifying(make_client, model):
    route = respx.post(URL).mock(side_effect=completion)
    clf = FixedClassifier(0.9)
    client, _ = make_client(clf)
    r = client.post("/v1/chat/completions", json=body(model), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Router-Model"] == model
    assert r.headers["X-Router-Score"] == "none"
    assert sent_models(route) == [model] and clf.seen == []


@respx.mock
def test_unknown_model_is_400(make_client):
    route = respx.post(URL).mock(side_effect=completion)
    client, _ = make_client(FixedClassifier(0.9))
    r = client.post("/v1/chat/completions", json=body("gpt-4o"), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert route.call_count == 0


def test_last_user_text_handles_content_parts():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "new"}, {"type": "image_url", "image_url": {}}],
        },
    ]
    assert serve.last_user_text(messages) == "new"
    with pytest.raises(ValueError):
        serve.last_user_text([{"role": "system", "content": "x"}])


# --- fail safe ---


@respx.mock
def test_classifier_error_routes_to_large(make_client):
    route = respx.post(URL).mock(side_effect=completion)
    client, log = make_client(BrokenClassifier())
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Router-Model"] == LARGE
    assert sent_models(route) == [LARGE]
    assert log.records()[0]["classifier_error"] is True


@respx.mock
def test_missing_classifier_routes_to_large(make_client):
    respx.post(URL).mock(side_effect=completion)
    client, _ = make_client(classifier=None)
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.headers["X-Router-Model"] == LARGE
    assert client.get("/healthz").json()["classifier_loaded"] is False


@respx.mock
def test_small_failure_retries_once_on_large(make_client):
    def flaky(request):
        if json.loads(request.content)["model"] == SMALL:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return completion(request)

    route = respx.post(URL).mock(side_effect=flaky)
    client, log = make_client(FixedClassifier(0.9))
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Router-Model"] == LARGE
    assert sent_models(route) == [SMALL, LARGE]
    assert log.records()[0]["fallback"] is True


@respx.mock
def test_large_failure_is_returned_not_retried(make_client):
    route = respx.post(URL).mock(return_value=httpx.Response(429, json={"error": "slow down"}))
    client, log = make_client(FixedClassifier(0.1))
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 429
    assert route.call_count == 1
    assert log.records()[0]["status"] == 429


@respx.mock
def test_upstream_unreachable_falls_back_from_small(make_client):
    def down(request):
        if json.loads(request.content)["model"] == SMALL:
            raise httpx.ConnectError("no route")
        return completion(request)

    respx.post(URL).mock(side_effect=down)
    client, _ = make_client(FixedClassifier(0.9))
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200 and r.headers["X-Router-Model"] == LARGE


# --- streaming ---

SSE = (
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    b'data: {"choices":[],"x_groq":{"usage":{"prompt_tokens":7,"completion_tokens":2}}}\n\n'
    b"data: [DONE]\n\n"
)


@respx.mock
def test_stream_passes_sse_through_unchanged(make_client):
    respx.post(URL).mock(
        return_value=httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})
    )
    client, log = make_client(FixedClassifier(0.9))
    with client.stream("POST", "/v1/chat/completions", json=body(stream=True), headers=AUTH) as r:
        received = b"".join(r.iter_raw())
        assert r.headers["X-Router-Model"] == SMALL
        assert r.headers["content-type"].startswith("text/event-stream")
    assert received == SSE
    record = log.records()[0]
    assert record["stream"] is True
    assert (record["input_tokens"], record["output_tokens"]) == (7, 2)


@respx.mock
def test_stream_small_failure_falls_back_before_streaming(make_client):
    def flaky(request):
        if json.loads(request.content)["model"] == SMALL:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})

    respx.post(URL).mock(side_effect=flaky)
    client, _ = make_client(FixedClassifier(0.9))
    with client.stream("POST", "/v1/chat/completions", json=body(stream=True), headers=AUTH) as r:
        assert r.headers["X-Router-Model"] == LARGE
        assert b"".join(r.iter_raw()) == SSE


def test_stream_usage_parser_handles_split_chunks():
    parser = serve.StreamUsage()
    for i in range(0, len(SSE), 7):
        parser.feed(SSE[i : i + 7])
    assert parser.usage == {"prompt_tokens": 7, "completion_tokens": 2}


# --- auth and limits ---


def test_missing_or_wrong_key_is_401(make_client):
    client, _ = make_client(FixedClassifier(0.9))
    assert client.post("/v1/chat/completions", json=body()).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    r = client.post("/v1/chat/completions", json=body(), headers=bad)
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    assert client.get("/stats").status_code == 401


def test_body_over_limit_is_413(make_client):
    client, _ = make_client(FixedClassifier(0.9))
    huge = body(text="x" * (settings().serve.max_body_bytes + 10))
    r = client.post("/v1/chat/completions", json=huge, headers=AUTH)
    assert r.status_code == 413


def test_body_over_limit_without_content_length_is_413(make_client):
    client, _ = make_client(FixedClassifier(0.9))

    def chunks():
        for _ in range(settings().serve.max_body_bytes // 50_000 + 2):
            yield b"x" * 50_000

    r = client.post("/v1/chat/completions", content=chunks(), headers=AUTH)
    assert r.status_code == 413


def test_app_refuses_to_start_without_proxy_key(monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "")
    monkeypatch.setattr(serve, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="PROXY_API_KEY"):
        serve.create_app(classifier=FixedClassifier(0.5), threshold=0.5, load_classifier=False)


# --- decision log and stats ---


@respx.mock
def test_decision_log_never_stores_prompt_text(make_client):
    respx.post(URL).mock(side_effect=completion)
    client, log = make_client(FixedClassifier(0.9))
    secret = "my secret medical question about rash"
    client.post("/v1/chat/completions", json=body(text=secret), headers=AUTH)
    raw = log.path.read_text()
    assert "secret" not in raw and "rash" not in raw
    record = log.records()[0]
    expected_keys = {
        "timestamp",
        "request_id",
        "prompt_hash",
        "score",
        "threshold",
        "model",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
    }
    assert expected_keys <= record.keys()
    assert record["threshold"] == THRESHOLD and record["model"] == SMALL


@respx.mock
def test_stats_endpoint(make_client):
    respx.post(URL).mock(side_effect=completion)
    client, _ = make_client(FixedClassifier(0.9))
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    client.post("/v1/chat/completions", json=body(LARGE), headers=AUTH)
    stats = client.get("/stats", headers=AUTH).json()
    assert stats["requests"] == 2 and stats["share_small"] == 0.5
    assert stats["estimated_savings_usd"] > 0


def test_summarize_math():
    p = settings().prices_per_million_tokens
    records = [
        {"role": "small", "input_tokens": 1_000_000, "output_tokens": 0, "cost_usd": p.small.input},
        {"role": "large", "input_tokens": 1_000_000, "output_tokens": 0, "cost_usd": p.large.input},
        {"role": "large", "input_tokens": None, "output_tokens": None, "cost_usd": None},
    ]
    s = summarize(records)
    assert s["requests"] == 3 and s["priced_requests"] == 2
    assert s["share_small"] == round(1 / 3, 4)
    assert s["cost_if_always_large_usd"] == pytest.approx(2 * p.large.input)
    assert s["estimated_savings_usd"] == pytest.approx(p.large.input - p.small.input)
    assert summarize([])["share_small"] is None


def test_threshold_comes_from_results_json(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"operating_point": {"threshold": 0.37}}))
    assert serve.resolve_threshold(path) == 0.37
    assert serve.resolve_threshold(tmp_path / "missing.json") is None


@respx.mock
def test_no_threshold_routes_auto_to_large(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "RESULTS_PATH", tmp_path / "missing.json")
    respx.post(URL).mock(side_effect=completion)
    app = serve.create_app(
        classifier=FixedClassifier(0.99),
        upstream=Upstream(api_key="g", base_url=BASE),
        proxy_api_key=KEY,
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        load_classifier=False,
    )
    assert settings().serve.threshold is None
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.headers["X-Router-Model"] == LARGE
