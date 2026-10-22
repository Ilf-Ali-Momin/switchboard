import json

import httpx
import pandas as pd
import respx

from switchboard.config import settings
from switchboard.data import generate as gen
from switchboard.llm import LLMClient

BASE = "https://groq.test/openai/v1"


def reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    prompt = body["messages"][0]["content"]
    if "fail" in prompt:
        return httpx.Response(400, json={"error": "bad"})
    cut = "long" in prompt
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [
                {
                    "message": {"content": "" if cut else f"answer from {body['model']}"},
                    "finish_reason": "length" if cut else "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        },
    )


@respx.mock
async def test_generate_answers_both_roles_and_records_failures(tmp_path, monkeypatch):
    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=reply)
    monkeypatch.setattr(
        gen,
        "LLMClient",
        lambda: LLMClient(
            api_key="k", base_url=BASE, cache_path=tmp_path / "c.jsonl", max_retries=0
        ),
    )
    prompts = pd.DataFrame({"id": ["a", "b"], "prompt": ["hello", "please fail"]})
    out = await gen.generate(prompts)

    assert len(out) == 4
    assert set(out["role"]) == {"small", "large"}
    good = out[out["id"] == "a"].set_index("role")
    assert good.loc["small", "model"] == settings().small_model
    assert good.loc["large", "answer"] == f"answer from {settings().large_model}"
    assert good["output_tokens"].tolist() == [7, 7]
    assert out[out["id"] == "b"]["error"].notna().all()

    sent = json.loads(route.calls[0].request.content)
    assert sent["temperature"] == 0
    assert sent["reasoning_effort"] == settings().generation.reasoning_effort


def test_select_prompts_is_seeded():
    df = pd.DataFrame({"id": [str(i) for i in range(100)]})
    a = gen.select_prompts(df, 20, seed=42)
    assert len(a) == 20
    assert a.equals(gen.select_prompts(df, 20, seed=42))
    assert gen.select_prompts(df, None, seed=42) is df


@respx.mock
async def test_cut_off_answer_is_kept_as_a_result(tmp_path, monkeypatch):
    respx.post(f"{BASE}/chat/completions").mock(side_effect=reply)
    monkeypatch.setattr(
        gen,
        "LLMClient",
        lambda: LLMClient(api_key="k", base_url=BASE, cache_path=tmp_path / "c.jsonl"),
    )
    out = await gen.generate(pd.DataFrame({"id": ["a", "b"], "prompt": ["very long", "hi"]}))
    cut = out[out["id"] == "a"]
    assert cut["error"].isna().all()
    assert (cut["finish_reason"] == "length").all()
    assert (cut["answer"] == "").all()
    assert (out[out["id"] == "b"]["finish_reason"] == "stop").all()
