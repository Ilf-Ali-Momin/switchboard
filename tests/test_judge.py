import json

import httpx
import pandas as pd
import pytest
import respx

from switchboard.data import judge as jd
from switchboard.llm import LLMClient

BASE = "https://groq.test/openai/v1"


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "judge",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@pytest.fixture
def client(tmp_path):
    return LLMClient(api_key="k", base_url=BASE, cache_path=tmp_path / "c.jsonl", max_retries=0)


# --- label logic ---


@pytest.mark.parametrize(
    ("small", "large", "margin", "expected"),
    [(5, 5, 0, True), (4, 5, 0, False), (5, 3, 0, True), (4, 5, 1, True), (3, 5, 1, False)],
)
def test_small_ok(small, large, margin, expected):
    assert jd.small_ok(small, large, margin) is expected


def test_make_labels_excludes_prompts_with_a_failed_judgment():
    prompts = pd.DataFrame(
        {"id": ["a", "b", "c"], "split": "train", "source": "s", "category": ["x", "x", "y"]}
    )
    judgments = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c"],
            "role": ["small", "large"] * 3,
            "score": [5, 5, 2, 4, None, 3],
        }
    )
    labels = jd.make_labels(prompts, judgments, margin=0)
    assert labels["id"].tolist() == ["a", "b"]
    assert labels["small_ok"].tolist() == [True, False]
    assert labels[["small_score", "large_score"]].values.tolist() == [[5, 5], [2, 4]]


def test_imbalanced_categories_flags_both_directions():
    labels = pd.DataFrame(
        {
            "category": ["math"] * 10 + ["code"] * 10 + ["qa"] * 10,
            "small_ok": [True] * 9 + [False] + [False] * 10 + [True] * 5 + [False] * 5,
        }
    )
    flagged = jd.imbalanced_categories(jd.label_balance(labels))
    assert set(flagged) == {"math", "code"}


# --- parsing ---


def test_parse_judgment_accepts_valid_json():
    assert jd.parse_judgment('{"score": 4, "reason": " fine "}') == (4, "fine")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        '{"score": 6, "reason": "x"}',
        '{"score": "4", "reason": "x"}',
        '{"score": true, "reason": "x"}',
        '{"score": 3}',
        '{"score": 3, "reason": "  "}',
    ],
)
def test_parse_judgment_rejects_invalid(text):
    with pytest.raises(jd.InvalidJudgment):
        jd.parse_judgment(text)


def test_judge_messages_hide_model_identity_and_mark_empty_answers():
    msgs = jd.judge_messages("What is 2+2?", "   ")
    text = json.dumps(msgs).lower()
    assert "(empty answer)" in text
    for name in ("small", "large", "gpt-oss", "20b", "120b"):
        assert name not in msgs[1]["content"].lower()


# --- retry behaviour ---


@respx.mock
async def test_invalid_reply_is_retried_once_then_succeeds(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[completion("oops"), completion('{"score": 3, "reason": "ok"}')]
    )
    j = await jd.judge_answer(client, "p", "a")
    assert (j.score, j.attempts, j.error) == (3, 2, None)
    assert route.call_count == 2
    retry_body = json.loads(route.calls[1].request.content)
    assert retry_body["messages"][-1]["content"] == jd.RETRY_MESSAGE
    await client.aclose()


@respx.mock
async def test_two_invalid_replies_mark_the_row_failed(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[completion("oops"), httpx.Response(400, json={"error": "json_validate"})]
    )
    j = await jd.judge_answer(client, "p", "a")
    assert j.score is None and j.attempts == 2
    assert j.error.startswith("InvalidJudgment")
    assert route.call_count == 2
    await client.aclose()


@respx.mock
async def test_judge_request_uses_json_mode_and_temperature_zero(client):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=completion('{"score": 5, "reason": "good"}')
    )
    await jd.judge_answer(client, "p", "a")
    body = json.loads(route.calls[0].request.content)
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    await client.aclose()


def test_answered_prompts_keeps_only_prompts_with_both_answers():
    prompts = pd.DataFrame({"id": ["a", "b", "c"], "prompt": ["pa", "pb", "pc"]})
    answers = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c"],
            "role": ["small", "large", "small", "large", "small"],
            "answer": ["s", "l", "s", None, "s"],
            "error": [None, None, None, "DailyLimitError", None],
        }
    )
    out = jd.answered_prompts(prompts, answers)
    assert out["id"].tolist() == ["a"]
    assert out.loc[0, "small_answer"] == "s" and out.loc[0, "large_answer"] == "l"


def test_make_labels_with_no_valid_scores_returns_empty():
    prompts = pd.DataFrame({"id": ["a"], "split": "train", "source": "s", "category": "x"})
    judgments = pd.DataFrame({"id": ["a", "a"], "role": ["small", "large"], "score": [None, None]})
    assert jd.make_labels(prompts, judgments, margin=0).empty
