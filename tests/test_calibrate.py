import json

import pandas as pd
import pytest

from switchboard.data import calibrate as cal


def make_rows(n_per_cat: int = 40) -> pd.DataFrame:
    rows = []
    for cat, source in [
        ("factual_qa", "dolly"),
        ("general_instruction", "oasst1"),
        ("summarization", "dolly"),
        ("math", "gsm8k"),
    ]:
        for i in range(n_per_cat):
            rows.append(
                {
                    "id": f"{cat}-{i}",
                    "split": "train" if i % 4 else "val",
                    "source": source,
                    "category": cat,
                    "small_ok": i % 2 == 0,
                    "prompt": f"p{i}",
                    "small_answer": "SMALL",
                    "large_answer": "LARGE",
                }
            )
    return pd.DataFrame(rows)


def test_sample_uses_train_and_judged_sources_only():
    sample = cal.sample_rows(make_rows(), n=30, seed=42)
    assert len(sample) == 30
    assert set(sample["split"]) == {"train"}
    assert "math" not in set(sample["category"])
    assert sample["id"].is_unique


def test_sample_is_stratified_and_reproducible():
    rows = make_rows()
    a = cal.sample_rows(rows, n=30, seed=42)
    assert a["category"].value_counts().to_dict() == {
        "factual_qa": 10,
        "general_instruction": 10,
        "summarization": 10,
    }
    assert a.equals(cal.sample_rows(rows, n=30, seed=42))


def test_small_position_is_stable_and_mixed():
    positions = [cal.small_position(f"id{i}", 42) for i in range(200)]
    assert positions == [cal.small_position(f"id{i}", 42) for i in range(200)]
    assert 60 < positions.count("A") < 140


@pytest.mark.parametrize(
    ("choice", "pos", "expected"),
    [("A", "A", True), ("B", "A", False), ("tie", "B", True), ("A", "B", False)],
)
def test_human_small_ok(choice, pos, expected):
    assert cal.human_small_ok(choice, pos) is expected


def test_render_hides_model_names_and_places_answers():
    row = make_rows(1).iloc[0]
    text = cal.render(row, "B", 1, 1)
    assert text.index("SMALL") > text.index("ANSWER B:") > text.index("LARGE")
    lowered = text.lower().replace("small", "").replace("large", "")
    for word in ("gpt", "20b", "120b", "model"):
        assert word not in lowered


def test_session_saves_resumes_and_skips(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = cal.sample_rows(make_rows(), n=6, seed=1)
    keys = iter(["x", "a", "s", "q"])  # invalid key, label, skip, quit
    n = cal.label_session(rows, 1, read=lambda _: next(keys), show=lambda _: None, path=path)
    assert n == 1
    saved = cal.load_human_labels(path)
    first = saved[rows.iloc[0]["id"]]
    assert first["choice"] == "A"
    assert first["human_small_ok"] == (first["small_position"] == "A")

    keys = iter(["t"] * 10)
    n = cal.label_session(rows, 1, read=lambda _: next(keys), show=lambda _: None, path=path)
    assert n == 5  # the skipped row comes back; the labelled one does not
    assert len(cal.load_human_labels(path)) == 6


def test_agreement_report_math():
    merged = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b"],
            "human_small_ok": [True, False, True, False],
            "small_ok": [True, False, False, False],
        }
    )
    r = cal.agreement_report(merged)
    assert r["agreement"] == 0.75
    assert r["confusion"] == {
        "both_ok": 1,
        "both_not_ok": 2,
        "judge_ok_human_not": 0,
        "human_ok_judge_not": 1,
    }
    assert r["per_category"]["b"] == {"n": 2, "agreement": 0.5}
    assert r["cohen_kappa"] == 0.5
    json.dumps(r)  # serializable
