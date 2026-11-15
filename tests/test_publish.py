import pandas as pd

from switchboard.data import publish


def test_wide_makes_one_row_per_prompt():
    long = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "role": ["small", "large", "small", "large"],
            "score": [5, 4, 2, 5],
        }
    )
    out = publish.wide(long, ["score"]).set_index("id")
    assert out.loc["a", "small_score"] == 5
    assert out.loc["b", "large_score"] == 5
    assert len(out) == 2


def test_dataset_values_counts_failed_rows_and_balance():
    rows = pd.DataFrame(
        {
            "source": ["dolly", "dolly", "gsm8k"],
            "license": ["cc-by-sa-3.0", "cc-by-sa-3.0", "mit"],
            "category": ["qa", "qa", "math"],
            "split": ["train", "val", "test"],
            "small_ok": pd.array([True, None, False], dtype="boolean"),
        }
    )
    vals = publish.dataset_values(rows)
    assert vals["n_prompts"] == "3"
    assert vals["n_labelled"] == "2"
    assert vals["n_failed"] == "1"
    assert vals["split_counts"] == "train 1, validation 1, test 1"
    assert "| `dolly` | cc-by-sa-3.0 | 2 |" in vals["source_table"]
    assert "| math | 1 | 0% |" in vals["balance_table"]
