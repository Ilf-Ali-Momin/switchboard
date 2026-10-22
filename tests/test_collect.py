import pandas as pd

from switchboard.data.collect import (
    assign_splits,
    dedupe,
    mbpp_prompt,
    normalize,
    source_quotas,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_normalize_ignores_case_punctuation_and_spacing():
    assert normalize("  What IS 2+2?\n") == normalize("what is 2 2")


def test_dedupe_drops_near_duplicates_keeps_first():
    df = pd.DataFrame(
        {"prompt": ["Hello, world!", "hello world", "Something else"], "n": [1, 2, 3]}
    )
    out = dedupe(df)
    assert out["n"].tolist() == [1, 3]


def test_source_quotas_sum_to_total():
    mix = {"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.2}
    for total in (20, 2000, 7, 1):
        assert sum(source_quotas(total, mix).values()) == total
    assert source_quotas(2000, mix) == {"a": 800, "b": 400, "c": 400, "d": 400}


def make_df(n_per_source: int) -> pd.DataFrame:
    rows = [{"source": s, "i": i} for s in ("a", "b") for i in range(n_per_source)]
    return pd.DataFrame(rows)


def test_splits_are_stratified_by_source():
    df = make_df(100)
    split = assign_splits(df, RATIOS, seed=42)
    counts = pd.crosstab(df["source"], split)
    assert counts.loc["a"].to_dict() == {"test": 15, "train": 70, "val": 15}
    assert counts.loc["b"].to_dict() == {"test": 15, "train": 70, "val": 15}


def test_splits_are_reproducible_and_seed_dependent():
    df = make_df(50)
    assert assign_splits(df, RATIOS, 42).equals(assign_splits(df, RATIOS, 42))
    assert not assign_splits(df, RATIOS, 42).equals(assign_splits(df, RATIOS, 7))


def test_mbpp_prompt_includes_first_test():
    p = mbpp_prompt("Write f.", ["assert f(1) == 2", "assert f(2) == 3"])
    assert "assert f(1) == 2" in p
    assert "assert f(2) == 3" not in p
