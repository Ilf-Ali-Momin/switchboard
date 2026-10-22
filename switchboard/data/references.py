"""Ground truth for the sources that have it: gsm8k gold answers and mbpp test cases.

Output: data/references.parquet with columns
    id, gold_answer (gsm8k), test_setup_code, tests (mbpp, list of assert lines)

The frozen prompts file is not touched. Rows are matched to it by id, and the prompt text is
checked so a changed upstream dataset cannot silently attach the wrong reference.
"""

from __future__ import annotations

import pandas as pd

from switchboard.config import ROOT
from switchboard.data.collect import load_gsm8k, load_mbpp, raw_gsm8k, raw_mbpp

DATA_DIR = ROOT / "data"
REFERENCES_PATH = DATA_DIR / "references.parquet"


def gsm8k_gold(solution: str) -> str:
    """The final answer after '####' in a gsm8k solution."""
    return solution.split("####")[-1].strip().replace(",", "")


def gsm8k_references() -> pd.DataFrame:
    raw = raw_gsm8k()
    return pd.DataFrame(
        {
            "id": "gsm8k-" + raw.index.astype(str),
            "gold_answer": raw["answer"].map(gsm8k_gold),
        }
    )


def mbpp_references() -> pd.DataFrame:
    raw = raw_mbpp()
    return pd.DataFrame(
        {
            "id": "mbpp-" + raw["task_id"].astype(str),
            "test_setup_code": raw["test_setup_code"].fillna(""),
            "tests": raw["test_list"].map(list),
        }
    )


def check_prompts_match(prompts: pd.DataFrame) -> None:
    """Fail if the upstream text behind an id differs from the frozen prompt."""
    fresh = pd.concat([load_gsm8k(), load_mbpp()])[["id", "prompt"]]
    merged = prompts.merge(fresh, on="id", suffixes=("", "_fresh"))
    bad = merged[merged["prompt"] != merged["prompt_fresh"]]
    if len(bad):
        raise ValueError(f"{len(bad)} prompts differ from upstream, e.g. {bad['id'].iloc[0]}")


def build_references(prompts: pd.DataFrame) -> pd.DataFrame:
    check_prompts_match(prompts[prompts["source"].isin(["gsm8k", "mbpp"])])
    refs = pd.concat([gsm8k_references(), mbpp_references()], ignore_index=True)
    return refs[refs["id"].isin(prompts["id"])].reset_index(drop=True)


def load_references(prompts: pd.DataFrame) -> pd.DataFrame:
    """Read the cached references, building them on first use."""
    if REFERENCES_PATH.exists():
        return pd.read_parquet(REFERENCES_PATH)
    refs = build_references(prompts)
    refs.to_parquet(REFERENCES_PATH, index=False)
    return refs
