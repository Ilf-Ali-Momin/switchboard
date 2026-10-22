"""Step 1: collect prompts from public Hub datasets and freeze train/val/test splits.

Output: data/prompts.parquet with columns
    id, prompt, source, category, subcategory, license, split

The full file is frozen once written. A rerun refuses to overwrite it unless --force is given.
A run with --limit N writes data/prompts.limitN.parquet instead, so smoke tests never touch
the frozen splits.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from switchboard.config import ROOT, settings

DATA_DIR = ROOT / "data"
COLUMNS = ["id", "prompt", "source", "category", "subcategory", "license"]

# Licenses checked on the Hub dataset pages on 2026-10-16.
LICENSES = {
    "dolly": "cc-by-sa-3.0",
    "oasst1": "apache-2.0",
    "gsm8k": "mit",
    "mbpp": "cc-by-4.0",
}

DOLLY_CATEGORY = {
    "open_qa": "factual_qa",
    "closed_qa": "factual_qa",
    "general_qa": "factual_qa",
    "summarization": "summarization",
    "information_extraction": "summarization",
    "brainstorming": "general_instruction",
    "creative_writing": "general_instruction",
    "classification": "general_instruction",
}


def load_dolly() -> pd.DataFrame:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
    has_context = ds["context"].str.strip().str.len() > 0
    prompt = ds["instruction"].str.strip()
    prompt = prompt.where(~has_context, prompt + "\n\n" + ds["context"].str.strip())
    return pd.DataFrame(
        {
            "id": "dolly-" + ds.index.astype(str),
            "prompt": prompt,
            "source": "dolly",
            "category": ds["category"].map(DOLLY_CATEGORY),
            "subcategory": ds["category"],
        }
    )


def load_oasst1() -> pd.DataFrame:
    ds = load_dataset("OpenAssistant/oasst1", split="train").to_pandas()
    roots = ds[
        ds["parent_id"].isna()
        & (ds["role"] == "prompter")
        & (ds["lang"] == "en")
        & ~ds["deleted"]
        & (ds["review_result"] == True)  # noqa: E712  (column holds None for unreviewed)
    ]
    return pd.DataFrame(
        {
            "id": "oasst1-" + roots["message_id"],
            "prompt": roots["text"].str.strip(),
            "source": "oasst1",
            "category": "general_instruction",
            "subcategory": "chat",
        }
    )


def raw_gsm8k() -> pd.DataFrame:
    """All gsm8k rows in a fixed order. Prompt ids are positions in this frame."""
    parts = [load_dataset("openai/gsm8k", "main", split=s) for s in ("train", "test")]
    return concatenate_datasets(parts).to_pandas()


def load_gsm8k() -> pd.DataFrame:
    ds = raw_gsm8k()
    return pd.DataFrame(
        {
            "id": "gsm8k-" + ds.index.astype(str),
            "prompt": ds["question"].str.strip(),
            "source": "gsm8k",
            "category": "math",
            "subcategory": "word_problem",
        }
    )


def mbpp_prompt(text: str, tests: list[str]) -> str:
    # The first test pins the function name and signature, as in the MBPP paper's prompt.
    return f"{text.strip()}\nYour code should pass this test:\n{tests[0]}"


def raw_mbpp() -> pd.DataFrame:
    splits = ("train", "validation", "test", "prompt")
    parts = [load_dataset("google-research-datasets/mbpp", "full", split=s) for s in splits]
    return concatenate_datasets(parts).to_pandas()


def load_mbpp() -> pd.DataFrame:
    ds = raw_mbpp()
    return pd.DataFrame(
        {
            "id": "mbpp-" + ds["task_id"].astype(str),
            "prompt": [
                mbpp_prompt(t, list(tl)) for t, tl in zip(ds["text"], ds["test_list"], strict=True)
            ],
            "source": "mbpp",
            "category": "coding",
            "subcategory": "python_function",
        }
    )


LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "dolly": load_dolly,
    "oasst1": load_oasst1,
    "gsm8k": load_gsm8k,
    "mbpp": load_mbpp,
}


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact and near duplicates: rows whose normalized text hashes collide."""
    hashes = df["prompt"].map(text_hash)
    return df[~hashes.duplicated(keep="first")].reset_index(drop=True)


def filter_length(df: pd.DataFrame, tokenizer, max_tokens: int) -> pd.DataFrame:
    lengths = [len(ids) for ids in tokenizer(df["prompt"].tolist())["input_ids"]]
    keep = (np.array(lengths) <= max_tokens) & (df["prompt"].str.len() > 0).to_numpy()
    return df[keep].reset_index(drop=True)


def source_quotas(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Integer quota per source that sums to total (largest remainder rounding)."""
    weight = sum(mix.values())
    raw = {s: total * w / weight for s, w in mix.items()}
    quotas = {s: int(v) for s, v in raw.items()}
    leftover = total - sum(quotas.values())
    for s in sorted(raw, key=lambda s: raw[s] - quotas[s], reverse=True)[:leftover]:
        quotas[s] += 1
    return quotas


def sample_sources(df: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    parts = []
    for source, n in quotas.items():
        pool = df[df["source"] == source]
        if len(pool) < n:
            raise ValueError(f"{source}: need {n} prompts, only {len(pool)} pass the filters")
        parts.append(pool.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def assign_splits(
    df: pd.DataFrame, ratios: dict[str, float], seed: int, by: str = "source"
) -> pd.Series:
    """Stratified split: each group is shuffled with the seed and cut by the ratios."""
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=df.index, dtype=object)
    for _, group in df.groupby(by, sort=True):
        idx = rng.permutation(group.index.to_numpy())
        n_train = round(len(idx) * ratios["train"])
        n_val = round(len(idx) * ratios["val"])
        split[idx[:n_train]] = "train"
        split[idx[n_train : n_train + n_val]] = "val"
        split[idx[n_train + n_val :]] = "test"
    return split


def output_path(limit: int | None) -> Path:
    name = "prompts.parquet" if limit is None else f"prompts.limit{limit}.parquet"
    return DATA_DIR / name


def collect(limit: int | None) -> pd.DataFrame:
    cfg = settings().dataset
    tokenizer = AutoTokenizer.from_pretrained(settings().encoder_model)
    frames = []
    for name in cfg.source_mix:
        raw = LOADERS[name]()
        raw["license"] = LICENSES[name]
        frames.append(raw[COLUMNS])
        print(f"loaded {name}: {len(raw)} candidate prompts")
    pool = dedupe(pd.concat(frames, ignore_index=True))
    pool = filter_length(pool, tokenizer, cfg.max_prompt_tokens)
    print(f"after dedupe and length filter: {len(pool)}")
    total = limit if limit is not None else cfg.total_prompts
    df = sample_sources(pool, source_quotas(total, cfg.source_mix), cfg.seed)
    df["split"] = assign_splits(df, cfg.split.model_dump(), cfg.seed)
    return df.sort_values(["split", "source", "id"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(f"\n{len(df)} prompts")
    print(pd.crosstab(df["category"], df["split"], margins=True), "\n")
    print(pd.crosstab(df["source"], df["split"], margins=True), "\n")
    for _, row in df.groupby("source").head(1).iterrows():
        preview = row["prompt"][:160].replace("\n", " ")
        print(f"[{row['source']}/{row['category']}] {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="collect only N prompts")
    parser.add_argument("--force", action="store_true", help="overwrite the frozen splits")
    args = parser.parse_args()

    path = output_path(args.limit)
    if args.limit is None and path.exists() and not args.force:
        raise SystemExit(f"{path} exists and splits are frozen. Pass --force to overwrite.")
    df = collect(args.limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    summarize(df)
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
