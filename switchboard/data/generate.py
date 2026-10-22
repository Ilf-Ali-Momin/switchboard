"""Step 2: answer every prompt with the small and the large model.

Input:  data/prompts.parquet (frozen splits from step 1)
Output: data/answers.parquet, one row per (prompt, model role) with columns
    id, role, model, answer, input_tokens, output_tokens, latency_s, finish_reason, error

An answer cut off at max_tokens (finish_reason "length") is kept as a real result: it is what
the model delivers under the configured budget, and the judge scores it as it stands.

All calls go through the cached LLM client, so a rerun is free and a crashed run resumes.
With --limit N the first N prompts of a seeded shuffle are answered and the output goes to
data/answers.limitN.parquet.

When a model's daily quota runs out, its remaining prompts are recorded with a DailyLimitError
and the run finishes normally. Rerun the same command the next day: answered prompts come from
the cache and only the missing ones are sent.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from switchboard.config import ROOT, settings
from switchboard.llm import DailyLimitError, LLMClient

DATA_DIR = ROOT / "data"
PROMPTS_PATH = DATA_DIR / "prompts.parquet"
ROLES = ("small", "large")


def model_for(role: str) -> str:
    return settings().small_model if role == "small" else settings().large_model


def generation_params() -> dict[str, Any]:
    cfg = settings().generation
    return {
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
        "reasoning_effort": cfg.reasoning_effort,
    }


def select_prompts(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return df
    return df.sample(frac=1.0, random_state=seed).head(limit).reset_index(drop=True)


async def answer_one(client: LLMClient, row_id: str, prompt: str, role: str) -> dict[str, Any]:
    model = model_for(role)
    base = {"id": row_id, "role": role, "model": model}
    messages = [{"role": "user", "content": prompt}]
    try:
        c = await client.chat(model, messages, **generation_params())
    except (httpx.HTTPError, DailyLimitError, KeyError, ValueError) as exc:
        return {
            **base,
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_s": None,
            "finish_reason": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    return {
        **base,
        "answer": c.content,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "latency_s": c.latency_s,
        "finish_reason": c.finish_reason,
        "error": None,
    }


async def generate(prompts: pd.DataFrame) -> pd.DataFrame:
    async with LLMClient() as client:
        tasks = [
            answer_one(client, row.id, row.prompt, role)
            for row in prompts.itertuples()
            for role in ROLES
        ]
        rows = []
        for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
            rows.append(await fut)
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} calls done")
    return pd.DataFrame(rows).sort_values(["id", "role"]).reset_index(drop=True)


def output_path(limit: int | None) -> Path:
    name = "answers.parquet" if limit is None else f"answers.limit{limit}.parquet"
    return DATA_DIR / name


def summarize(answers: pd.DataFrame) -> None:
    ok = answers[answers["error"].isna()]
    limited = answers["error"].fillna("").str.startswith("DailyLimitError")
    print(f"\n{len(answers)} answers, {answers['error'].notna().sum()} failed")
    print(f"pending on daily quota: {limited.groupby(answers['role']).sum().to_dict()}")
    stats = ok.groupby("role").agg(
        n=("id", "size"),
        mean_in=("input_tokens", "mean"),
        mean_out=("output_tokens", "mean"),
        p50_latency_s=("latency_s", "median"),
        empty_answers=("answer", lambda s: (s.str.strip() == "").sum()),
        cut_off=("finish_reason", lambda s: (s == "length").sum()),
    )
    print(stats.round(2), "\n")
    for err in answers.loc[~limited, "error"].dropna().unique()[:5]:
        print("error:", err)
    for _, row in ok.head(4).iterrows():
        preview = row["answer"][:200].replace("\n", " ")
        print(f"[{row['id']} / {row['role']}] {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="answer only N prompts")
    args = parser.parse_args()

    prompts = select_prompts(pd.read_parquet(PROMPTS_PATH), args.limit, settings().dataset.seed)
    answers = asyncio.run(generate(prompts))
    path = output_path(args.limit)
    answers.to_parquet(path, index=False)
    summarize(answers)
    print(f"\nwrote {path.relative_to(ROOT)}")
    if answers["error"].notna().any():
        raise SystemExit(2)  # incomplete: rerun to fill the gaps


if __name__ == "__main__":
    main()
