"""Step 3: score every answer on its own and derive the routing label.

Input:  data/prompts.parquet, data/answers.parquet, data/references.parquet
Output: data/judgments.parquet  one row per (prompt, role):
            id, role, scorer, score, reason, attempts, error
        data/labels.parquet     one row per prompt with both answers judged:
            id, split, source, category, small_score, large_score, small_ok

Scorer per source:
    gsm8k         gsm8k_exact  final number equals the gold answer (5) or not (1)
    mbpp          mbpp_tests   code passes all tests in a sandboxed subprocess (5) or not (1)
    dolly, oasst1 llm_judge    the LLM judge below

The judge sees the prompt and one answer, never which model wrote it, and returns
{"score": int, "reason": str}. An invalid reply is retried once with a correction message;
a second failure marks the row as failed and it is excluded from the labels.

Only prompts whose small and large answers both exist are judged. With --limit N the first N
of those in a seeded shuffle are judged and outputs get a .limitN suffix. --sources restricts
the run to some sources (gsm8k mbpp needs no LLM calls) and adds a .src-... suffix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from switchboard.config import ROOT, settings
from switchboard.data.reference_scoring import RefScore, score_gsm8k, score_mbpp
from switchboard.data.references import load_references
from switchboard.llm import DailyLimitError, LLMClient

DATA_DIR = ROOT / "data"
ROLES = ("small", "large")
MIN_CLASS_SHARE = 0.20
SANDBOX_SLOTS = 4  # mbpp test runs at once
Score = int | pd.Series

SYSTEM_PROMPT = """You are a strict, fair grader. You see a user's request and one \
assistant answer. Score the answer from 1 to 5 on three things together:
- correctness: facts, math, logic and code are right
- completeness: the whole request is addressed
- instruction following: the answer does what was asked, in the form that was asked

Scale:
5 = fully correct and complete, follows the instruction exactly
4 = correct and useful, with minor omissions or style issues
3 = partly correct or incomplete, still somewhat useful
2 = mostly wrong, off topic or missing key parts
1 = wrong, empty, cut off before the answer, or refuses without reason

Judge only the answer you are shown. Length is not quality: do not reward padding.
Reply with JSON only: {"score": <integer 1-5>, "reason": "<one or two sentences>"}"""

RETRY_MESSAGE = (
    "Your reply was not valid. Reply with JSON only, exactly in the form "
    '{"score": <integer 1-5>, "reason": "<one or two sentences>"}.'
)


class InvalidJudgment(ValueError):
    pass


@dataclass
class Judgment:
    score: int | None
    reason: str | None
    attempts: int
    error: str | None = None


def judge_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    shown = answer if answer.strip() else "(empty answer)"
    user = f"## Request\n{prompt}\n\n## Answer\n{shown}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_judgment(text: str) -> tuple[int, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidJudgment(f"not JSON: {text[:100]!r}") from exc
    if not isinstance(data, dict):
        raise InvalidJudgment("not a JSON object")
    score, reason = data.get("score"), data.get("reason")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise InvalidJudgment(f"bad score: {score!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidJudgment("missing reason")
    return score, reason.strip()


def judge_params() -> dict[str, Any]:
    return {
        "temperature": 0,
        "max_tokens": settings().judging.max_tokens,
        "response_format": {"type": "json_object"},
    }


async def ask_judge(client: LLMClient, messages: list[dict[str, str]]) -> tuple[int, str, str]:
    """One judge call. Returns (score, reason, raw text) or raises InvalidJudgment."""
    try:
        c = await client.chat(settings().judge_model, messages, **judge_params())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400:  # Groq rejects replies that fail JSON mode
            raise InvalidJudgment(f"rejected by JSON mode: {exc.response.text[:100]}") from exc
        raise
    score, reason = parse_judgment(c.content)
    return score, reason, c.content


async def judge_answer(client: LLMClient, prompt: str, answer: str) -> Judgment:
    messages = judge_messages(prompt, answer)
    try:
        score, reason, _ = await ask_judge(client, messages)
        return Judgment(score, reason, attempts=1)
    except InvalidJudgment:
        # The correction message changes the cache key, so the retry is a fresh call.
        retry = [*messages, {"role": "user", "content": RETRY_MESSAGE}]
        try:
            score, reason, _ = await ask_judge(client, retry)
            return Judgment(score, reason, attempts=2)
        except InvalidJudgment as second:
            return Judgment(None, None, attempts=2, error=f"InvalidJudgment: {second}")


async def judge_row(client: LLMClient, row_id: str, role: str, prompt: str, answer: str) -> dict:
    try:
        j = await judge_answer(client, prompt, answer)
    except (httpx.HTTPError, DailyLimitError) as exc:
        j = Judgment(None, None, attempts=0, error=f"{type(exc).__name__}: {exc}"[:500])
    return {
        "id": row_id,
        "role": role,
        "scorer": "llm_judge",
        "score": j.score,
        "reason": j.reason,
        "attempts": j.attempts,
        "error": j.error,
    }


def reference_row(row_id: str, role: str, scorer: str, ref: RefScore) -> dict:
    return {
        "id": row_id,
        "role": role,
        "scorer": scorer,
        "score": ref.score,
        "reason": ref.reason,
        "attempts": 0,
        "error": None,
    }


async def score_row(client: LLMClient, sandbox: asyncio.Semaphore, r: Any, role: str) -> dict:
    """Score one answer with the scorer its source calls for."""
    answer = getattr(r, f"{role}_answer")
    if r.source == "gsm8k":
        return reference_row(r.id, role, "gsm8k_exact", score_gsm8k(answer, r.gold_answer))
    if r.source == "mbpp":
        async with sandbox:
            ref = await asyncio.to_thread(score_mbpp, answer, r.test_setup_code, list(r.tests))
        return reference_row(r.id, role, "mbpp_tests", ref)
    return await judge_row(client, r.id, role, r.prompt, answer)


def answered_prompts(prompts: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Prompts whose small and large answers both exist, with the answers as columns."""
    ok = answers[answers["error"].isna()]
    wide = ok.pivot(index="id", columns="role", values="answer").dropna(subset=list(ROLES))
    wide.columns = [f"{c}_answer" for c in wide.columns]
    return prompts.merge(wide, left_on="id", right_index=True, how="inner")


def select(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return df.reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).head(limit).reset_index(drop=True)


async def judge_all(rows: pd.DataFrame) -> pd.DataFrame:
    sandbox = asyncio.Semaphore(SANDBOX_SLOTS)
    async with LLMClient() as client:
        tasks = [score_row(client, sandbox, r, role) for r in rows.itertuples() for role in ROLES]
        out = []
        for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
            out.append(await fut)
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} judgments done")
    return pd.DataFrame(out).sort_values(["id", "role"]).reset_index(drop=True)


def small_ok(small_score: Score, large_score: Score, margin: float) -> bool | pd.Series:
    """The routing label. Works on ints and on pandas Series alike."""
    return small_score >= large_score - margin


def make_labels(prompts: pd.DataFrame, judgments: pd.DataFrame, margin: float) -> pd.DataFrame:
    """One row per prompt where both answers got a valid score."""
    scored = judgments[judgments["score"].notna()]
    wide = scored.pivot(index="id", columns="role", values="score")
    wide = wide.reindex(columns=list(ROLES)).dropna()  # a role may have no scores at all
    wide = wide.astype(int).rename(columns={"small": "small_score", "large": "large_score"})
    labels = prompts[["id", "split", "source", "category"]].merge(
        wide, left_on="id", right_index=True, how="inner"
    )
    labels["small_ok"] = small_ok(labels["small_score"], labels["large_score"], margin)
    return labels.sort_values("id").reset_index(drop=True)


def label_balance(labels: pd.DataFrame) -> pd.DataFrame:
    by_cat = labels.groupby("category")["small_ok"].agg(n="size", small_ok_share="mean")
    total = pd.DataFrame(
        {"n": [len(labels)], "small_ok_share": [labels["small_ok"].mean()]}, index=["ALL"]
    )
    return pd.concat([by_cat, total])


def imbalanced_categories(balance: pd.DataFrame, min_share: float = MIN_CLASS_SHARE) -> list[str]:
    share = balance["small_ok_share"]
    return balance.index[(share < min_share) | (share > 1 - min_share)].tolist()


def paths(limit: int | None, sources: list[str] | None = None) -> tuple[Path, Path]:
    suffix = "" if limit is None else f".limit{limit}"
    if sources:
        suffix += ".src-" + "-".join(sorted(sources))
    return DATA_DIR / f"judgments{suffix}.parquet", DATA_DIR / f"labels{suffix}.parquet"


def summarize(judgments: pd.DataFrame, labels: pd.DataFrame) -> None:
    limited = judgments["error"].fillna("").str.startswith("DailyLimitError")
    invalid = judgments["error"].fillna("").str.startswith("InvalidJudgment")
    print(
        f"\n{len(judgments)} judgments: {judgments['score'].notna().sum()} scored, "
        f"{invalid.sum()} invalid after retry, {limited.sum()} pending on daily quota, "
        f"{(judgments['error'].notna() & ~invalid & ~limited).sum()} other errors"
    )
    print(f"retried once: {(judgments['attempts'] == 2).sum()}")
    for err in judgments.loc[judgments["error"].notna() & ~limited, "error"].unique()[:5]:
        print("error:", err)
    print("\nscore distribution by scorer:")
    print(pd.crosstab([judgments["scorer"], judgments["role"]], judgments["score"]), "\n")
    print("mean score:", judgments.groupby("role")["score"].mean().round(2).to_dict())
    balance = label_balance(labels)
    print(f"\nlabel balance ({len(labels)} labelled prompts):")
    print(balance.round(2))
    flagged = imbalanced_categories(balance)
    if flagged:
        print(f"\nWARNING: a class is under {MIN_CLASS_SHARE:.0%} in: {', '.join(flagged)}")


def show_examples(rows: pd.DataFrame, judgments: pd.DataFrame, n: int = 3) -> None:
    j = judgments.set_index(["id", "role"])
    for r in rows.head(n).itertuples():
        print(f"\n--- {r.id} [{r.category}] {r.prompt[:120]!r}")
        for role in ROLES:
            score, reason = j.loc[(r.id, role), ["score", "reason"]]
            print(f"  {role}: score={score} | {str(reason)[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="judge only N prompts")
    parser.add_argument(
        "--sources", nargs="+", default=None, help="only these sources, e.g. gsm8k mbpp"
    )
    args = parser.parse_args()

    prompts = pd.read_parquet(DATA_DIR / "prompts.parquet")
    answers = pd.read_parquet(DATA_DIR / "answers.parquet")
    refs = load_references(prompts)
    pool = answered_prompts(prompts, answers).merge(refs, on="id", how="left")
    if args.sources:
        pool = pool[pool["source"].isin(args.sources)]
    print(f"{len(pool)}/{len(prompts)} prompts have both answers")
    rows = select(pool, args.limit, settings().dataset.seed)

    judgments = asyncio.run(judge_all(rows))
    labels = make_labels(prompts, judgments, settings().labeling.margin)
    judgments_path, labels_path = paths(args.limit, args.sources)
    judgments.to_parquet(judgments_path, index=False)
    labels.to_parquet(labels_path, index=False)
    summarize(judgments, labels)
    show_examples(rows, judgments)
    print(f"\nwrote {judgments_path.relative_to(ROOT)} and {labels_path.relative_to(ROOT)}")
    if judgments["error"].notna().any():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
