"""Step 4: check the LLM judge against hand labels.

Samples rows from the train split, stratified by category, among prompts the LLM judge scored
(gsm8k and mbpp are scored against ground truth and need no calibration). For each row it shows
the prompt and both answers as "Answer A" and "Answer B" in a random order with no model names,
and asks which is better. A choice of the small model's answer or a tie means the small answer
was good enough, which is the label rule with margin 0.

Each answer is saved at once to data/calibration_labels.jsonl, so a session can stop with q and
resume later. When enough rows are labelled, agreement and Cohen's kappa against the judge's
labels are written to results/judge_calibration.json.

    uv run python -m switchboard.data.calibrate            # label, then report
    uv run python -m switchboard.data.calibrate --report   # report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from switchboard.config import ROOT, settings
from switchboard.data.collect import source_quotas

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
HUMAN_LABELS_PATH = DATA_DIR / "calibration_labels.jsonl"
REPORT_PATH = RESULTS_DIR / "judge_calibration.json"
JUDGED_SOURCES = ("dolly", "oasst1")
SAMPLE_SIZE = 100
CHOICES = {"a": "A", "b": "B", "t": "tie"}


# --- sampling ---


def sample_rows(labels: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n train rows scored by the LLM judge, allocated to categories by their share."""
    pool = labels[(labels["split"] == "train") & labels["source"].isin(JUDGED_SOURCES)]
    shares = pool["category"].value_counts(normalize=True).to_dict()
    quotas = source_quotas(min(n, len(pool)), shares)
    parts = [
        pool[pool["category"] == cat].sample(n=k, random_state=seed)
        for cat, k in sorted(quotas.items())
        if k
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def small_position(row_id: str, seed: int) -> str:
    """Where the small answer is shown, A or B. Stable per row so a resumed session matches."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return "A" if random.Random(digest).random() < 0.5 else "B"


def human_small_ok(choice: str, small_pos: str) -> bool:
    """The small answer is good enough when the person prefers it or calls a tie."""
    return choice in ("tie", small_pos)


# --- labelling loop ---


def load_human_labels(path: Path | None = None) -> dict[str, dict]:
    path = path or HUMAN_LABELS_PATH
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def append_human_label(record: dict, path: Path | None = None) -> None:
    path = path or HUMAN_LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def render(row: pd.Series, small_pos: str, index: int, total: int) -> str:
    shown = {small_pos: row["small_answer"], "AB".replace(small_pos, ""): row["large_answer"]}
    rule = "=" * 80
    parts = [
        rule,
        f"[{index}/{total}]  category: {row['category']}",
        rule,
        "PROMPT:",
        textwrap.indent(row["prompt"], "  "),
    ]
    for letter in "AB":
        text = shown[letter] if shown[letter].strip() else "(empty answer)"
        parts += ["-" * 80, f"ANSWER {letter}:", textwrap.indent(text, "  ")]
    return "\n".join(parts + [rule])


def ask_choice(read: Callable[[str], str]) -> str | None:
    """Returns 'A', 'B', 'tie', 'skip', or None to quit."""
    prompt = "Which answer is better? [a] A  [b] B  [t] tie  [s] skip  [q] quit: "
    while True:
        key = read(prompt).strip().lower()
        if key in CHOICES:
            return CHOICES[key]
        if key == "s":
            return "skip"
        if key == "q":
            return None


def label_session(
    rows: pd.DataFrame,
    seed: int,
    read: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    path: Path | None = None,
) -> int:
    """Ask about every row not yet labelled. Returns how many were labelled this session."""
    done = load_human_labels(path)
    todo = rows[~rows["id"].isin(done)]
    show(f"{len(done)} already labelled, {len(todo)} to go. q saves and quits.")
    labelled = 0
    for i, row in enumerate(todo.itertuples(index=False), start=len(done) + 1):
        row = pd.Series(row._asdict())
        pos = small_position(row["id"], seed)
        show(render(row, pos, i, len(rows)))
        choice = ask_choice(read)
        if choice is None:
            break
        if choice == "skip":
            continue
        append_human_label(
            {
                "id": row["id"],
                "choice": choice,
                "small_position": pos,
                "human_small_ok": human_small_ok(choice, pos),
                "labelled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            path,
        )
        labelled += 1
    return labelled


# --- report ---


def agreement_report(merged: pd.DataFrame) -> dict:
    """Agreement and Cohen's kappa between human_small_ok and the judge's small_ok."""
    human, judge = merged["human_small_ok"].astype(bool), merged["small_ok"].astype(bool)
    per_category = {
        cat: {
            "n": int(len(g)),
            "agreement": round(float((g["human_small_ok"] == g["small_ok"]).mean()), 3),
        }
        for cat, g in merged.groupby("category")
    }
    kappa = cohen_kappa_score(human, judge) if human.nunique() + judge.nunique() > 2 else None
    return {
        "n": int(len(merged)),
        "agreement": round(float((human == judge).mean()), 3),
        "cohen_kappa": None if kappa is None else round(float(kappa), 3),
        "human_small_ok_share": round(float(human.mean()), 3),
        "judge_small_ok_share": round(float(judge.mean()), 3),
        "confusion": {
            "both_ok": int((human & judge).sum()),
            "both_not_ok": int((~human & ~judge).sum()),
            "judge_ok_human_not": int((judge & ~human).sum()),
            "human_ok_judge_not": int((human & ~judge).sum()),
        },
        "per_category": per_category,
    }


def write_report(labels: pd.DataFrame, path: Path | None = None) -> dict:
    # Paths resolve at call time so tests and dry runs can point them elsewhere.
    path = path or REPORT_PATH
    human = pd.DataFrame(load_human_labels().values())
    if human.empty:
        raise SystemExit("no hand labels yet: run without --report first")
    merged = human.merge(labels[["id", "category", "small_ok"]], on="id", how="inner")
    report = {
        "judge_model": settings().judge_model,
        "sources": list(JUDGED_SOURCES),
        "created": datetime.now(UTC).date().isoformat(),
        **agreement_report(merged),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


# --- entry point ---


def load_rows(labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(labels_path)
    prompts = pd.read_parquet(DATA_DIR / "prompts.parquet")[["id", "prompt"]]
    answers = pd.read_parquet(DATA_DIR / "answers.parquet")
    wide = answers.pivot(index="id", columns="role", values="answer")
    wide.columns = [f"{c}_answer" for c in wide.columns]
    rows = labels.merge(prompts, on="id").merge(wide, left_on="id", right_index=True)
    return labels, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="only write the report")
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_DIR / "labels.parquet",
        help="judge labels to calibrate against",
    )
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE, help="rows to hand label")
    args = parser.parse_args()

    seed = settings().dataset.seed
    labels, rows = load_rows(args.labels)
    if not args.report:
        label_session(sample_rows(rows, args.n, seed), seed)
    report = write_report(labels)
    print(json.dumps({k: report[k] for k in ("n", "agreement", "cohen_kappa")}, indent=2))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
