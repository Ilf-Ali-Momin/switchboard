"""Publish the dataset, the model and the demo Space to the Hugging Face Hub.

By default this only builds everything under data/hub/ so it can be reviewed. Nothing leaves
the machine without --push, and everything pushed is public.

    uv run python -m switchboard.data.publish              # build data/hub/ for review
    uv run python -m switchboard.data.publish --push dataset model space

Cards are rendered from docs/*_card.template.md with numbers from results/*.json, like the
README. Repo names: <HF_USERNAME>/switchboard-routing (dataset), <HF_USERNAME>/switchboard-router
(model), <HF_USERNAME>/switchboard (Space).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from switchboard.config import ROOT, settings
from switchboard.eval import readme

DATA_DIR = ROOT / "data"
HUB_DIR = DATA_DIR / "hub"
DOCS_DIR = ROOT / "docs"
CHART = ROOT / "results" / "cost_vs_quality.png"
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
TARGETS = ("dataset", "model", "space")


# --- dataset ---


def wide(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per prompt, with small_<col> and large_<col> columns."""
    out = df.pivot(index="id", columns="role", values=columns)
    out.columns = [f"{role}_{col}" for col, role in out.columns]
    return out.reset_index()


def build_rows() -> pd.DataFrame:
    prompts = pd.read_parquet(DATA_DIR / "prompts.parquet")
    answers = wide(pd.read_parquet(DATA_DIR / "answers.parquet"), ["answer", "output_tokens"])
    judged = wide(pd.read_parquet(DATA_DIR / "judgments.parquet"), ["score", "reason"])
    scorer = pd.read_parquet(DATA_DIR / "judgments.parquet").groupby("id")["scorer"].first()
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")[["id", "small_ok"]]
    df = (
        prompts[["id", "prompt", "source", "category", "license", "split"]]
        .merge(answers, on="id")
        .merge(judged, on="id", how="left")
        .merge(scorer.rename("scorer").reset_index(), on="id", how="left")
        .merge(labels, on="id", how="left")
    )
    df["small_ok"] = df["small_ok"].astype("boolean")  # null where a judgment failed
    return df.sort_values("id").reset_index(drop=True)


def write_dataset(rows: pd.DataFrame, out: Path) -> None:
    (out / "data").mkdir(parents=True, exist_ok=True)
    for split, name in SPLIT_NAMES.items():
        part = rows[rows["split"] == split].drop(columns="split")
        part.to_parquet(out / "data" / f"{name}.parquet", index=False)


def dataset_values(rows: pd.DataFrame) -> dict[str, str]:
    labelled = rows.dropna(subset=["small_ok"])
    by_source = rows.groupby(["source", "license"]).size().reset_index(name="n")
    balance = labelled.groupby("category")["small_ok"].agg(["size", "mean"])
    return {
        "n_prompts": str(len(rows)),
        "n_labelled": str(len(labelled)),
        "n_failed": str(len(rows) - len(labelled)),
        "label_share": readme.pct(labelled["small_ok"].mean()),
        "split_counts": ", ".join(
            f"{name} {int((rows['split'] == s).sum())}" for s, name in SPLIT_NAMES.items()
        ),
        "source_table": "\n".join(
            ["| Source | License | Prompts |", "|---|---|---|"]
            + [f"| `{r.source}` | {r.license} | {r.n} |" for r in by_source.itertuples()]
        ),
        "balance_table": "\n".join(
            ["| Category | Labelled prompts | small_ok |", "|---|---|---|"]
            + [
                f"| {c} | {int(r['size'])} | {readme.pct(r['mean'])} |"
                for c, r in balance.iterrows()
            ]
        ),
    }


# --- cards and folders ---


def all_values(user: str, rows: pd.DataFrame) -> dict[str, str]:
    vals = readme.values(readme.load_results(readme.RESULTS_DIR))
    vals |= dataset_values(rows)
    vals |= {
        "hf_user": user,
        "dataset_repo": f"{user}/switchboard-routing",
        "model_repo": f"{user}/switchboard-router",
        "space_repo": f"{user}/switchboard",
        "github_url": "https://github.com/Ilf-Ali-Momin/switchboard",
    }
    return vals


def render_card(name: str, vals: dict[str, str], out: Path) -> None:
    text = readme.render((DOCS_DIR / f"{name}_card.template.md").read_text(), vals)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(text)


def build_model_dir(out: Path) -> None:
    src = ROOT / settings().training.output_dir
    for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(src / f, out / f)
    shutil.copy2(CHART, out / CHART.name)


def build_space_dir(out: Path, vals: dict[str, str]) -> None:
    space_src = ROOT / "space"
    for f in ("app.py", "requirements.txt"):
        shutil.copy2(space_src / f, out / f)
    shutil.copy2(CHART, out / CHART.name)
    threshold = vals["threshold"]
    (out / "settings.env").write_text(
        f"MODEL_REPO={vals['model_repo']}\nDATASET_REPO={vals['dataset_repo']}\n"
        f"THRESHOLD={threshold}\nMAX_PROMPT_TOKENS={settings().dataset.max_prompt_tokens}\n"
    )


def build_all(user: str) -> dict[str, Path]:
    if HUB_DIR.exists():
        shutil.rmtree(HUB_DIR)
    rows = build_rows()
    vals = all_values(user, rows)
    dirs = {t: HUB_DIR / t for t in TARGETS}
    write_dataset(rows, dirs["dataset"])
    render_card("dataset", vals, dirs["dataset"])
    render_card("model", vals, dirs["model"])
    build_model_dir(dirs["model"])
    render_card("space", vals, dirs["space"])
    build_space_dir(dirs["space"], vals)
    return dirs


# --- push ---


def push(target: str, folder: Path, user: str, token: str) -> str:
    from huggingface_hub import HfApi

    repo_type = {"dataset": "dataset", "model": "model", "space": "space"}[target]
    name = {"dataset": "switchboard-routing", "model": "switchboard-router"}.get(
        target, "switchboard"
    )
    repo_id = f"{user}/{name}"
    api = HfApi(token=token)
    extra = {"space_sdk": "gradio"} if target == "space" else {}
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, **extra)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder))
    return f"https://huggingface.co/{'' if target == 'model' else repo_type + 's/'}{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--push", nargs="+", choices=TARGETS, default=[], help="public upload")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    user = os.environ.get("HF_USERNAME") or "HF_USERNAME"
    dirs = build_all(user)
    for target, folder in dirs.items():
        print(f"built {target}: {folder}")
    if not args.push:
        print("nothing pushed: review data/hub/, then rerun with --push")
        return
    token = os.environ.get("HF_TOKEN")
    if not token or user == "HF_USERNAME":
        raise SystemExit("set HF_TOKEN and HF_USERNAME in .env before --push")
    for target in args.push:
        print(f"pushed {target}: {push(target, dirs[target], user, token)}")


if __name__ == "__main__":
    main()
