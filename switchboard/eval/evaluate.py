"""Step 6: evaluate the router against baselines.

The operating threshold is picked on val: the cheapest point that keeps
`eval.target_quality_retained` of the large model's quality. Every reported number is then
measured on test at that threshold. Random routing at the same share (mean of 20 seeds) is the
bar the router must clear; if it does not, results.json says so.

Input:  data/labels.parquet, data/prompts.parquet, data/answers.parquet, models/router/
Output: results/results.json, results/cost_vs_quality.png
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from switchboard.config import ROOT, settings
from switchboard.eval import baselines
from switchboard.eval.metrics import (
    answer_cost,
    pick_threshold,
    routing_metrics,
    safe_roc_auc,
    sweep,
    threshold_grid,
)
from switchboard.eval.plots import plot_cost_vs_quality

DATA_DIR = ROOT / "data"


# --- data ---


def token_costs(answers: pd.DataFrame) -> pd.DataFrame:
    """Per prompt cost of the small and the large answer, from real token usage."""
    ok = answers[answers["error"].isna()].copy()
    ok["cost"] = 0.0
    for role in ("small", "large"):
        rows = ok["role"] == role
        ok.loc[rows, "cost"] = answer_cost(
            ok.loc[rows, "input_tokens"], ok.loc[rows, "output_tokens"], role
        )
    wide = ok.pivot(index="id", columns="role", values="cost")
    return wide.rename(columns={"small": "small_cost", "large": "large_cost"}).dropna()


def eval_frame(labels_path: Path) -> pd.DataFrame:
    """Val and test prompts with both scores, both costs and the prompt length in tokens."""
    labels = pd.read_parquet(labels_path)
    labels = labels[labels["split"].isin(["val", "test"])]
    prompts = pd.read_parquet(DATA_DIR / "prompts.parquet")[["id", "prompt"]]
    costs = token_costs(pd.read_parquet(DATA_DIR / "answers.parquet"))
    df = labels.merge(prompts, on="id").merge(costs, left_on="id", right_index=True)
    tokenizer = AutoTokenizer.from_pretrained(settings().encoder_model)
    df["prompt_tokens"] = [len(ids) for ids in tokenizer(df["prompt"].tolist())["input_ids"]]
    return df.reset_index(drop=True)


def router_scores(df: pd.DataFrame, model_dir: Path, scores_path: Path | None) -> np.ndarray:
    """P(small_ok) per row, from a saved scores file or by running the model."""
    if scores_path is not None:
        scores = pd.read_parquet(scores_path).set_index("id")["router_score"]
        return df["id"].map(scores).to_numpy(dtype=float)
    from switchboard.router_model import RouterModel  # torch import only when needed

    return np.asarray(RouterModel(model_dir).predict(df["prompt"].tolist()))


# --- evaluation ---


def rounded(d: dict, digits: int = 4) -> dict:
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in d.items()}


def beats_random(df: pd.DataFrame, to_small: np.ndarray) -> dict:
    """Router quality against random routing at the same share, seed by seed."""
    router = routing_metrics(df, to_small)["mean_quality"]
    rand = baselines.random_routing(df, float(to_small.mean()))
    per_seed = []
    n_small = int(to_small.sum())
    for seed in range(baselines.RANDOM_SEEDS):
        mask = np.zeros(len(df), dtype=bool)
        mask[np.random.default_rng(seed).choice(len(df), n_small, replace=False)] = True
        per_seed.append(routing_metrics(df, mask)["mean_quality"])
    return {
        "router_mean_quality": round(router, 4),
        "random_mean_quality": round(rand["mean_quality"], 4),
        "random_mean_quality_std": round(rand["mean_quality_std"], 4),
        "quality_gain_over_random": round(router - rand["mean_quality"], 4),
        "seeds_beaten": f"{sum(router > q for q in per_seed)}/{len(per_seed)}",
        # At share 0 or 1 every strategy routes the same way, so there is nothing to compare.
        "beats_random": None if n_small in (0, len(df)) else bool(router > rand["mean_quality"]),
    }


def per_category(df: pd.DataFrame, to_small: np.ndarray) -> dict:
    out = {}
    for cat, idx in df.groupby("category").indices.items():
        m = routing_metrics(df.iloc[idx], to_small[idx])
        out[cat] = {"n": len(idx), **rounded(m, 3)}
    return out


def headline(threshold: float, test: dict) -> str:
    return (
        f"At threshold {threshold:.2f}, Switchboard sends {test['share_small']:.0%} of prompts "
        f"to the small model and keeps {test['quality_retained']:.1%} of the large model's "
        f"quality, cutting cost by {test['cost_reduction']:.0%}."
    )


def evaluate(df: pd.DataFrame, scores: np.ndarray, target: float) -> tuple[dict, dict]:
    """Returns (results, curves). The threshold comes from val only."""
    val, test = df["split"] == "val", df["split"] == "test"
    dval, dtest = df[val].reset_index(drop=True), df[test].reset_index(drop=True)
    sval, stest = scores[val.to_numpy()], scores[test.to_numpy()]
    grid = threshold_grid()

    val_curve = sweep(dval, sval, grid)
    op = pick_threshold(val_curve, target)
    threshold = float(op["threshold"])
    to_small = stest >= threshold
    test_point = routing_metrics(dtest, to_small)
    share = float(to_small.mean())

    test_curve = sweep(dtest, stest, grid)
    curves = {
        "router": test_curve,
        "random": baselines.random_curve(dtest, np.linspace(0, 1, 21)),
        "length": baselines.length_curve(dtest),
        "always_small": baselines.always(dtest, True),
        "always_large": baselines.always(dtest, False),
        "operating_point": test_point,
    }
    vs_random = beats_random(dtest, to_small)
    results = {
        "headline": headline(threshold, test_point),
        "router_beats_random": vs_random["beats_random"],
        "operating_point": {
            "threshold": threshold,
            "picked_on": "val",
            "target_quality_retained": target,
            "val": rounded(op.drop("threshold").to_dict()),
            "test": rounded(test_point),
        },
        "test_baselines_at_same_share": {
            "random": rounded(baselines.random_routing(dtest, share)),
            "length_heuristic": rounded(baselines.length_at_share(dtest, share)),
        },
        "test_baselines": {
            "always_small": rounded(curves["always_small"]),
            "always_large": rounded(curves["always_large"]),
        },
        "router_vs_random_on_test": vs_random,
        "test_router_roc_auc": safe_roc_auc(dtest["small_ok"].astype(int).to_numpy(), stest),
        "test_per_category": per_category(dtest, to_small),
        "n": {"val": int(val.sum()), "test": int(test.sum())},
        "test_curve": test_curve.round(4).to_dict(orient="records"),
    }
    return results, curves


def metadata() -> dict:
    s = settings()
    return {
        "created": datetime.now(UTC).date().isoformat(),
        "small_model": s.small_model,
        "large_model": s.large_model,
        "judge_model": s.judge_model,
        "encoder_model": s.encoder_model,
        "prices_per_million_tokens": s.prices_per_million_tokens.model_dump(),
        "label_margin": s.labeling.margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.parquet")
    parser.add_argument("--model-dir", type=Path, default=ROOT / settings().training.output_dir)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="parquet with id, router_score (skips running the model)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N prompts")
    args = parser.parse_args()

    df = eval_frame(args.labels)
    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=settings().dataset.seed)
        df = df.reset_index(drop=True)
    scores = router_scores(df, args.model_dir, args.scores)
    results, curves = evaluate(df, scores, settings().eval.target_quality_retained)
    results = {"meta": metadata(), **results}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_cost_vs_quality(curves, results, args.output_dir / "cost_vs_quality.png")

    print(results["headline"])
    print(json.dumps(results["router_vs_random_on_test"], indent=2))
    if results["router_beats_random"] is None:
        print("FINDING: the operating point sends every prompt to one model, so the router adds")
        print("nothing over a fixed choice at this quality target.")
    elif not results["router_beats_random"]:
        print("FINDING: the router does not beat random routing at the same share on test.")
    print(f"wrote {args.output_dir / 'results.json'} and {args.output_dir / 'cost_vs_quality.png'}")


if __name__ == "__main__":
    main()
