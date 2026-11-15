"""Render README.md from docs/README.template.md and the files in results/.

Every number in the README comes from results/*.json through this module, so none is typed
by hand. Placeholders look like {{name}}; an unknown name is an error.

    uv run python -m switchboard.eval.readme
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from switchboard.config import ROOT

TEMPLATE_PATH = ROOT / "docs" / "README.template.md"
README_PATH = ROOT / "README.md"
RESULTS_DIR = ROOT / "results"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

BASELINE_ROWS = (
    ("Switchboard router", "router"),
    ("Random routing, same share (mean of 20 seeds)", "random"),
    ("Prompt length heuristic, same share", "length_heuristic"),
    ("Always small", "always_small"),
    ("Always large", "always_large"),
)


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def load_results(results_dir: Path) -> dict[str, dict]:
    names = ("results", "judge_calibration", "train_metrics", "classifier_latency")
    return {n: json.loads((results_dir / f"{n}.json").read_text()) for n in names}


def baseline_table(results: dict) -> str:
    rows = {
        "router": results["operating_point"]["test"],
        **results["test_baselines_at_same_share"],
        **results["test_baselines"],
    }
    lines = [
        "| Strategy | Sent to small | Mean judge score | Quality kept | Cost cut |",
        "|---|---|---|---|---|",
    ]
    for label, key in BASELINE_ROWS:
        r = rows[key]
        name = f"**{label}**" if key == "router" else label
        lines.append(
            f"| {name} | {pct(r['share_small'])} | {r['mean_quality']:.2f} | "
            f"{pct(r['quality_retained'], 1)} | {pct(r['cost_reduction'])} |"
        )
    return "\n".join(lines)


def category_table(results: dict) -> str:
    lines = [
        "| Category | Test prompts | Sent to small | Quality kept | Cost cut |",
        "|---|---|---|---|---|",
    ]
    for cat, r in sorted(results["test_per_category"].items()):
        lines.append(
            f"| {cat} | {r['n']} | {pct(r['share_small'])} | "
            f"{pct(r['quality_retained'], 1)} | {pct(r['cost_reduction'])} |"
        )
    return "\n".join(lines)


def values(data: dict[str, dict]) -> dict[str, str]:
    res, cal = data["results"], data["judge_calibration"]
    train, lat = data["train_metrics"], data["classifier_latency"]
    meta, op = res["meta"], res["operating_point"]
    vs_random = res["router_vs_random_on_test"]
    prices = meta["prices_per_million_tokens"]
    return {
        "headline": res["headline"],
        "beats_random_sentence": (
            f"The router beats random routing at the same share on "
            f"{vs_random['seeds_beaten']} seeds: mean judge score "
            f"{vs_random['router_mean_quality']:.2f} against "
            f"{vs_random['random_mean_quality']:.2f} ± {vs_random['random_mean_quality_std']:.2f}."
            if res["router_beats_random"]
            else "The router does not beat random routing at the same share."
        ),
        "baseline_table": baseline_table(res),
        "category_table": category_table(res),
        "threshold": f"{op['threshold']:.2f}",
        "target": pct(op["target_quality_retained"]),
        "n_val": str(res["n"]["val"]),
        "n_test": str(res["n"]["test"]),
        "n_train": str(train["train"]["n"]),
        "train_small_ok": pct(train["train"]["small_ok_share"]),
        "val_auc": f"{train['val_roc_auc']:.2f}",
        "test_auc": f"{res['test_router_roc_auc']:.2f}",
        "best_lr": f"{train['best_learning_rate']:g}",
        "small_model": meta["small_model"],
        "large_model": meta["large_model"],
        "judge_model": meta["judge_model"],
        "encoder_model": meta["encoder_model"],
        "prices_checked_on": prices["checked_on"],
        "small_price": f"${prices['small']['input']:g} / ${prices['small']['output']:g}",
        "large_price": f"${prices['large']['input']:g} / ${prices['large']['output']:g}",
        "cal_n": str(cal["hand_labelled"]),
        "cal_agreement": pct(cal["agreement"]),
        "cal_kappa": f"{cal['cohen_kappa']:.2f}",
        "cal_human_ok": pct(cal["human_small_ok_share"]),
        "cal_judge_ok": pct(cal["judge_small_ok_share"]),
        "lat_p50": f"{lat['p50_ms']:.0f}",
        "lat_p95": f"{lat['p95_ms']:.0f}",
        "lat_n": str(lat["n"]),
        "lat_threads": str(lat["torch_threads"]),
    }


def render(template: str, vals: dict[str, str]) -> str:
    missing = sorted(set(PLACEHOLDER.findall(template)) - vals.keys())
    if missing:
        raise KeyError(f"no value for placeholders: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: vals[m.group(1)], template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=README_PATH)
    args = parser.parse_args()
    text = render(args.template.read_text(), values(load_results(args.results_dir)))
    args.output.write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
