"""Baselines the router must beat: always small, always large, random and prompt length."""

from __future__ import annotations

import numpy as np
import pandas as pd

from switchboard.eval.metrics import routing_metrics

RANDOM_SEEDS = 20


def always(df: pd.DataFrame, to_small: bool) -> dict[str, float]:
    return routing_metrics(df, np.full(len(df), to_small))


def random_routing(df: pd.DataFrame, share: float, seeds: int = RANDOM_SEEDS) -> dict[str, float]:
    """Mean metrics over seeds of sending round(share * n) random prompts to small."""
    n_small = round(share * len(df))
    runs = []
    for seed in range(seeds):
        to_small = np.zeros(len(df), dtype=bool)
        to_small[np.random.default_rng(seed).choice(len(df), n_small, replace=False)] = True
        runs.append(routing_metrics(df, to_small))
    mean = pd.DataFrame(runs).mean().to_dict()
    mean["mean_quality_std"] = float(pd.DataFrame(runs)["mean_quality"].std(ddof=0))
    return mean


def random_curve(df: pd.DataFrame, shares: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([random_routing(df, s) for s in shares])


def length_curve(df: pd.DataFrame, cutoffs: np.ndarray | None = None) -> pd.DataFrame:
    """Send prompts with at most `cutoff` tokens to small, for a sweep of cutoffs."""
    tokens = df["prompt_tokens"].to_numpy()
    if cutoffs is None:
        cutoffs = np.unique(np.concatenate([[-1], np.quantile(tokens, np.linspace(0, 1, 51))]))
    rows = [{"cutoff": float(c), **routing_metrics(df, tokens <= c)} for c in cutoffs]
    return pd.DataFrame(rows)


def length_at_share(df: pd.DataFrame, share: float) -> dict[str, float]:
    """The length heuristic with the cutoff that sends about `share` of prompts to small."""
    order = np.argsort(df["prompt_tokens"].to_numpy(), kind="stable")
    to_small = np.zeros(len(df), dtype=bool)
    to_small[order[: round(share * len(df))]] = True
    return routing_metrics(df, to_small)
