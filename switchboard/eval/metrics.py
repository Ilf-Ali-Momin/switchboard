"""Quality and cost of a routing decision.

Every routing strategy is a boolean per prompt: True sends it to the small model. The chosen
answer's judge score is its quality; its real token usage times the price table is its cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from switchboard.config import settings

REQUIRED = ("small_score", "large_score", "small_cost", "large_cost")


def answer_cost(input_tokens: pd.Series, output_tokens: pd.Series, role: str) -> pd.Series:
    """USD for one answer from its real token usage."""
    price = getattr(settings().prices_per_million_tokens, role)
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def routing_metrics(df: pd.DataFrame, to_small: np.ndarray) -> dict[str, float]:
    """Share sent to small, mean quality, quality retained and cost versus always large."""
    to_small = np.asarray(to_small, dtype=bool)
    quality = np.where(to_small, df["small_score"], df["large_score"]).mean()
    cost = np.where(to_small, df["small_cost"], df["large_cost"]).sum()
    large_quality = df["large_score"].mean()
    large_cost = df["large_cost"].sum()
    return {
        "share_small": float(to_small.mean()),
        "mean_quality": float(quality),
        "quality_retained": float(quality / large_quality),
        "cost_usd": float(cost),
        "cost_relative": float(cost / large_cost),
        "cost_reduction": float(1 - cost / large_cost),
    }


def threshold_grid(step: float = 0.01) -> np.ndarray:
    """0 to 1, plus one value above 1 so the sweep always includes 'everything to large'."""
    return np.append(np.round(np.arange(0, 1 + step / 2, step), 6), 1.000001)


def sweep(df: pd.DataFrame, scores: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Router metrics at each threshold: a prompt goes to small when its score >= threshold."""
    rows = [{"threshold": float(t), **routing_metrics(df, scores >= t)} for t in thresholds]
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target: float) -> pd.Series:
    """The cheapest point that keeps the target quality. Ties go to more small routing."""
    ok = curve[curve["quality_retained"] >= target]
    return ok.sort_values(["cost_usd", "share_small"], ascending=[True, False]).iloc[0]


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC, or None when only one class is present."""
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))
