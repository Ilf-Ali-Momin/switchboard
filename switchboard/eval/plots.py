"""The cost vs quality chart: router curve against the baselines, on test."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1-3 (light surface, all pairs). Aqua is under 3:1 contrast, so
# every line is also direct labelled and carries its own dash pattern.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = {
    "router": ("#2a78d6", "-", "Switchboard router"),
    "random": ("#eb6834", "--", "Random routing (mean of 20 seeds)"),
    "length": ("#1baf7a", ":", "Prompt length heuristic"),
}
# Where each direct label sits, as a share of the x range, and its offset in points.
LABEL_SPOT = {"router": (0.75, (0, -14)), "random": (0.55, (0, 9)), "length": (0.4, (0, 6))}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _pct(values) -> list[float]:
    return [100 * v for v in values]


def plot_cost_vs_quality(curves: dict, results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    for key, (color, dash, label) in SERIES.items():
        curve = curves[key].sort_values("cost_relative")
        x, y = _pct(curve["cost_relative"]), _pct(curve["quality_retained"])
        ax.plot(x, y, color=color, linestyle=dash, linewidth=2, label=label)
        frac, offset = LABEL_SPOT[key]
        x_target = min(x) + frac * (max(x) - min(x))
        i = min(range(len(x)), key=lambda j: abs(x[j] - x_target))
        ax.annotate(
            label.split(" (")[0],
            (x[i], y[i]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=color if key != "length" else TEXT_MUTED,
            ha="center",
        )

    target = results["operating_point"]["target_quality_retained"]
    ax.axhline(100 * target, color=TEXT_MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.annotate(
        f"{target:.0%} quality target",
        (0.01, 100 * target),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT_MUTED,
    )

    for key, name in (("always_small", "Always small"), ("always_large", "Always large")):
        p = curves[key]
        ax.scatter(
            100 * p["cost_relative"],
            100 * p["quality_retained"],
            s=64,
            color=TEXT,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=5,
        )
        ax.annotate(
            name,
            (100 * p["cost_relative"], 100 * p["quality_retained"]),
            xytext=(-8, -16) if key == "always_small" else (-8, -16),
            ha="left" if key == "always_small" else "right",
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT,
        )

    op = curves["operating_point"]
    threshold = results["operating_point"]["threshold"]
    ax.scatter(
        100 * op["cost_relative"],
        100 * op["quality_retained"],
        s=110,
        color=SERIES["router"][0],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=6,
    )
    on_left = op["cost_relative"] < 0.75  # keep the callout inside the plot
    ax.annotate(
        f"threshold {threshold:.2f}: {op['share_small']:.0%} to small,\n"
        f"{op['quality_retained']:.1%} quality, {op['cost_reduction']:.0%} cheaper",
        (100 * op["cost_relative"], 100 * op["quality_retained"]),
        xytext=(12, 18) if on_left else (-12, 18),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT,
        ha="left" if on_left else "right",
        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.8},
    )

    ax.set_xlabel("Cost, % of always using the large model", fontsize=10, color=TEXT)
    ax.set_ylabel("Quality, % of the large model's", fontsize=10, color=TEXT)
    ax.set_title("Cost vs quality on the test split", fontsize=12, color=TEXT, loc="left", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
