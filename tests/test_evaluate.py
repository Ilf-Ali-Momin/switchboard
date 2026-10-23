import numpy as np
import pandas as pd
import pytest

from switchboard.eval import baselines
from switchboard.eval import evaluate as ev
from switchboard.eval.metrics import pick_threshold, routing_metrics, sweep, threshold_grid


def toy() -> pd.DataFrame:
    # Prompts 0-1: small is as good as large. Prompts 2-3: small is worse.
    return pd.DataFrame(
        {
            "small_score": [5, 4, 1, 2],
            "large_score": [5, 4, 5, 4],
            "small_cost": [1.0, 1.0, 1.0, 1.0],
            "large_cost": [3.0, 3.0, 3.0, 3.0],
            "prompt_tokens": [10, 20, 30, 40],
            "category": ["a", "a", "b", "b"],
            "small_ok": [True, True, False, False],
        }
    )


def test_routing_metrics_math():
    m = routing_metrics(toy(), np.array([True, True, False, False]))
    assert m["share_small"] == 0.5
    assert m["mean_quality"] == pytest.approx((5 + 4 + 5 + 4) / 4)
    assert m["quality_retained"] == pytest.approx(1.0)
    assert m["cost_usd"] == pytest.approx(8.0)
    assert m["cost_relative"] == pytest.approx(8 / 12)
    assert m["cost_reduction"] == pytest.approx(1 - 8 / 12)


def test_always_small_and_large():
    small = baselines.always(toy(), True)
    large = baselines.always(toy(), False)
    assert large["quality_retained"] == 1.0 and large["cost_reduction"] == 0.0
    assert small["quality_retained"] == pytest.approx(12 / 18)
    assert small["cost_relative"] == pytest.approx(4 / 12)


def test_threshold_grid_covers_all_large():
    grid = threshold_grid()
    assert grid[0] == 0.0 and grid[-1] > 1.0 and len(grid) == 102


def test_sweep_routes_score_at_or_above_threshold_to_small():
    curve = sweep(toy(), np.array([0.9, 0.8, 0.3, 0.1]), np.array([0.0, 0.5, 0.85, 1.01]))
    assert curve["share_small"].tolist() == [1.0, 0.5, 0.25, 0.0]
    assert curve["quality_retained"].tolist()[1] == pytest.approx(1.0)


def test_pick_threshold_is_cheapest_point_meeting_target():
    curve = sweep(toy(), np.array([0.9, 0.8, 0.3, 0.1]), threshold_grid())
    op = pick_threshold(curve, target=0.95)
    assert op["share_small"] == 0.5  # sending the two easy prompts keeps 100% quality
    assert op["quality_retained"] >= 0.95
    assert op["cost_usd"] == pytest.approx(8.0)


def test_pick_threshold_falls_back_to_all_large():
    curve = sweep(toy(), np.array([0.1, 0.1, 0.9, 0.9]), threshold_grid())
    op = pick_threshold(curve, target=0.99)
    assert op["share_small"] == 0.0


def test_random_routing_matches_expected_share_and_quality():
    df = pd.concat([toy()] * 50, ignore_index=True)
    r = baselines.random_routing(df, 0.5)
    assert r["share_small"] == 0.5
    expected = routing_metrics(df, np.ones(len(df), bool))["mean_quality"] * 0.5 + 4.5 * 0.5
    assert r["mean_quality"] == pytest.approx(expected, abs=0.1)


def test_length_heuristic_sends_shortest_prompts():
    m = baselines.length_at_share(toy(), 0.5)
    assert m["quality_retained"] == pytest.approx(1.0)  # the two shortest are the easy ones
    curve = baselines.length_curve(toy())
    assert curve["share_small"].min() == 0.0 and curve["share_small"].max() == 1.0


def test_beats_random_and_headline():
    df = pd.concat([toy()] * 25, ignore_index=True)
    perfect = df["small_ok"].to_numpy()
    result = ev.beats_random(df, perfect)
    assert result["beats_random"] and result["seeds_beaten"] == "20/20"
    text = ev.headline(0.42, routing_metrics(df, perfect))
    assert text == (
        "At threshold 0.42, Switchboard sends 50% of prompts to the small model and keeps "
        "100.0% of the large model's quality, cutting cost by 33%."
    )


def test_evaluate_picks_on_val_and_reports_test(tmp_path):
    val = toy().assign(split="val")
    test = toy().assign(split="test", small_ok=[False, True, True, False])
    df = pd.concat([val, test], ignore_index=True)
    scores = np.array([0.9, 0.8, 0.3, 0.1] * 2)
    results, curves = ev.evaluate(df, scores, target=0.95)
    assert results["operating_point"]["picked_on"] == "val"
    assert results["operating_point"]["threshold"] == pytest.approx(0.31)
    assert results["operating_point"]["test"]["share_small"] == 0.5
    assert results["n"] == {"val": 4, "test": 4}
    from switchboard.eval.plots import plot_cost_vs_quality

    plot_cost_vs_quality(curves, results, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 10_000


def test_beats_random_is_undefined_when_everything_goes_to_one_model():
    df = pd.concat([toy()] * 5, ignore_index=True)
    assert ev.beats_random(df, np.ones(len(df), bool))["beats_random"] is None
    assert ev.beats_random(df, np.zeros(len(df), bool))["beats_random"] is None
