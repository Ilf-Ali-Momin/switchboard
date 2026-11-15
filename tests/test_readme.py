import pytest

from switchboard.eval import readme


def test_render_fills_placeholders():
    assert readme.render("At {{t}}, {{x}}.", {"t": "0.79", "x": "80%"}) == "At 0.79, 80%."


def test_render_rejects_unknown_placeholder():
    with pytest.raises(KeyError, match="missing_one"):
        readme.render("{{missing_one}}", {})


def test_render_leaves_single_braces_alone():
    assert readme.render('{"model": "auto"} {{a}}', {"a": "x"}) == '{"model": "auto"} x'


def test_template_renders_from_committed_results():
    text = readme.render(
        readme.TEMPLATE_PATH.read_text(), readme.values(readme.load_results(readme.RESULTS_DIR))
    )
    assert "{{" not in text
    assert readme.load_results(readme.RESULTS_DIR)["results"]["headline"] in text


def test_baseline_table_marks_router_and_formats_percent():
    results = {
        "operating_point": {"test": _row(0.8, 4.3, 0.98, 0.3)},
        "test_baselines_at_same_share": {
            "random": _row(0.8, 4.23, 0.965, 0.36),
            "length_heuristic": _row(0.8, 4.19, 0.955, 0.37),
        },
        "test_baselines": {
            "always_small": _row(1.0, 4.19, 0.956, 0.44),
            "always_large": _row(0.0, 4.39, 1.0, 0.0),
        },
    }
    table = readme.baseline_table(results)
    assert "| **Switchboard router** | 80% | 4.30 | 98.0% | 30% |" in table
    assert "| Always large | 0% | 4.39 | 100.0% | 0% |" in table


def _row(share: float, quality: float, kept: float, cut: float) -> dict:
    return {
        "share_small": share,
        "mean_quality": quality,
        "quality_retained": kept,
        "cost_reduction": cut,
    }
