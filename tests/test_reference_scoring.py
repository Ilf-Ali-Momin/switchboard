import shutil

import pytest

from switchboard.data import reference_scoring as rs
from switchboard.data.references import gsm8k_gold

needs_sandbox = pytest.mark.skipif(
    shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec not available"
)

TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]


# --- gsm8k ---


def test_gsm8k_gold_takes_text_after_hashes():
    assert gsm8k_gold("She has 3 + 2 = 5 apples.\n#### 1,250") == "1250"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("So the total is **18** dollars.", 18.0),
        ("Step 1: 4 * 3 = 12. Step 2: 12 + 6 = 18", 18.0),
        ("The answer is $1,250.", 1250.0),
        ("We get \\boxed{42} after checking 7 cases.", 42.0),
        ("It costs -3.5 degrees", -3.5),
        ("no numbers here", None),
    ],
)
def test_final_number(answer, expected):
    assert rs.final_number(answer) == expected


def test_score_gsm8k_correct_wrong_and_empty():
    assert rs.score_gsm8k("Total: 1,250", "1250").score == rs.CORRECT
    assert rs.score_gsm8k("Total: 18.0", "18").score == rs.CORRECT
    assert rs.score_gsm8k("Total: 17", "18").score == rs.WRONG
    assert rs.score_gsm8k("", "18").score == rs.WRONG


# --- mbpp code extraction ---


def test_extract_code_prefers_defining_blocks():
    answer = (
        "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\n"
        "Usage:\n```python\nprint(add(1, 2))\n```"
    )
    code = rs.extract_code(answer)
    assert "def add" in code and "print(add" not in code


def test_extract_code_without_fences_returns_answer():
    assert rs.extract_code("def f():\n    return 1") == "def f():\n    return 1"


# --- mbpp sandbox ---


@needs_sandbox
def test_correct_code_passes():
    answer = "```python\ndef add(a, b):\n    return a + b\n```"
    assert rs.score_mbpp(answer, "", TESTS).score == rs.CORRECT


@needs_sandbox
def test_wrong_code_fails_and_reports_progress():
    answer = "```python\ndef add(a, b):\n    return a + b + (a < 0)\n```"
    result = rs.score_mbpp(answer, "", TESTS)
    assert result.score == rs.WRONG
    assert "passed 1/2" in result.reason


@needs_sandbox
def test_main_guard_does_not_run():
    answer = (
        "```python\ndef add(a, b):\n    return a + b\n\n"
        "if __name__ == '__main__':\n    input()\n```"
    )
    assert rs.score_mbpp(answer, "", TESTS).score == rs.CORRECT


@needs_sandbox
def test_setup_code_runs_before_tests():
    answer = "```python\ndef twice(x):\n    return 2 * x\n```"
    assert rs.score_mbpp(answer, "K = 4", ["assert twice(K) == 8"]).score == rs.CORRECT


@needs_sandbox
def test_infinite_loop_times_out(monkeypatch):
    monkeypatch.setattr(rs, "MBPP_TIMEOUT_S", 2)
    answer = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    result = rs.score_mbpp(answer, "", TESTS)
    assert result.score == rs.WRONG
    assert "timeout" in result.reason or "crashed" in result.reason


@needs_sandbox
def test_sandbox_blocks_network_and_writes_outside_workdir(tmp_path):
    target = tmp_path / "escaped.txt"
    code = (
        "import socket\n"
        "def net():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "        return 'open'\n"
        "    except OSError:\n"
        "        return 'blocked'\n"
        "def write():\n"
        "    try:\n"
        f"        open({str(target)!r}, 'w').write('x')\n"
        "        return 'open'\n"
        "    except OSError:\n"
        "        return 'blocked'\n"
    )
    passed, error = rs.run_tests(
        code, "", ["assert net() == 'blocked'", "assert write() == 'blocked'"]
    )
    assert (passed, error) == (2, None)
    assert not target.exists()


def test_empty_answer_is_wrong_without_running_anything():
    assert rs.score_mbpp("  ", "", TESTS).score == rs.WRONG
