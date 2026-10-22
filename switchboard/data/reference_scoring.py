"""Score answers against ground truth instead of the LLM judge.

gsm8k: the answer's final number must equal the gold answer.
mbpp:  the answer's code must pass every test case, run in a sandboxed subprocess.

Scores use the judge's scale so all sources stay comparable: 5 when correct, 1 when not.
"""

from __future__ import annotations

import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CORRECT, WRONG = 5, 1
MBPP_TIMEOUT_S = 10
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class RefScore:
    score: int
    reason: str


# --- gsm8k ---


def to_number(text: str) -> float | None:
    match = NUMBER.findall(text.replace("$", ""))
    if not match:
        return None
    try:
        return float(match[-1].replace(",", ""))
    except ValueError:
        return None


def final_number(answer: str) -> float | None:
    """The answer's final number: the last \\boxed{} value if any, else the last number."""
    boxed = BOXED.findall(answer)
    if boxed and (value := to_number(boxed[-1])) is not None:
        return value
    return to_number(answer)


def score_gsm8k(answer: str, gold: str) -> RefScore:
    predicted, expected = final_number(answer), to_number(gold)
    if expected is None:
        raise ValueError(f"gold answer is not a number: {gold!r}")
    if predicted is None:
        return RefScore(WRONG, f"no number in answer; gold {gold}")
    if abs(predicted - expected) < 1e-6:
        return RefScore(CORRECT, f"final answer {predicted:g} matches gold {gold}")
    return RefScore(WRONG, f"final answer {predicted:g} != gold {gold}")


# --- mbpp ---


def extract_code(answer: str) -> str:
    """Join the fenced code blocks that define something; fall back to the raw answer."""
    blocks = FENCE.findall(answer)
    if not blocks:
        return answer
    defining = [b for b in blocks if re.search(r"^\s*(def|class|import|from)\s", b, re.M)]
    return "\n\n".join(defining or blocks)


# Runs inside the sandbox. The solution is exec'd under a non-main __name__, so example usage
# guarded by `if __name__ == "__main__":` does not run.
HARNESS = """
import json, sys
spec = json.load(open("spec.json"))
ns = {"__name__": "solution"}
try:
    exec(compile(spec["code"], "solution.py", "exec"), ns)
    exec(spec["setup"], ns)
except BaseException as exc:
    print(json.dumps({"passed": 0, "error": f"load: {type(exc).__name__}: {exc}"[:300]}))
    sys.exit(0)
passed, error = 0, None
for test in spec["tests"]:
    try:
        exec(test, ns)
        passed += 1
    except BaseException as exc:
        error = error or f"{test[:80]} -> {type(exc).__name__}: {exc}"[:300]
print(json.dumps({"passed": passed, "error": error}))
"""


def sandbox_profile(workdir: Path) -> str:
    """macOS sandbox: no network, writes only inside the temporary work folder."""
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f'(allow file-write* (subpath "{workdir.resolve()}"))'
        '(allow file-write* (literal "/dev/null"))'
    )


def limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (MBPP_TIMEOUT_S, MBPP_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))


def run_tests(code: str, setup: str, tests: list[str]) -> tuple[int, str | None]:
    """Run tests against code in a sandbox. Returns (tests passed, first error)."""
    if shutil.which("sandbox-exec") is None:
        raise RuntimeError("sandbox-exec not found: refusing to run model code unsandboxed")
    with tempfile.TemporaryDirectory(prefix="mbpp-") as tmp:
        workdir = Path(tmp)
        (workdir / "spec.json").write_text(
            json.dumps({"code": code, "setup": setup, "tests": tests})
        )
        (workdir / "harness.py").write_text(HARNESS)
        cmd = [
            "sandbox-exec",
            "-p",
            sandbox_profile(workdir),
            sys.executable,
            "-I",
            "-B",
            "harness.py",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=MBPP_TIMEOUT_S,
                env={"PATH": "/usr/bin:/bin"},
                preexec_fn=limit_resources,
            )
        except subprocess.TimeoutExpired:
            return 0, f"timeout after {MBPP_TIMEOUT_S}s"
    lines = proc.stdout.strip().splitlines()
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return 0, f"crashed: {proc.stderr.strip()[-200:]}"
    return int(result["passed"]), result["error"]


def score_mbpp(answer: str, setup: str, tests: list[str]) -> RefScore:
    if not answer.strip():
        return RefScore(WRONG, "empty answer")
    passed, error = run_tests(extract_code(answer), setup, tests)
    if passed == len(tests):
        return RefScore(CORRECT, f"passed {passed}/{len(tests)} tests")
    return RefScore(WRONG, f"passed {passed}/{len(tests)} tests; {error}")
