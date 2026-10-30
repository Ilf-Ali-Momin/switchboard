#!/bin/bash
# Resume the step 3 judge run on the Groq free tier.
# Run by launchd every 6 hours (see scripts/com.switchboard.judge-resume.plist; replace
# __REPO__ in it with the absolute repo path and copy it to ~/Library/LaunchAgents).
# Skips when a run is live, when the run is complete, or when the last run hit a real error.
# Judgments still invalid after the retry are final (the cache returns the same reply), so
# the run is complete when nothing is pending on the daily quota and there are no other errors.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# launchd runs with a minimal PATH, so fall back to uv's default install location.
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
LOG="$ROOT/data/judge_full.log"
RESUME_LOG="$ROOT/data/judge_resume.log"

note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$RESUME_LOG"; }

cd "$ROOT" || exit 1

if pgrep -f "switchboard.data.judge" > /dev/null; then
    note "skip: a judge run is live ($(grep 'judgments done' "$LOG" | tail -1))"
    exit 0
fi

if [ -f "$LOG" ]; then
    if grep -qE "0 pending on daily quota, 0 other errors" "$LOG"; then
        note "skip: step 3 complete"
        exit 0
    fi
    # A 429 or the daily quota is transient. Anything else needs a human.
    real_error=$(grep -E "^error:|^Traceback" "$LOG" | grep -vE "429|InvalidJudgment" | head -1)
    if [ -n "$real_error" ]; then
        note "skip: last run hit a real error, see $LOG: $real_error"
        exit 0
    fi
    cp "$LOG" "$ROOT/data/judge_full.prev.log"
fi

note "start: cache has $(wc -l < data/llm_cache.jsonl | tr -d ' ') entries"
"$UV" run python -u -m switchboard.data.judge > "$LOG" 2>&1
status=$?
note "end: exit $status, $(grep -E ' judgments: ' "$LOG" | tail -1)"
