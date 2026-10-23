#!/bin/bash
# Resume the step 2 generate run on the Groq free tier.
# Run by launchd every 6 hours (see scripts/com.switchboard.generate-resume.plist; replace
# __REPO__ in it with the absolute repo path and copy it to ~/Library/LaunchAgents).
# Skips when a run is live, when the run is complete, or when the last run hit a real error.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# launchd runs with a minimal PATH, so fall back to uv's default install location.
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
LOG="$ROOT/data/generate_full.log"
RESUME_LOG="$ROOT/data/resume.log"

note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$RESUME_LOG"; }

cd "$ROOT" || exit 1

if pgrep -f "switchboard.data.generate" > /dev/null; then
    note "skip: a generate run is live ($(grep 'calls done' "$LOG" | tail -1))"
    exit 0
fi

if [ -f "$LOG" ]; then
    if grep -q "^4000 answers, 0 failed" "$LOG"; then
        note "skip: step 2 complete"
        exit 0
    fi
    # A 429 that outlasted the retries is transient. Anything else needs a human.
    real_error=$(grep -E "^error:|^Traceback" "$LOG" | grep -v "429" | head -1)
    if [ -n "$real_error" ]; then
        note "skip: last run hit a real error, see $LOG: $real_error"
        exit 0
    fi
    cp "$LOG" "$ROOT/data/generate_full.prev.log"
fi

note "start: cache has $(wc -l < data/llm_cache.jsonl | tr -d ' ') entries"
"$UV" run python -u -m switchboard.data.generate > "$LOG" 2>&1
status=$?
note "end: exit $status, $(grep -E 'answers,' "$LOG" | tail -1)"
