# Switchboard

A learned LLM router. For each prompt, a small fine tuned classifier decides whether a cheap
small model is good enough or the request needs an expensive large model.

The headline result and chart will go here once `results/results.json` exists. No numbers are
written by hand.

## Reproduce

bash
uv sync
uv run python -m switchboard.data.collect --limit 20

