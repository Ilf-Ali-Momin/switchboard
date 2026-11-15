# Switchboard

A learned LLM router. For each prompt, a small fine tuned classifier decides whether a cheap
small model is good enough or the request needs an expensive large model. It runs behind an
OpenAI compatible proxy, so a client only changes its base URL and sets `"model": "auto"`.

[Demo](https://huggingface.co/spaces/Ilf-Ali/switchboard) (runs in the browser) ·
[Model](https://huggingface.co/Ilf-Ali/switchboard-router) ·
[Dataset](https://huggingface.co/datasets/Ilf-Ali/switchboard-routing)

> **{{headline}}**

![Cost vs quality on the test split](results/cost_vs_quality.png)

{{beats_random_sentence}} The threshold was picked on the validation split ({{n_val}} prompts)
as the cheapest point that keeps {{target}} of the large model's quality. All numbers below are
measured on the held out test split ({{n_test}} prompts) at that threshold.

{{baseline_table}}

At the same share the router costs a little more than random routing, because the prompts it
keeps on the large model are the hard ones with long answers. At the same quality it is
cheaper, as the chart shows: random routing needs a larger share of large model calls to reach
the quality target.

Small model: `{{small_model}}`. Large model: `{{large_model}}`. Both on Groq. Cost uses the real
token usage of every answer and Groq's prices per million tokens (input / output), checked on
{{prices_checked_on}}: small {{small_price}}, large {{large_price}}. The small model is half the
price per token but writes longer answers, so even always using it cuts cost by less than half.

<!-- This file is generated from docs/README.template.md by `python -m switchboard.eval.readme`.
     Edit the template, not this file. -->

## Where the router makes its choices

{{category_table}}

Coding, math and summarization prompts almost always get a good enough answer from the small
model, and the router learned to send them all there. The real decisions happen on open ended
instructions and factual questions.

## How it works

1. **Collect prompts** from public Hugging Face datasets: `databricks/databricks-dolly-15k`
   (CC BY-SA 3.0), `OpenAssistant/oasst1` (Apache 2.0), `openai/gsm8k` (MIT) and
   `google-research-datasets/mbpp` (CC BY 4.0). English, single turn, deduplicated, split into
   train, val and test by source with a fixed seed.
2. **Answer every prompt** with both models at temperature 0, recording tokens and latency.
3. **Score every answer.** Math and coding answers are checked against the dataset's reference
   answer. The rest are scored 1 to 5 by an LLM judge (`{{judge_model}}`, a different model
   family) that sees one answer at a time and never learns which model wrote it. The label is
   `small_ok = small_score >= large_score`.
4. **Check the judge** against {{cal_n}} hand labelled rows: {{cal_agreement}} agreement, Cohen's
   kappa {{cal_kappa}}. The judge is more lenient than the hand labels: it calls the small answer
   good enough on {{cal_judge_ok}} of these rows, against {{cal_human_ok}} by hand.
5. **Train** `{{encoder_model}}` as a binary classifier on the prompt alone ({{n_train}} train
   prompts, {{train_small_ok}} labelled `small_ok`). The learning rate was tuned on val (best
   {{best_lr}}). ROC AUC: {{val_auc}} on val, {{test_auc}} on test.
6. **Evaluate** on test against always small, always large, random routing at the same share
   and a prompt length heuristic.
7. **Serve** it as a FastAPI proxy. Classifier overhead on CPU ({{lat_threads}} threads, Apple
   silicon, {{lat_n}} test prompts): p50 {{lat_p50}} ms, p95 {{lat_p95}} ms.

## Use the proxy

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

- `"model": "auto"` routes through the classifier. The small or large model ID passes through
  unchanged. Any other value returns 400.
- `stream: true` passes the upstream server sent events through unchanged.
- Response headers `X-Router-Model` and `X-Router-Score` show the decision.
- If the classifier fails, the request goes to the large model. If the small model fails, it is
  retried once on the large model.
- Every decision is appended to a JSONL log with a hash of the prompt, never the prompt text.
  `GET /stats` summarizes it: request count, share sent to small and estimated savings.

Run it with Docker (needs `models/router/`, `results/results.json` and a `.env` with
`GROQ_API_KEY` and `PROXY_API_KEY`):

```bash
docker compose up -d --build
```

## Reproduce

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and a Groq API key in `.env` (see
`.env.example`). API calls are cached on disk, so reruns are free and a stopped run resumes.

```bash
uv sync
uv run python -m switchboard.data.collect
uv run python -m switchboard.data.generate
uv run python -m switchboard.data.judge
uv run python -m switchboard.data.calibrate
uv run python -m switchboard.train.train
uv run python -m switchboard.eval.evaluate
uv run python -m switchboard.serve.bench
uv run python -m switchboard.eval.readme
uv run uvicorn switchboard.serve.app:app --port 8100
uv run pytest
```

Every script accepts `--limit N` for a small trial run. Training also runs on a free Colab GPU
with `notebooks/train_colab.ipynb`. Model IDs, prices and thresholds live in
`config/settings.yaml`.

## Limitations

- **English only**, and only the kinds of prompts in the four source datasets.
- **Judge bias.** Quality is the judge's score, and the judge is more lenient than a human
  (kappa {{cal_kappa}}). "Quality kept" means kept by the judge's standard.
- **Models change.** The labels describe `{{small_model}}` and `{{large_model}}` on Groq as of
  {{prices_checked_on}}. A different model pair needs new labels.
- **Small test split.** {{n_test}} test prompts, so per category numbers are noisy.

## Prior work

Switchboard follows the idea of learned routing between a cheap and a strong model:

- Ong et al., [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), 2024.
- Chen et al., [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176), 2023.
