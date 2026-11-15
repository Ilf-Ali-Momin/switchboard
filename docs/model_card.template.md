---
license: apache-2.0
language:
- en
base_model: {{encoder_model}}
datasets:
- {{dataset_repo}}
pipeline_tag: text-classification
library_name: transformers
tags:
- llm-routing
---

# Switchboard router

A learned LLM router. Given a prompt, it predicts the probability that a cheap small model's
answer is good enough, so a request only goes to an expensive large model when it needs to.
Code and the OpenAI compatible proxy: {{github_url}}. Demo: [{{space_repo}}](https://huggingface.co/spaces/{{space_repo}}).

> **{{headline}}**

![Cost vs quality on the test split](cost_vs_quality.png)

{{beats_random_sentence}} The threshold was picked on validation; all numbers are on the held out
test split ({{n_test}} prompts).

{{baseline_table}}

## Intended use

Route each prompt to `{{small_model}}` when the score is at or above the threshold
({{threshold}} at the operating point above), otherwise to `{{large_model}}`. Raise the threshold
for more quality, lower it for lower cost. It is meant for prompts like the training data:
English, single turn, general instructions, factual questions, summarization, math and coding.

```python
from transformers import pipeline

router = pipeline("text-classification", model="{{model_repo}}", top_k=None)
scores = {r["label"]: r["score"] for r in router("What is the capital of France?")[0]}
use_small = scores["small_ok"] >= {{threshold}}
```

## Training

- Base model: `{{encoder_model}}`, fine tuned as a binary classifier on the prompt alone.
- Data: [{{dataset_repo}}](https://huggingface.co/datasets/{{dataset_repo}}), {{n_train}} train
  prompts, {{train_small_ok}} labelled `small_ok`. Labels compare judge scores of both models'
  answers; the judge is `{{judge_model}}`.
- The learning rate was tuned on validation (best {{best_lr}}). ROC AUC: {{val_auc}} on
  validation, {{test_auc}} on test.
- CPU latency per prompt ({{lat_threads}} threads, Apple silicon): p50 {{lat_p50}} ms, p95
  {{lat_p95}} ms.

## Limitations

- **English only.**
- **Judge bias.** Quality is measured by an LLM judge that agrees with hand labels on
  {{cal_agreement}} of rows (kappa {{cal_kappa}}) and is more lenient than a human. "Quality
  kept" means kept by the judge's standard.
- **Tied to one model pair.** Trained for `{{small_model}}` and `{{large_model}}` on Groq as of
  {{prices_checked_on}}. Other models, or newer versions of these, need new labels.
- **Small evaluation.** {{n_test}} test prompts, so per category numbers are noisy.
