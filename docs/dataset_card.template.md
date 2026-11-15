---
license: cc-by-sa-3.0
language:
- en
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- llm-routing
- llm-as-a-judge
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# Switchboard routing labels

{{n_prompts}} English prompts, each answered by a small and a large LLM, with a quality score for
every answer and a label saying whether the small model's answer was good enough. It is the
training data for [{{model_repo}}](https://huggingface.co/{{model_repo}}), a learned router.
Code: {{github_url}}.

Splits ({{split_counts}}) are stratified by source with a fixed seed and were frozen before any
results were seen.

## Columns

| Column | Meaning |
|---|---|
| `id` | Source dataset and row, e.g. `dolly-10138` |
| `prompt` | The user prompt, single turn |
| `source`, `category`, `license` | Where the prompt comes from, its task type and its license |
| `small_answer`, `large_answer` | Answers from `{{small_model}}` and `{{large_model}}`, temperature 0 |
| `small_output_tokens`, `large_output_tokens` | Output tokens of each answer |
| `small_score`, `large_score` | Quality score of each answer, 1 to 5 |
| `small_reason`, `large_reason` | The judge's short reason (LLM judged rows) |
| `scorer` | `llm_judge`, `gsm8k_exact` (final number matches the reference) or `mbpp_tests` (the reference tests pass) |
| `small_ok` | `small_score >= large_score`; null where a judgment failed |

## Sources and licenses

{{source_table}}

The dataset is released under CC BY-SA 3.0 because the Dolly prompts carry share alike terms.
The other sources (MIT, Apache 2.0, CC BY 4.0) allow this. The per row `license` column keeps
each prompt's original license. Answers come from `{{small_model}}` and `{{large_model}}`
(Apache 2.0 open weights) served by Groq; the judge is `{{judge_model}}` (Apache 2.0).

## How the labels were made

- Math (`openai/gsm8k`) and coding (`google-research-datasets/mbpp`) answers are scored against
  the dataset's reference answer.
- Every other answer is scored 1 to 5 by `{{judge_model}}` on correctness, completeness and
  following the instruction. The judge sees the prompt and one answer, never both answers and
  never the model name. Invalid judge output was retried once; {{n_failed}} prompts still failed
  and have no label.
- `small_ok` is true when the small answer scores at least as high as the large one.
  {{label_share}} of the {{n_labelled}} labelled prompts are `small_ok`.

{{balance_table}}

## Calibration

The judge was checked against {{cal_n}} hand labelled rows from the train split, shown in random
order without model names: {{cal_agreement}} agreement, Cohen's kappa {{cal_kappa}}. The judge
calls the small answer good enough on {{cal_judge_ok}} of those rows, the hand labels on
{{cal_human_ok}}.

## Known biases

- **Lenient judge.** The judge accepts the small answer more often than a human did, so the
  labels lean toward `small_ok`.
- **Imbalanced classes.** Coding, math and summarization are mostly `small_ok`.
- **One model pair.** Labels describe these two models on Groq as of {{prices_checked_on}}.
  Another model pair needs new answers and labels.
- **English only**, and only the task types in the four sources. Long answers that hit the
  token limit are scored as they are.
