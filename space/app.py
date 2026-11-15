"""Switchboard demo Space: type a prompt, see the routing score and decision.

Loads the router from the Hub. Makes no LLM calls and holds no API key. The examples tab shows
test prompts with both models' answers and their scores from the published dataset.

Settings come from settings.env next to this file (written by switchboard.data.publish) and can
be overridden by environment variables. MODEL_PATH and DATASET_PATH point at local copies for
testing before anything is on the Hub.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent


def load_settings() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_file = HERE / "settings.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip():
                cfg[key.strip()] = value.strip()
    keys = (
        "MODEL_REPO",
        "DATASET_REPO",
        "THRESHOLD",
        "MAX_PROMPT_TOKENS",
        "MODEL_PATH",
        "DATASET_PATH",
    )
    cfg.update({k: os.environ[k] for k in keys if k in os.environ})
    return cfg


CFG = load_settings()
THRESHOLD = float(CFG.get("THRESHOLD", "0.5"))
MAX_TOKENS = int(CFG.get("MAX_PROMPT_TOKENS", "512"))
MODEL_SOURCE = CFG.get("MODEL_PATH") or CFG["MODEL_REPO"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE).eval()
SMALL_OK = model.config.label2id.get("small_ok", 1)


@torch.inference_mode()
def score(prompt: str) -> float:
    batch = tokenizer(prompt, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    return float(torch.softmax(model(**batch).logits.float(), dim=-1)[0, SMALL_OK])


def decision_text(p: float) -> str:
    model_name = "small model" if p >= THRESHOLD else "large model"
    return (
        f"### Route to the **{model_name}**\n\n"
        f"Score {p:.3f}, the probability that the small model's answer is good enough. "
        f"Threshold {THRESHOLD:.2f}: at or above it the prompt goes to the small model."
    )


def route(prompt: str) -> tuple[float | None, str]:
    if not prompt.strip():
        return None, "Type a prompt first."
    p = score(prompt)
    return round(p, 3), decision_text(p)


def load_examples() -> pd.DataFrame:
    path = CFG.get("DATASET_PATH")
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(CFG["DATASET_REPO"], "data/test.parquet", repo_type="dataset")
    df = pd.read_parquet(path).dropna(subset=["small_ok"])
    # A spread of categories, fixed so every visitor sees the same examples.
    shuffled = df.sample(frac=1, random_state=0)
    return shuffled.groupby("category").head(8).sort_values("category").reset_index(drop=True)


EXAMPLES = load_examples()
CHOICES = [f"[{r.category}] {r.prompt[:90]}" for r in EXAMPLES.itertuples()]


def show_example(choice: str) -> tuple[str, str, str, str]:
    r = EXAMPLES.iloc[CHOICES.index(choice)]
    p = score(r.prompt)
    verdict = "good enough" if r.small_ok else "not good enough"
    summary = (
        f"{decision_text(p)}\n\n"
        f"Judge scores: small **{r.small_score:g}**, large **{r.large_score:g}** "
        f"(scored by `{r.scorer}`), so the small answer was **{verdict}**."
    )
    return r.prompt, summary, r.small_answer, r.large_answer


with gr.Blocks(title="Switchboard: a learned LLM router") as demo:
    gr.Markdown(
        "# Switchboard\nA learned LLM router: a small classifier predicts, from the prompt "
        "alone, whether a cheap small model's answer is good enough. This demo only scores "
        "prompts; it calls no LLM."
    )
    with gr.Tab("Try a prompt"):
        prompt_box = gr.Textbox(label="Prompt", lines=4, placeholder="Ask anything")
        go = gr.Button("Route it", variant="primary")
        probs = gr.Number(label="Score: probability the small model is good enough")
        verdict_md = gr.Markdown()
        go.click(route, prompt_box, [probs, verdict_md])
        prompt_box.submit(route, prompt_box, [probs, verdict_md])
    with gr.Tab("Test set examples"):
        pick = gr.Dropdown(CHOICES, label="Pick a test prompt", value=CHOICES[0])
        ex_prompt = gr.Textbox(label="Prompt", lines=3, interactive=False)
        ex_summary = gr.Markdown()
        with gr.Row():
            small_box = gr.Textbox(label="Small model answer", lines=12, interactive=False)
            large_box = gr.Textbox(label="Large model answer", lines=12, interactive=False)
        outputs = [ex_prompt, ex_summary, small_box, large_box]
        pick.change(show_example, pick, outputs)
        demo.load(show_example, pick, outputs)
    with gr.Tab("Cost vs quality"):
        gr.Image(str(HERE / "cost_vs_quality.png"), show_label=False)

if __name__ == "__main__":
    demo.launch()
