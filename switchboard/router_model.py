"""Load the trained router and score prompts. Shared by evaluation and the proxy."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from switchboard.config import settings


class RouterModel:
    """Wraps the fine tuned classifier. predict() returns P(small model is good enough)."""

    def __init__(self, path: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.small_ok_index = self.model.config.label2id.get("small_ok", 1)
        self.max_length = settings().dataset.max_prompt_tokens

    @torch.inference_mode()
    def predict(self, prompts: list[str], batch_size: int = 32) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(prompts), batch_size):
            batch = self.tokenizer(
                prompts[start : start + batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**batch).logits.float(), dim=-1)
            scores.extend(probs[:, self.small_ok_index].tolist())
        return scores
