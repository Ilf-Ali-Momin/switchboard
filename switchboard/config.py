"""Load config/settings.yaml into typed objects. All tunables live in that file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


class TokenPrice(BaseModel):
    input: float
    output: float


class Prices(BaseModel):
    checked_on: str
    small: TokenPrice
    large: TokenPrice


class GroqSettings(BaseModel):
    base_url: str
    concurrency: int
    max_retries: int
    timeout_s: float
    cache_path: str


class GenerationSettings(BaseModel):
    max_tokens: int
    reasoning_effort: str


class SplitRatios(BaseModel):
    train: float
    val: float
    test: float


class DatasetSettings(BaseModel):
    total_prompts: int
    max_prompt_tokens: int
    split: SplitRatios
    seed: int
    source_mix: dict[str, float]


class LabelingSettings(BaseModel):
    margin: float


class EvalSettings(BaseModel):
    target_quality_retained: float


class Settings(BaseModel):
    small_model: str
    large_model: str
    judge_model: str
    encoder_model: str
    prices_per_million_tokens: Prices
    groq: GroqSettings
    generation: GenerationSettings
    dataset: DatasetSettings
    labeling: LabelingSettings
    eval: EvalSettings


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    return Settings.model_validate(yaml.safe_load(path.read_text()))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return load_settings()
