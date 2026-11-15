"""Export the trained router to ONNX for transformers.js, the browser runtime of the static Space.

Writes onnx/model.onnx (fp32) and onnx/model_fp16.onnx into the model folder, the layout
transformers.js expects, then checks both against the PyTorch scores on test prompts. The
Space uses fp16: dynamic int8 quantization moved scores by up to 0.18, fp16 by about 0.001.

    uv run --group export python -m switchboard.train.export_onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from switchboard.config import ROOT, settings

INPUTS = ("input_ids", "attention_mask")


def export(model_dir: Path, out: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sample = tokenizer(
        ["a short prompt", "a second, longer prompt here"], padding=True, return_tensors="pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    dynamic = {name: {0: "batch", 1: "sequence"} for name in INPUTS}
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(out),
        input_names=list(INPUTS),
        output_names=["logits"],
        dynamic_axes=dynamic | {"logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )


def to_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    onnx.save(convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))


def small_ok_probs(logits: np.ndarray, index: int) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e[:, index] / e.sum(axis=1)


def compare(model_dir: Path, onnx_path: Path, prompts: list[str]) -> dict[str, float]:
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    index = model.config.label2id.get("small_ok", 1)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_len = settings().dataset.max_prompt_tokens
    diffs = []
    for prompt in prompts:
        enc = tokenizer([prompt], truncation=True, max_length=max_len, return_tensors="pt")
        with torch.inference_mode():
            ref = small_ok_probs(model(**enc).logits.numpy(), index)
        got = session.run(None, {k: enc[k].numpy() for k in INPUTS})[0]
        diffs.append(abs(float(small_ok_probs(got, index)[0] - ref[0])))
    return {"max_abs_diff": max(diffs), "mean_abs_diff": float(np.mean(diffs))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / settings().training.output_dir)
    parser.add_argument("--n-check", type=int, default=50)
    args = parser.parse_args()
    fp32 = args.model_dir / "onnx" / "model.onnx"
    fp16 = args.model_dir / "onnx" / "model_fp16.onnx"
    export(args.model_dir, fp32)
    to_fp16(fp32, fp16)
    prompts = pd.read_parquet(ROOT / "data" / "prompts.parquet").query("split == 'test'")
    sample = prompts.sample(n=args.n_check, random_state=0)["prompt"].tolist()
    for path in (fp32, fp16):
        mb = path.stat().st_size / 1e6
        print(f"{path.name}: {mb:.0f} MB, vs PyTorch {compare(args.model_dir, path, sample)}")


if __name__ == "__main__":
    main()
