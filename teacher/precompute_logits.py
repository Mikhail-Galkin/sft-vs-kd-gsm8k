"""
[T][#3] Precompute teacher logits for the train split (teacher forcing).

For each example in train, we do a single forward pass with the full
[question + GT answer] sequence and save the per-token logit distributions.

Output: artifacts/logits/gt/{idx}.npz
  Each file contains:
    logits: float16 array of shape [seq_len, vocab_size]
    input_ids: int32 array of shape [seq_len]  (for alignment verification)

Run from project root: python teacher/precompute_logits.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
from mlx_lm import load


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_split(split: str) -> list[dict]:
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_sequence(question: str, answer: str) -> str:
    """Full [question + answer] sequence fed to teacher (teacher forcing)."""
    return (
        "Solve the following math problem step by step. "
        "At the end write the final answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution: {answer}"
    )


def forward_pass_logits(model, input_ids: mx.array) -> np.ndarray:
    """Run one forward pass, return logits as float16 numpy array."""
    logits = model(input_ids[None])[0].astype(mx.float16)  # [seq_len, vocab_size]
    mx.eval(logits)
    return np.array(logits)


def main() -> None:
    cfg = load_config()
    logits_dir = ROOT / "artifacts" / "logits" / "gt"
    logits_dir.mkdir(parents=True, exist_ok=True)

    examples = load_split("train")
    print(f"Loaded {len(examples)} examples from train")

    print(f"Loading teacher model: {cfg['teacher_model']} (4-bit)...")
    model, tokenizer = load(cfg["teacher_model"])
    model.eval()

    already_done = {int(p.stem) for p in logits_dir.glob("*.npz")}
    if already_done:
        print(f"Resuming: {len(already_done)} examples already processed")

    for idx, ex in tqdm(enumerate(examples), total=len(examples), desc="Precomputing logits"):
        if idx in already_done:
            continue

        sequence = make_sequence(ex["question"], ex["answer"])
        input_ids_list = tokenizer.encode(sequence)
        input_ids = mx.array(input_ids_list, dtype=mx.int32)

        logits = forward_pass_logits(model, input_ids)

        out_path = logits_dir / f"{idx}.npz"
        np.savez_compressed(
            out_path,
            logits=logits,
            input_ids=np.array(input_ids_list, dtype=np.int32),
        )

    total_size_mb = sum(p.stat().st_size for p in logits_dir.glob("*.npz")) / 1e6
    print(f"\nDone. {len(list(logits_dir.glob('*.npz')))} files, {total_size_mb:.1f} MB total")
    print(f"Logits saved to {logits_dir}/")


if __name__ == "__main__":
    main()
