"""
[T][#2][#4] Generate teacher text responses for train_unlabeled split,
and precompute logits on the generated sequences (teacher forcing on synthetic text).

For each example:
  1. Generate response text (autoregressive)           → artifacts/responses/
  2. Forward pass on [prompt + response] → logits      → artifacts/logits/synthetic/

Run from project root: python teacher/generate_responses.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx_lm import load, generate
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESPONSES_PATH = ROOT / "artifacts" / "responses" / "train_unlabeled_responses.jsonl"
LOGITS_DIR     = ROOT / "artifacts" / "logits" / "synthetic"


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_split(split: str) -> list[dict]:
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "At the end write the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def get_logits(model, input_ids: list[int]) -> np.ndarray:
    """Single forward pass over a complete sequence. Returns float16 [seq_len, vocab_size]."""
    x = mx.array(input_ids, dtype=mx.int32)[None]  # [1, seq_len]
    logits = model(x)[0].astype(mx.float16)         # [seq_len, vocab_size]
    mx.eval(logits)
    return np.array(logits)


def load_already_done() -> set[int]:
    done = set()
    if RESPONSES_PATH.exists():
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["idx"])
    return done


def main() -> None:
    cfg = load_config()
    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_split("train_unlabeled")
    already_done = load_already_done()

    if already_done:
        print(f"Resuming: {len(already_done)} examples already done, skipping")

    remaining = len(examples) - len(already_done)
    print(f"To process: {remaining} examples")
    print(f"Loading teacher: {cfg['teacher_model']}...")

    model, tokenizer = load(cfg["teacher_model"])

    with open(RESPONSES_PATH, "a", encoding="utf-8") as f_resp:
        for idx, ex in tqdm(enumerate(examples), total=len(examples), desc="Teacher"):
            if idx in already_done:
                continue

            prompt = make_prompt(ex["question"])

            # Step 1: generate response text
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=cfg["inference"]["max_new_tokens"],
                verbose=False,
            )

            # Step 2: forward pass on [prompt + response] to get logits
            full_sequence = prompt + response
            input_ids = tokenizer.encode(full_sequence)
            logits = get_logits(model, input_ids)

            # Save text
            f_resp.write(json.dumps({
                "idx": idx,
                "question": ex["question"],
                "teacher_response": response,
            }, ensure_ascii=False) + "\n")
            f_resp.flush()

            # Save logits
            np.savez_compressed(
                LOGITS_DIR / f"{idx}.npz",
                logits=logits,
                input_ids=np.array(input_ids, dtype=np.int32),
            )

    total_mb = sum(p.stat().st_size for p in LOGITS_DIR.glob("*.npz")) / 1e6
    print(f"\nDone.")
    print(f"Responses → {RESPONSES_PATH}")
    print(f"Logits    → {LOGITS_DIR}/ ({total_mb:.1f} MB)")


if __name__ == "__main__":
    main()
