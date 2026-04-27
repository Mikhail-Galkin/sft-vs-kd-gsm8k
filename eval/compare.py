"""
[#0][#1][#2][#3][#4] Final evaluation of all five conditions on the test split.

Runs inference for each condition, prints a comparison table, and saves
per-example predictions for inspection.

Usage (from project root):
  python eval/compare.py

Run only after all four training runs are complete.
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import exact_match, extract_answer  # noqa: E402


CONDITIONS = [
    {"id": 0, "name": "Baseline",             "checkpoint": None},
    {"id": 1, "name": "SFT",                  "checkpoint": "checkpoints/sft/best"},
    {"id": 2, "name": "Response distill",     "checkpoint": "checkpoints/response_distill/best"},
    {"id": 3, "name": "Logit KD (GT)",        "checkpoint": "checkpoints/logit_kd_gt/best"},
    {"id": 4, "name": "Logit KD (synthetic)", "checkpoint": "checkpoints/logit_kd_synthetic/best"},
]


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_test_split() -> list[dict]:
    path = ROOT / "data" / "splits" / "test.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "At the end write the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def run_inference(
    model, tokenizer, examples: list[dict], max_new_tokens: int, device: str
) -> list[str]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for ex in tqdm(examples, desc="Inference", leave=False):
            inputs = tokenizer(make_prompt(ex["question"]), return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
    return predictions


def save_predictions(predictions: list[str], examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex, pred in zip(examples, predictions):
            f.write(json.dumps({
                "question": ex["question"],
                "reference": ex["answer"],
                "prediction": pred,
                "correct": extract_answer(pred) == extract_answer(ex["answer"]),
            }, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    max_new_tokens = cfg["inference"]["max_new_tokens"]
    examples = load_test_split()

    print(f"Test split: {len(examples)} examples, device={device}\n")

    results = []

    for cond in CONDITIONS:
        checkpoint = ROOT / cond["checkpoint"] if cond["checkpoint"] else None

        if checkpoint and not checkpoint.exists():
            print(f"[#{cond['id']}] {cond['name']}: checkpoint not found, skipping\n")
            continue

        print(f"[#{cond['id']}] {cond['name']}")
        tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
        model = AutoModelForCausalLM.from_pretrained(cfg["student_model"], torch_dtype=torch.bfloat16)

        if checkpoint:
            model = PeftModel.from_pretrained(model, str(checkpoint))

        model = model.to(device)

        predictions = run_inference(model, tokenizer, examples, max_new_tokens, device)
        references = [ex["answer"] for ex in examples]

        em = exact_match(predictions, references)
        avg_len = sum(len(tokenizer.encode(p)) for p in predictions) / len(predictions)

        results.append({**cond, "exact_match": em, "avg_len": avg_len})

        out_path = ROOT / "artifacts" / "predictions" / f"condition_{cond['id']}.jsonl"
        save_predictions(predictions, examples, out_path)
        print(f"  exact_match={em:.4f}  avg_len={avg_len:.1f}  → {out_path}\n")

        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()

    print("=" * 54)
    print(f"{'#':<4} {'Condition':<22} {'Exact match':>12} {'Avg len':>10}")
    print("-" * 54)
    for r in results:
        print(f"{r['id']:<4} {r['name']:<22} {r['exact_match']:>11.4f} {r['avg_len']:>9.1f}")
    print("=" * 54)


if __name__ == "__main__":
    main()
