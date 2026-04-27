"""
[~] Run inference and evaluate a model checkpoint on a GSM8K split.

Usage (from project root):
  # Condition #0: baseline (no training)
  python eval/run_eval.py --split test

  # After training:
  python eval/run_eval.py --split test --checkpoint checkpoints/sft

Options:
  --split        val | test  (default: test)
  --checkpoint   Path to LoRA adapter dir, or omit for baseline
  --output       Optional path to save per-example predictions as JSONL
  --limit        Evaluate on first N examples (for quick sanity checks)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import exact_match, extract_answer  # noqa: E402 — after sys.path


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
        "At the end of your solution write the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for ex in tqdm(examples, desc="Inference"):
            prompt = make_prompt(ex["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            predictions.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    max_new_tokens = cfg["inference"]["max_new_tokens"]

    examples = load_split(args.split)
    if args.limit:
        examples = examples[: args.limit]

    print(f"Evaluating on '{args.split}' ({len(examples)} examples), device={device}")
    print(f"Checkpoint: {args.checkpoint or 'none (baseline)'}\n")

    model_name = cfg["student_model"]
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if args.checkpoint:
        print(f"Loading LoRA adapter from {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)

    model = model.to(device)

    predictions = run_inference(model, tokenizer, examples, max_new_tokens, device)
    references = [ex["answer"] for ex in examples]

    em = exact_match(predictions, references)
    avg_len = sum(len(tokenizer.encode(p)) for p in predictions) / len(predictions)

    print(f"\n{'='*40}")
    print(f"Split:        {args.split}")
    print(f"Checkpoint:   {args.checkpoint or 'baseline'}")
    print(f"Exact match:  {em:.4f}  ({int(em * len(examples))}/{len(examples)})")
    print(f"Avg resp len: {avg_len:.1f} tokens")
    print(f"{'='*40}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex, pred in zip(examples, predictions):
                f.write(json.dumps({
                    "question": ex["question"],
                    "reference": ex["answer"],
                    "prediction": pred,
                    "correct": extract_answer(pred) == extract_answer(ex["answer"]),
                }, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
