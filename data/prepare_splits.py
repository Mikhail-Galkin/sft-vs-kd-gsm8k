"""
[D] Download GSM8K and create fixed experiment splits.

Splits (from the original 7473-example train set + 1319-example test set):
  train          [0:1000]    — student training (SFT, logit KD)
  train_unlabeled[1000:2000] — teacher response generation (GT answers ignored)
  val            [2000:2473] — validation / early stopping
  test           [0:1319]    — final evaluation (hold-out until all runs done)

Run from project root: python data/prepare_splits.py
"""

import json
from pathlib import Path

from datasets import load_dataset

SPLITS_DIR = Path(__file__).parent / "splits"

TRAIN_RANGES = {
    "train": (0, 1000),
    "train_unlabeled": (1000, 2000),
    "val": (2000, 2473),
}


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records)} examples")


def main() -> None:
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]   # 7473 examples
    test_ds = ds["test"]     # 1319 examples

    print(f"Source sizes — train: {len(train_ds)}, test: {len(test_ds)}")
    print(f"\nSaving splits to {SPLITS_DIR}/")

    for split_name, (start, end) in TRAIN_RANGES.items():
        records = [
            {"question": ex["question"], "answer": ex["answer"]}
            for ex in train_ds.select(range(start, end))
        ]
        save_jsonl(records, SPLITS_DIR / f"{split_name}.jsonl")

    test_records = [
        {"question": ex["question"], "answer": ex["answer"]}
        for ex in test_ds
    ]
    save_jsonl(test_records, SPLITS_DIR / "test.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
