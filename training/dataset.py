"""
[#1][#2][#3][#4] Dataset classes for all training conditions.

SFTDataset      — conditions #1 (GT answers) and #2 (teacher responses)
LogitKDDataset  — conditions #3 (GT logits) and #4 (synthetic logits)

All datasets mask prompt tokens in labels so loss is computed only on response tokens.
LogitKDDataset uses saved input_ids from .npz files to guarantee alignment with teacher logits.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

ROOT = Path(__file__).parent.parent


def make_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "At the end write the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SFTDataset(Dataset):
    """[#1][#2] Standard CE training on target tokens."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        response_key: str,
        max_length: int = 1024,
    ):
        self.data = []
        for ex in examples:
            prompt = make_prompt(ex["question"])
            response = ex[response_key]

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            # Truncate response if sequence exceeds max_length
            max_resp = max_length - len(prompt_ids) - 1  # reserve 1 for eos
            response_ids = response_ids[:max_resp]

            input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
            labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

            self.data.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class LogitKDDataset(Dataset):
    """[#3][#4] KL + CE training with precomputed teacher logits.

    Logits are loaded lazily in __getitem__ to avoid loading all ~60GB into RAM.
    Uses input_ids saved in .npz files to guarantee token-level alignment.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        logits_dir: Path,
        response_key: str,
        max_length: int = 1024,
    ):
        self.logits_dir = logits_dir
        self.max_length = max_length
        self.data = []
        for idx, ex in enumerate(examples):
            npz_path = logits_dir / f"{idx}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing logits file: {npz_path}")

            # Load only input_ids at init time (small), not logits
            input_ids = np.load(npz_path)["input_ids"].tolist()
            input_ids = input_ids[:max_length]

            prompt = make_prompt(ex["question"])
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.data.append({"idx": idx, "input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        # Load logits on demand — only one example at a time in memory
        logits = np.load(self.logits_dir / f"{item['idx']}.npz")["logits"]
        logits = logits[:self.max_length]
        return {**item, "teacher_logits": logits}


def sft_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def logit_kd_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    result = sft_collate_fn(batch, pad_token_id)
    max_len = result["input_ids"].shape[1]
    vocab_size = batch[0]["teacher_logits"].shape[1]
    padded = []
    for x in batch:
        tl = x["teacher_logits"]
        pad = max_len - tl.shape[0]
        if pad > 0:
            tl = np.concatenate([tl, np.zeros((pad, vocab_size), dtype=np.float16)])
        padded.append(tl)
    result["teacher_logits"] = torch.tensor(np.stack(padded), dtype=torch.float32)
    return result


# ── Factory ──────────────────────────────────────────────────────────────────

CONDITION_CONFIG = {
    "sft": {
        "split": "train",
        "response_key": "answer",
        "has_logits": False,
    },
    "response_distill": {
        "split_file": "artifacts/responses/train_unlabeled_responses.jsonl",
        "response_key": "teacher_response",
        "has_logits": False,
    },
    "logit_kd_gt": {
        "split": "train",
        "response_key": "answer",
        "has_logits": True,
        "logits_dir": "artifacts/logits/gt",
    },
    "logit_kd_synthetic": {
        "split_file": "artifacts/responses/train_unlabeled_responses.jsonl",
        "response_key": "teacher_response",
        "has_logits": True,
        "logits_dir": "artifacts/logits/synthetic",
    },
}


def build_dataset(condition: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
    cfg = CONDITION_CONFIG[condition]
    response_key = cfg["response_key"]

    if "split_file" in cfg:
        examples = load_jsonl(ROOT / cfg["split_file"])
    else:
        examples = load_jsonl(ROOT / "data" / "splits" / f"{cfg['split']}.jsonl")

    if cfg["has_logits"]:
        logits_dir = ROOT / cfg["logits_dir"]
        dataset = LogitKDDataset(examples, tokenizer, logits_dir, response_key, max_length)
        collate_fn = partial(logit_kd_collate_fn, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    else:
        dataset = SFTDataset(examples, tokenizer, response_key, max_length)
        collate_fn = partial(sft_collate_fn, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    return dataset, collate_fn
