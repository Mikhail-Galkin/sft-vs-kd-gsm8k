"""
[#1][#2][#3][#4] Training script for all conditions.

Usage (from project root):
  python training/train.py --condition sft
  python training/train.py --condition response_distill
  python training/train.py --condition logit_kd_gt
  python training/train.py --condition logit_kd_synthetic
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import build_dataset
from training.loss import ce_loss, logit_kd_loss
from eval.metrics import exact_match, extract_answer


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: str):
    model_name = cfg["student_model"]
    lora_cfg = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    return model, tokenizer


# ── Val evaluation ────────────────────────────────────────────────────────────

def make_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "At the end write the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


@torch.no_grad()
def evaluate(model, tokenizer, val_examples: list[dict], device: str, max_new_tokens: int = 256) -> dict:
    model.eval()
    predictions = []
    for ex in tqdm(val_examples, desc="Val inference", leave=False):
        prompt = make_prompt(ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))

    references = [ex["answer"] for ex in val_examples]
    em = exact_match(predictions, references)
    model.train()
    return {"val/exact_match": em}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(condition: str, cfg: dict, device: str) -> None:
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    kd_cfg = cfg["logit_kd"]
    has_logits = condition.startswith("logit_kd")

    checkpoint_dir = ROOT / cfg["paths"]["checkpoints_dir"] / condition
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    model, tokenizer = build_model(cfg, device)

    train_dataset, collate_fn = build_dataset(condition, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_examples = load_jsonl(ROOT / "data" / "splits" / "val.jsonl")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    total_steps = (len(train_loader) // train_cfg["grad_accum"]) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb.init(
        project="sft-vs-distillation",
        name=condition,
        config={**cfg, "condition": condition},
    )

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = 0
    best_em = -1.0
    patience_count = 0
    best_checkpoint_dir = checkpoint_dir / "best"

    model.train()
    optimizer.zero_grad()

    for epoch in range(train_cfg["epochs"]):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, seq_len, vocab_size]

            if has_logits:
                teacher_logits = batch["teacher_logits"].to(device)
                loss, ce, kl = logit_kd_loss(
                    logits, teacher_logits, labels,
                    alpha=kd_cfg["alpha"],
                    temperature=kd_cfg["temperature"],
                )
                wandb.log({"train/ce_loss": ce.item(), "train/kl_loss": kl.item()}, step=global_step)
            else:
                loss = ce_loss(logits, labels)

            # Gradient accumulation
            (loss / train_cfg["grad_accum"]).backward()

            if (batch_idx + 1) % train_cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

                # ── Evaluation ────────────────────────────────────────────────
                if global_step % train_cfg["eval_steps"] == 0:
                    metrics = evaluate(model, tokenizer, val_examples, device)
                    wandb.log(metrics, step=global_step)

                    em = metrics["val/exact_match"]
                    print(f"  step {global_step}: val/exact_match={em:.4f}")

                    if em > best_em:
                        best_em = em
                        patience_count = 0
                        model.save_pretrained(str(best_checkpoint_dir))
                        tokenizer.save_pretrained(str(best_checkpoint_dir))
                    else:
                        patience_count += 1
                        if patience_count >= train_cfg["early_stopping_patience"]:
                            print(f"Early stopping at step {global_step} (patience={patience_count})")
                            wandb.finish()
                            return

    model.save_pretrained(str(best_checkpoint_dir))
    tokenizer.save_pretrained(str(best_checkpoint_dir))

    print(f"\nDone. Best val/exact_match={best_em:.4f} → {best_checkpoint_dir}")
    wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=["sft", "response_distill", "logit_kd_gt", "logit_kd_synthetic"],
    )
    args = parser.parse_args()

    cfg = load_config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    train(args.condition, cfg, device)


if __name__ == "__main__":
    main()
