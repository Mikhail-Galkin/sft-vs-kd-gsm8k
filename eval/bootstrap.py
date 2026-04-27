"""
[#0][#1][#2][#3][#4] Paired bootstrap significance test on test predictions.

Reads artifacts/predictions/condition_{i}.jsonl, resamples indices with
replacement, and reports:
  - per-condition exact_match with 95% CI
  - pairwise comparison: P(condition_A > condition_B)

Usage:
  python eval/bootstrap.py
  python eval/bootstrap.py --n-bootstrap 50000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONDITIONS = [
    {"id": 0, "name": "Baseline"},
    {"id": 1, "name": "SFT"},
    {"id": 2, "name": "Response distill"},
    {"id": 3, "name": "Logit KD (GT)"},
    {"id": 4, "name": "Logit KD (synthetic)"},
]


def load_correctness(condition_id: int) -> np.ndarray:
    path = ROOT / "artifacts" / "predictions" / f"condition_{condition_id}.jsonl"
    if not path.exists():
        return None
    correct = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            correct.append(int(json.loads(line)["correct"]))
    return np.array(correct, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load correctness vectors
    data = {}
    for cond in CONDITIONS:
        c = load_correctness(cond["id"])
        if c is not None:
            data[cond["id"]] = {"name": cond["name"], "correct": c}
        else:
            print(f"[#{cond['id']}] {cond['name']}: predictions file not found, skipping")

    n_examples = len(next(iter(data.values()))["correct"])
    print(f"\nTest examples: {n_examples}")
    print(f"Bootstrap iterations: {args.n_bootstrap}\n")

    # Sample indices once per iteration, use for all conditions (paired bootstrap)
    boot_scores = {cid: np.empty(args.n_bootstrap) for cid in data}
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n_examples, size=n_examples)
        for cid, d in data.items():
            boot_scores[cid][b] = d["correct"][idx].mean()

    # Per-condition CIs
    print("=" * 64)
    print(f"{'#':<3} {'Condition':<22} {'EM':>8} {'95% CI':>20}")
    print("-" * 64)
    for cid, d in data.items():
        scores = boot_scores[cid]
        em = d["correct"].mean()
        ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
        print(f"{cid:<3} {d['name']:<22} {em:>8.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    print("=" * 64)

    # Pairwise: P(A > B) — fraction of bootstrap iterations where A beats B
    print(f"\n{'Pairwise P(row > col):':<24}")
    ids = list(data.keys())
    header = " " * 24 + "  ".join(f"#{cid:>3}" for cid in ids)
    print(header)
    for cid_a in ids:
        row = [f"#{cid_a} {data[cid_a]['name']:<19}"]
        for cid_b in ids:
            if cid_a == cid_b:
                row.append("  -  ")
            else:
                p = (boot_scores[cid_a] > boot_scores[cid_b]).mean()
                row.append(f"{p:.3f}")
        print("  ".join(row))

    # Highlight significant comparisons (P > 0.95 or < 0.05)
    print("\nSignificant pairs (P > 0.95):")
    for cid_a in ids:
        for cid_b in ids:
            if cid_a == cid_b:
                continue
            p = (boot_scores[cid_a] > boot_scores[cid_b]).mean()
            if p > 0.95:
                a = data[cid_a]["name"]
                b = data[cid_b]["name"]
                em_a = data[cid_a]["correct"].mean()
                em_b = data[cid_b]["correct"].mean()
                diff = em_a - em_b
                print(f"  #{cid_a} {a} > #{cid_b} {b}:  Δ={diff:+.4f}, P={p:.3f}")


if __name__ == "__main__":
    main()
