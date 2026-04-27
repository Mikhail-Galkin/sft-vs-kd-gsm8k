"""
[~] Evaluation metrics for GSM8K.

exact_match: extract the number after '####' in both prediction and reference,
compare as strings (after stripping commas and whitespace).
"""

import re


def extract_answer(text: str) -> str | None:
    """Return the numeric answer after the '####' marker, or None."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions whose extracted answer equals the reference answer."""
    if not predictions:
        return 0.0
    correct = sum(
        extract_answer(pred) == extract_answer(ref)
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)
