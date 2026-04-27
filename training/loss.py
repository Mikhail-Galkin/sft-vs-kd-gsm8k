"""
[#1][#2][#3][#4] Loss functions.

ce_loss       — standard cross-entropy on response tokens (#1, #2)
logit_kd_loss — α·CE + (1-α)·KL·T² on response tokens (#3, #4)
"""

import torch
import torch.nn.functional as F


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on response tokens.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        labels:         [batch, seq_len], prompt positions are -100
    """
    shift_logits = student_logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """α·CE + (1-α)·KL·T² on response tokens.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        labels:         [batch, seq_len], prompt positions are -100
        alpha:          weight of CE loss
        temperature:    distillation temperature

    Returns:
        total_loss, ce_component, kl_component
    """
    ce = ce_loss(student_logits, labels)

    shift_student = student_logits[:, :-1].contiguous()   # [B, S-1, V]
    shift_teacher = teacher_logits[:, :-1].contiguous()   # [B, S-1, V]
    shift_labels  = labels[:, 1:].contiguous()            # [B, S-1]

    # Align vocab sizes (base vs instruct models differ by 128 special tokens)
    min_vocab = min(shift_student.shape[-1], shift_teacher.shape[-1])
    shift_student = shift_student[..., :min_vocab]
    shift_teacher = shift_teacher[..., :min_vocab]

    mask = (shift_labels != -100).float()

    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs     = F.softmax(shift_teacher / temperature, dim=-1)

    # KL per token, summed over vocab, masked to response positions only
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)
    kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

    total = alpha * ce + (1 - alpha) * kl * (temperature ** 2)
    return total, ce, kl
