"""Alignment training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    chord_mask: torch.Tensor,
) -> torch.Tensor:
    """Average BCE-with-logits over real frame/chord cells only."""

    valid = frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)
    if not bool(valid.any()):
        raise ValueError("masked_bce_loss requires at least one valid target cell")
    return F.binary_cross_entropy_with_logits(
        logits[valid].to(torch.float32),
        target[valid].to(torch.float32),
    )


def alignment_loss_count(
    frame_mask: torch.Tensor, chord_mask: torch.Tensor
) -> int:
    """Return the number of real frame/chord cells in a mean BCE loss."""

    return int((frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)).sum())


def alignment_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert independent onset logits to float32 probabilities."""

    return torch.sigmoid(logits.to(torch.float32))


def alignment_probability_loss(
    probability: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return BCE and its cell count for committed probabilities."""

    probability = probability.to(torch.float32)
    target = target.to(torch.float32)
    weight = probability.numel()
    loss = (
        F.binary_cross_entropy(probability, target)
        if weight
        else probability.new_tensor(float("nan"))
    )
    return loss, weight


__all__ = [
    "alignment_loss_count",
    "alignment_probability_loss",
    "alignment_probabilities",
    "masked_bce_loss",
]
