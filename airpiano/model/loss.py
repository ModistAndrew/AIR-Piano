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


__all__ = ["masked_bce_loss"]
