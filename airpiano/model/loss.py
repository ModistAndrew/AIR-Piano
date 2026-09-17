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


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    chord_mask: torch.Tensor,
) -> torch.Tensor:
    """Average categorical cross-entropy over real frames and chords."""

    supervised_frames = frame_mask & target.any(dim=-1)
    if not bool(supervised_frames.any()):
        raise ValueError(
            "masked_cross_entropy_loss requires at least one target frame"
        )
    masked_logits = logits.to(torch.float32).masked_fill(
        ~chord_mask.unsqueeze(1), -torch.inf
    )
    target_indices = target.argmax(dim=-1)
    return F.cross_entropy(
        masked_logits[supervised_frames], target_indices[supervised_frames]
    )


def masked_alignment_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    chord_mask: torch.Tensor,
    alignment_loss: str,
) -> torch.Tensor:
    """Apply the configured alignment loss to unpadded positions."""

    if alignment_loss == "bce":
        return masked_bce_loss(logits, target, frame_mask, chord_mask)
    if alignment_loss == "cross_entropy":
        return masked_cross_entropy_loss(
            logits, target, frame_mask, chord_mask
        )
    raise ValueError(f"unsupported alignment loss: {alignment_loss}")


def alignment_loss_count(
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    chord_mask: torch.Tensor,
    alignment_loss: str,
) -> int:
    """Return the number of units represented by a mean alignment loss."""

    if alignment_loss == "bce":
        return int(
            (frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)).sum()
        )
    if alignment_loss == "cross_entropy":
        return int((frame_mask & target.any(dim=-1)).sum())
    raise ValueError(f"unsupported alignment loss: {alignment_loss}")


def alignment_probabilities(
    logits: torch.Tensor,
    chord_mask: torch.Tensor,
    alignment_loss: str,
) -> torch.Tensor:
    """Convert logits to probabilities for the configured objective."""

    logits = logits.to(torch.float32)
    if alignment_loss == "bce":
        return torch.sigmoid(logits)
    if alignment_loss == "cross_entropy":
        masked_logits = logits.masked_fill(
            ~chord_mask.unsqueeze(1), -torch.inf
        )
        return torch.softmax(masked_logits, dim=-1).masked_fill(
            ~chord_mask.unsqueeze(1), 0.0
        )
    raise ValueError(f"unsupported alignment loss: {alignment_loss}")


def alignment_probability_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    alignment_loss: str,
) -> tuple[torch.Tensor, int]:
    """Return loss and aggregation weight for committed probabilities."""

    probability = probability.to(torch.float32)
    target = target.to(torch.float32)
    if alignment_loss == "bce":
        weight = probability.numel()
        loss = (
            F.binary_cross_entropy(probability, target)
            if weight
            else probability.new_tensor(float("nan"))
        )
        return loss, weight
    if alignment_loss == "cross_entropy":
        supervised_frames = target.any(dim=-1)
        weight = int(supervised_frames.sum())
        loss = (
            F.nll_loss(
                probability.clamp_min(
                    torch.finfo(probability.dtype).tiny
                ).log()[supervised_frames],
                target.argmax(dim=-1)[supervised_frames],
            )
            if weight
            else probability.new_tensor(float("nan"))
        )
        return loss, weight
    raise ValueError(f"unsupported alignment loss: {alignment_loss}")


__all__ = [
    "alignment_loss_count",
    "alignment_probability_loss",
    "alignment_probabilities",
    "masked_alignment_loss",
    "masked_bce_loss",
    "masked_cross_entropy_loss",
]
