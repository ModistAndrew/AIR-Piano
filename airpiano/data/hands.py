"""Canonical hand-cache loading and configurable coordinate transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .types import HandSequence


CANONICAL_HAND_SCHEMA = "airpiano-wilor-canonical-v1"


class CanonicalNpzReader:
    """Read the trusted canonical cache produced by WiLoR preprocessing."""

    def read(self, path: str | Path) -> HandSequence:
        with np.load(Path(path), allow_pickle=False) as payload:
            # Copy before closing the archive so the in-memory song owns its data.
            pose = np.array(payload["pose"], copy=True)
            presence = np.array(payload["presence"], copy=True)
            fps = float(payload["fps"])
        return HandSequence(
            pose=torch.from_numpy(pose),
            presence=torch.from_numpy(presence),
            fps=fps,
        )


class OffsetScaleTransform:
    """Apply per-axis offset and scale while keeping absent hands at zero."""

    def __init__(self, offset: list[float], scale: list[float]) -> None:
        self.offset = tuple(offset)
        self.scale = tuple(scale)

    def __call__(self, pose: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        offset = pose.new_tensor(self.offset)
        scale = pose.new_tensor(self.scale)
        transformed = (pose - offset) * scale
        return torch.where(
            presence[..., None, None], transformed, torch.zeros_like(transformed)
        )


class HandLoader:
    """Load canonical coordinates and apply one replaceable postprocessor."""

    def __init__(self, reader: CanonicalNpzReader, transform: OffsetScaleTransform):
        self.reader = reader
        self.transform = transform

    def load(self, path: str | Path) -> HandSequence:
        sequence = self.reader.read(path)
        return HandSequence(
            pose=self.transform(sequence.pose, sequence.presence),
            presence=sequence.presence,
            fps=sequence.fps,
        )
