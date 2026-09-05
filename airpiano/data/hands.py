"""Canonical hand-cache loading and baseline coordinate transforms."""

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


class CenteredXYZ10Transform:
    """Center full-image XY, scale wrist-relative Z, and zero absent hands."""

    def __call__(self, pose: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        offset = pose.new_tensor((0.5, 0.5, 0.0))
        scale = pose.new_tensor((1.0, 1.0, 10.0))
        transformed = (pose - offset) * scale
        return torch.where(
            presence[..., None, None], transformed, torch.zeros_like(transformed)
        )


class HandLoader:
    """Load canonical coordinates and apply one replaceable postprocessor."""

    def __init__(self, reader: CanonicalNpzReader, transform: CenteredXYZ10Transform):
        self.reader = reader
        self.transform = transform

    def load(self, path: str | Path) -> HandSequence:
        sequence = self.reader.read(path)
        return HandSequence(
            pose=self.transform(sequence.pose, sequence.presence),
            presence=sequence.presence,
            fps=sequence.fps,
        )
