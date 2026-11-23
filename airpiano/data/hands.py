"""Canonical hand-cache loading and configurable coordinate transforms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .types import HandSequence
from ..utils import resolve_path


CANONICAL_HAND_SCHEMA = "airpiano-wilor-canonical-v1"


class CanonicalNpzReader:
    """Read the trusted canonical cache produced by WiLoR preprocessing."""

    def __init__(self, representation: str = "wrist_relative") -> None:
        self.representation = representation

    def read(self, path: str | Path) -> HandSequence:
        from ..preprocessing.canonical import validate_canonical_representation

        with np.load(Path(path), allow_pickle=False) as payload:
            validate_canonical_representation(payload, self.representation)
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
    """Apply offsets/scale and optional fixed wrist-XY centering; mask absent hands."""

    def __init__(
        self, offset: list[float], scale: list[float],
        wrist_mean_path: str | None = None, center_wrist_xy: bool = False,
        training_split: str | None = None,
    ) -> None:
        self.offset = tuple(offset)
        self.scale = tuple(scale)
        if wrist_mean_path is not None:
            stats = json.loads(resolve_path(wrist_mean_path).read_text())
            if training_split is not None:
                split_hash = hashlib.sha256(resolve_path(training_split).read_bytes()).hexdigest()
                if stats["training_split_sha256"] != split_hash:
                    raise ValueError("wrist mean must be fitted on the configured training split")
            mean = np.asarray(stats["mean_xy"], dtype=np.float64)
            if mean.shape != (2, 2) or not np.isfinite(mean).all():
                raise ValueError("wrist mean must contain finite left/right XY values")
            offsets = np.broadcast_to(np.asarray(self.offset, dtype=np.float64), (2, 21, 3)).copy()
            offsets[:, 1:, :2] += mean[:, None, :]
            if center_wrist_xy:
                offsets[:, 0, :2] += mean
            self.offset = offsets.tolist()

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
