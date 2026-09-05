"""Load the compact NPZ contract shared by test and realtime inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InferenceArtifact:
    """Arrays needed by both renderers.

    Probability and target have shape ``[frames, chord_slots]``. The two
    one-dimensional masks identify real frames and populated chord slots.
    """

    probability: np.ndarray
    target: np.ndarray
    frame_mask: np.ndarray
    chord_mask: np.ndarray
    chord_pitches: np.ndarray
    fps: float
    record_id: str
    frame_start: int

    @property
    def support(self) -> np.ndarray:
        """Return ``[frames, chord_slots]`` entries represented by this slice."""

        return self.frame_mask[:, None] & self.chord_mask[None, :]


def load_artifact(input_path: str | Path) -> InferenceArtifact:
    """Load a trusted inference NPZ, copying arrays before closing the archive."""

    path = Path(input_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        return InferenceArtifact(
            probability=payload["probability"].copy(),
            target=payload["target"].copy(),
            frame_mask=payload["frame_mask"].copy(),
            chord_mask=payload["chord_mask"].copy(),
            chord_pitches=payload["chord_pitches"].copy(),
            fps=float(payload["fps"]),
            record_id=str(payload["record_id"]),
            frame_start=int(payload["frame_start"]),
        )
