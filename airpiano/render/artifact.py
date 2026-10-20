"""Load the compact NPZ contract shared by inference stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils import resolve_path


@dataclass(frozen=True)
class InferenceArtifact:
    """Arrays needed by both renderers.

    Probability and target have shape ``[frames, chord_slots]``. The two
    one-dimensional masks identify real frames and populated chord slots, and
    chord IDs map populated slots back to the source score.
    """

    probability: np.ndarray
    target: np.ndarray
    frame_mask: np.ndarray
    chord_mask: np.ndarray
    chord_pitches: np.ndarray
    chord_ids: np.ndarray
    fps: float
    record_id: str
    frame_start: int
    release_time: np.ndarray | None = None
    predicted_onset: np.ndarray | None = None
    release_frame_index: np.ndarray | None = None
    release_frame_time: np.ndarray | None = None
    note_duration: float | None = None
    note_velocity: int | None = None
    onset_prediction_offset: float | None = None

    @property
    def support(self) -> np.ndarray:
        """Return ``[frames, chord_slots]`` entries represented by this slice."""

        return self.frame_mask[:, None] & self.chord_mask[None, :]


def load_artifact(
    input_path: str | Path, *, require_releases: bool = False
) -> InferenceArtifact:
    """Load arrays, requiring saved real-time metadata when requested.

    Missing release fields raise KeyError; window heatmaps use the default.
    """

    path = resolve_path(input_path)
    with np.load(path, allow_pickle=False) as payload:
        release_metadata = {}
        if require_releases or "release_time" in payload:
            release_metadata = {
                "release_time": payload["release_time"].copy(),
                "predicted_onset": payload["predicted_onset"].copy(),
                "release_frame_index": payload["release_frame_index"].copy(),
                "release_frame_time": payload["release_frame_time"].copy(),
                "note_duration": float(payload["note_duration"]),
                "note_velocity": int(payload["note_velocity"]),
                "onset_prediction_offset": float(
                    payload["onset_prediction_offset"]
                ),
            }
        return InferenceArtifact(
            probability=payload["probability"].copy(),
            target=payload["target"].copy(),
            frame_mask=payload["frame_mask"].copy(),
            chord_mask=payload["chord_mask"].copy(),
            chord_pitches=payload["chord_pitches"].copy(),
            chord_ids=payload["chord_ids"].copy(),
            fps=float(payload["fps"]),
            record_id=str(payload["record_id"]),
            frame_start=int(payload["frame_start"]),
            **release_metadata,
        )
