"""Chord-event extraction and continuous alignment targets."""

from __future__ import annotations

import torch

from .types import Chord


class ChordOnsetEvents:
    """Extract each chord initializer onset without exposing note timing tokens."""

    def times(self, chords: tuple[Chord, ...]) -> torch.Tensor:
        return torch.tensor(
            [chord.onset for chord in chords], dtype=torch.float64
        )


class TriangularSoftener:
    """Make radius-limited triangular event targets."""

    def __init__(self, radius_seconds: float = 0.2):
        self.radius_seconds = float(radius_seconds)

    def soften(
        self, frame_times: torch.Tensor, event_times: torch.Tensor
    ) -> torch.Tensor:
        distance = torch.abs(frame_times[:, None] - event_times[None, :])
        return (1.0 - distance / self.radius_seconds).clamp(0.0, 1.0)


class TargetGenerator:
    """Generate a padded frame-by-chord target on the global frame clock."""

    def __init__(
        self,
        event_generator: ChordOnsetEvents,
        softener: TriangularSoftener,
    ):
        self.event_generator = event_generator
        self.softener = softener

    def generate(
        self,
        chords: tuple[Chord, ...],
        *,
        frame_start: int,
        segment_frames: int,
        fps: float,
        chord_count: int,
    ) -> torch.Tensor:
        # Frame indices remain global so every window uses the song's time clock.
        frame_times = (
            torch.arange(
                frame_start,
                frame_start + segment_frames,
                dtype=torch.float64,
            )
            / float(fps)
        )
        values = self.softener.soften(
            frame_times, self.event_generator.times(chords)
        )
        target = torch.zeros((segment_frames, chord_count), dtype=torch.float32)
        target[:, : len(chords)] = values.to(torch.float32)
        return target
