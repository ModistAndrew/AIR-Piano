"""Chord-event extraction and frame-to-chord alignment targets."""

from __future__ import annotations

import torch

from .types import Chord


class ChordOnsetEvents:
    """Extract each chord's mean physical onset time."""

    def times(self, chords: tuple[Chord, ...]) -> torch.Tensor:
        return torch.tensor(
            [chord.time for chord in chords],
            dtype=torch.float64,
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
    """Generate triangular onset targets on the global frame clock."""

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
        frame_count: int,
        fps: float,
    ) -> torch.Tensor:
        # Frame indices remain global so every window uses the song's time clock.
        frame_times = (
            torch.arange(
                frame_start,
                frame_start + frame_count,
                dtype=torch.float64,
            )
            / float(fps)
        )
        event_times = self.event_generator.times(chords)
        return self.softener.soften(frame_times, event_times).to(torch.float32)
