"""Chord-event extraction and frame-to-chord alignment targets."""

from __future__ import annotations

import torch

from .types import Chord


class ChordOnsetEvents:
    """Extract each chord's representative action time."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = float(delay_seconds)

    def times(self, chords: tuple[Chord, ...]) -> torch.Tensor:
        return torch.tensor(
            [chord.time + self.delay_seconds for chord in chords],
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
    """Generate onset or last-played targets on the global frame clock."""

    def __init__(
        self,
        event_generator: ChordOnsetEvents,
        softener: TriangularSoftener,
        target_mode: str = "onset",
        velocity_target: bool = False,
    ):
        if target_mode not in {"onset", "last_chord"}:
            raise ValueError(f"unsupported target mode: {target_mode}")
        self.event_generator = event_generator
        self.softener = softener
        self.target_mode = target_mode
        self.velocity_target = bool(velocity_target)

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
        if self.target_mode == "onset":
            target = self.softener.soften(frame_times, event_times).to(
                torch.float32
            )
        else:
            target = torch.zeros(
                (frame_count, len(chords)), dtype=torch.float32
            )
            if chords:
                chord_indices = torch.searchsorted(
                    event_times, frame_times, right=True
                ) - 1
                assigned = chord_indices >= 0
                target[
                    torch.arange(frame_count)[assigned],
                    chord_indices[assigned],
                ] = 1.0

        if self.velocity_target:
            amplitudes = torch.tensor(
                [
                    sum(event.note.velocity for event in chord.events)
                    / (128.0 * len(chord.events))
                    for chord in chords
                ],
                dtype=torch.float32,
            )
            target = target * amplitudes
        return target
