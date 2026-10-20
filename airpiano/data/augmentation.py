"""Training augmentations on raw alignment windows before tokenization."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import replace

from .types import AlignmentWindow, Chord, Song


class WindowAugmenter:
    """Jitter hand bounds and vary score inputs without changing physical clocks.

    Source song tensors and note records are never modified. Duration factors
    scale raw ticks; only the final tokenizer saturates them to its bin range.
    """

    def __init__(
        self,
        fps: float = 30.0,
        random_hand_shift_seconds: float = 0.0,
        random_pitch_shift_semitones: int = 0,
        random_duration_scale_factors: Sequence[float] | None = None,
    ):
        self.random_hand_shift_seconds = float(random_hand_shift_seconds)
        self.random_pitch_shift_semitones = int(random_pitch_shift_semitones)
        self.random_duration_scale_factors = (
            ()
            if random_duration_scale_factors is None
            else tuple(float(value) for value in random_duration_scale_factors)
        )
        self.max_random_hand_shift_frames = round(
            self.random_hand_shift_seconds * float(fps)
        )

    def _shift_frame_bounds(
        self, frame_start: int, frame_stop: int, frame_count: int
    ) -> tuple[int, int]:
        shift = self.max_random_hand_shift_frames
        if shift == 0:
            return frame_start, frame_stop
        while True:
            shifted_start = max(
                0, min(frame_start + random.randint(-shift, shift), frame_count)
            )
            shifted_stop = max(
                0, min(frame_stop + random.randint(-shift, shift), frame_count)
            )
            if shifted_stop > shifted_start:
                return shifted_start, shifted_stop

    def _shift_pitches(self, chords: tuple[Chord, ...]) -> tuple[Chord, ...]:
        max_shift = self.random_pitch_shift_semitones
        if max_shift == 0:
            return chords
        pitches = [note.pitch for chord in chords for note in chord.onset_notes]
        if not pitches:
            return chords
        minimum_shift = max(-max_shift, -min(pitches))
        maximum_shift = min(max_shift, 127 - max(pitches))
        shift = random.randint(minimum_shift, maximum_shift)
        if shift == 0:
            return chords
        return tuple(
            replace(
                chord,
                onset_notes=tuple(
                    replace(note, pitch=note.pitch + shift)
                    for note in chord.onset_notes
                ),
            )
            for chord in chords
        )

    def _scale_durations(self, chords: tuple[Chord, ...]) -> tuple[Chord, ...]:
        if not self.random_duration_scale_factors:
            return chords
        factor = random.choice(self.random_duration_scale_factors)
        if factor == 1.0:
            return chords
        return tuple(
            replace(chord, duration_ticks=math.ceil(chord.duration_ticks * factor))
            for chord in chords
        )

    def augment(self, window: AlignmentWindow, song: Song) -> AlignmentWindow:
        """Return augmented raw data, reading expanded frame bounds from song."""

        frame_start, frame_stop = self._shift_frame_bounds(
            window.frame_start,
            window.frame_start + len(window.hands),
            len(song.hands),
        )
        chords = self._shift_pitches(window.chords)
        chords = self._scale_durations(chords)
        return AlignmentWindow(
            hands=song.hands[frame_start:frame_stop],
            hand_presence=song.presence[frame_start:frame_stop],
            chords=chords,
            frame_start=frame_start,
        )
