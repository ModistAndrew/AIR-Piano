"""Chord-window indices and aligned hand-frame bounds."""

from __future__ import annotations

import math

from .types import Song


class FutureChordSlicer:
    """Build fixed-width chord windows cropped to their hand-frame bounds.

    ``segment_frames`` and ``stride_frames`` count chords; the argument names
    are retained for existing callers.
    """

    def __init__(
        self,
        fps: float = 30.0,
        segment_frames: int = 50,
        stride_frames: int = 12,
        max_zero_hand_fraction: float = 0.2,
    ):
        self.fps = float(fps)
        self.segment_frames = int(segment_frames)
        self.stride_frames = int(stride_frames)
        self.max_zero_hand_fraction = float(max_zero_hand_fraction)

    def starts(self, item_count: int) -> tuple[int, ...]:
        """Return regular starts plus the exact right-aligned final start."""

        if item_count <= self.segment_frames:
            return (0,)
        final_start = item_count - self.segment_frames
        starts = list(range(0, final_start + 1, self.stride_frames))
        if starts[-1] != final_start:
            starts.append(final_start)
        return tuple(starts)

    def chord_frame_bounds(
        self,
        song: Song,
        chord_positions: tuple[int, ...],
    ) -> tuple[int, int]:
        """Map a chord window to its clamped hand-frame interval."""

        first_position = chord_positions[0]
        last_position = chord_positions[-1]
        frame_count = len(song.hands)
        frame_start = math.floor(song.chords[first_position].time * self.fps)
        last_chord = song.chords[last_position]
        frame_stop = (
            math.ceil((last_chord.time + last_chord.duration) * self.fps)
            if last_position == len(song.chords) - 1
            else math.ceil(last_chord.time * self.fps) + 1
        )
        return (
            max(0, min(frame_start, frame_count)),
            max(0, min(frame_stop, frame_count)),
        )

    def _accepts_hand_window(
        self,
        song: Song,
        frame_start: int,
        frame_stop: int,
    ) -> bool:
        """Return whether a hand interval passes the absence gate."""

        zero_hand_frames = int(
            (~song.presence[frame_start:frame_stop].any(dim=-1)).sum()
        )
        zero_hand_fraction = zero_hand_frames / (frame_stop - frame_start)
        return zero_hand_fraction <= self.max_zero_hand_fraction

    def windows(
        self, song: Song
    ) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        """Return frame bounds and selected chord positions for one song."""

        windows: list[tuple[int, int, tuple[int, ...]]] = []
        if not song.chords:
            return ()
        for chord_start in self.starts(len(song.chords)):
            chord_stop = min(
                chord_start + self.segment_frames,
                len(song.chords),
            )
            positions = tuple(range(chord_start, chord_stop))
            frame_start, frame_stop = self.chord_frame_bounds(song, positions)
            if frame_stop > frame_start and self._accepts_hand_window(
                song, frame_start, frame_stop
            ):
                windows.append((frame_start, frame_stop, positions))
        return tuple(windows)
