"""Fixed-window hand and future-chord slicing."""

from __future__ import annotations

import bisect

import torch

from .types import Chord, Song


class FutureChordSlicer:
    """Build fixed-width windows using overlap candidates, then successors."""

    def __init__(
        self,
        fps: float = 30.0,
        segment_frames: int = 210,
        stride_frames: int = 105,
        chord_count: int = 96,
    ):
        self.fps = float(fps)
        self.segment_frames = int(segment_frames)
        self.stride_frames = int(stride_frames)
        self.chord_count = int(chord_count)

    def starts(self, frame_count: int) -> tuple[int, ...]:
        """Return regular starts plus the exact right-aligned final start."""

        if frame_count <= self.segment_frames:
            return (0,)
        final_start = frame_count - self.segment_frames
        starts = list(range(0, final_start + 1, self.stride_frames))
        if starts[-1] != final_start:
            starts.append(final_start)
        return tuple(starts)

    def candidate_positions(
        self, chords: tuple[Chord, ...], frame_start: int
    ) -> tuple[int, ...]:
        start_seconds = frame_start / self.fps
        end_seconds = (frame_start + self.segment_frames) / self.fps
        overlaps = [
            index
            for index, chord in enumerate(chords)
            if any(
                note.onset < end_seconds and note.offset > start_seconds
                for note in chord.notes
            )
        ]
        if len(overlaps) > self.chord_count:
            raise ValueError(
                f"window at frame {frame_start} contains {len(overlaps)} "
                f"overlapping chords, exceeding chord_count={self.chord_count}"
            )

        selected = list(overlaps)
        if len(selected) < self.chord_count:
            if overlaps:
                successor_start = overlaps[-1] + 1
            else:
                successor_start = bisect.bisect_left(
                    [chord.onset for chord in chords], end_seconds
                )
            successor_stop = min(
                len(chords), successor_start + self.chord_count - len(selected)
            )
            selected.extend(range(successor_start, successor_stop))
        return tuple(selected)

    def slice(
        self,
        song: Song,
        frame_start: int,
        chord_positions: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        valid_frames = min(self.segment_frames, len(song.hands) - frame_start)
        hands = torch.zeros(
            (self.segment_frames, 2, 21, 3), dtype=torch.float32
        )
        presence = torch.zeros((self.segment_frames, 2), dtype=torch.bool)
        hands[:valid_frames] = song.hands[frame_start : frame_start + valid_frames]
        presence[:valid_frames] = song.presence[
            frame_start : frame_start + valid_frames
        ]
        frame_mask = torch.arange(self.segment_frames) < valid_frames

        positions = (
            self.candidate_positions(song.chords, frame_start)
            if chord_positions is None
            else chord_positions
        )
        selected_chords = tuple(song.chords[index] for index in positions)
        chord_values = torch.zeros((self.chord_count, 128), dtype=torch.float32)
        chord_mask = torch.zeros(self.chord_count, dtype=torch.bool)
        chord_ids = torch.full((self.chord_count,), -1, dtype=torch.long)
        for local_index, chord in enumerate(selected_chords):
            chord_mask[local_index] = True
            chord_ids[local_index] = chord.id
            for note in chord.notes:
                # Duplicate physical notes intentionally collapse in multi-hot space.
                chord_values[local_index, note.pitch] = 1.0

        return {
            "hands": hands,
            "hand_presence": presence,
            "frame_mask": frame_mask,
            "chords": chord_values,
            "chord_mask": chord_mask,
            "chord_ids": chord_ids,
            "selected_chords": selected_chords,
        }
