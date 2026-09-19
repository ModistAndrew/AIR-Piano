"""Frame-based and chord-token hand/score slicing."""

from __future__ import annotations

import bisect
import math

import torch

from .midi import DURATION_BINS
from .types import Chord, Song


PITCH_FEATURES = 128


class FutureChordSlicer:
    """Build frame windows or fixed-width chord-token windows."""

    def __init__(
        self,
        fps: float = 30.0,
        segment_frames: int = 210,
        stride_frames: int = 105,
        chord_count: int = 96,
        mode: str = "frame",
        end_padding_seconds: float = 0.0,
        hand_context_chords: int = 0,
        crop_before_first_chord: bool = False,
        crop_to_chord_bounds: bool = False,
        max_zero_hand_fraction: float | None = None,
    ):
        if mode not in {"frame", "chord"}:
            raise ValueError(f"unsupported slicing mode: {mode}")
        if (
            max_zero_hand_fraction is not None
            and not 0.0 <= max_zero_hand_fraction <= 1.0
        ):
            raise ValueError(
                "max_zero_hand_fraction must be between zero and one"
            )
        self.mode = mode
        self.fps = float(fps)
        self.segment_frames = int(segment_frames)
        self.stride_frames = int(stride_frames)
        self.chord_count = int(chord_count)
        self.end_padding_seconds = float(end_padding_seconds)
        self.hand_context_chords = int(hand_context_chords)
        self.crop_before_first_chord = bool(crop_before_first_chord)
        self.crop_to_chord_bounds = bool(crop_to_chord_bounds)
        self.max_zero_hand_fraction = (
            None
            if max_zero_hand_fraction is None
            else float(max_zero_hand_fraction)
        )

    def starts(self, item_count: int) -> tuple[int, ...]:
        """Return regular starts plus the exact right-aligned final start."""

        if item_count <= self.segment_frames:
            return (0,)
        final_start = item_count - self.segment_frames
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

    def chord_frame_bounds(
        self,
        song: Song,
        chord_positions: tuple[int, ...],
    ) -> tuple[int, int]:
        """Map a chord-token window to its clamped hand-frame interval."""

        first_position = chord_positions[0]
        last_position = chord_positions[-1]
        frame_count = len(song.hands)
        if self.crop_to_chord_bounds:
            frame_start = math.ceil(
                song.chords[first_position].onset * self.fps
            )
            frame_stop = math.ceil(
                max(
                    note.offset
                    for note in song.chords[last_position].notes
                )
                * self.fps
            )
            return (
                max(0, min(frame_start, frame_count)),
                max(0, min(frame_stop, frame_count)),
            )

        context_start_position = max(
            0, first_position - self.hand_context_chords
        )
        context_stop_position = min(
            len(song.chords) - 1,
            last_position + self.hand_context_chords,
        )
        if self.crop_before_first_chord:
            frame_start = math.ceil(
                song.chords[first_position].onset * self.fps
            )
        else:
            frame_start = (
                0
                if first_position == 0
                else math.floor(
                    song.chords[context_start_position].onset * self.fps
                )
            )
        frame_stop = (
            frame_count
            if last_position == len(song.chords) - 1
            else math.ceil(
                (
                    song.chords[context_stop_position].onset
                    + self.end_padding_seconds
                )
                * self.fps
            )
            + 1
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
        """Return whether a hand interval passes the optional absence gate."""

        if self.max_zero_hand_fraction is None:
            return True
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
        if self.mode == "frame":
            for frame_start in self.starts(len(song.hands)):
                positions = self.candidate_positions(song.chords, frame_start)
                frame_stop = min(
                    frame_start + self.segment_frames,
                    len(song.hands),
                )
                if (
                    positions
                    and frame_stop > frame_start
                    and self._accepts_hand_window(song, frame_start, frame_stop)
                ):
                    windows.append((frame_start, frame_stop, positions))
            return tuple(windows)

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

    def slice(
        self,
        song: Song,
        frame_start: int,
        frame_stop: int,
        chord_positions: tuple[int, ...],
    ) -> dict[str, object]:
        selected_chords = tuple(
            song.chords[index] for index in chord_positions
        )
        chord_values = torch.zeros(
            (len(selected_chords), PITCH_FEATURES + DURATION_BINS),
            dtype=torch.float32,
        )
        for local_index, chord in enumerate(selected_chords):
            for note in chord.notes:
                # Duplicate physical notes intentionally collapse in multi-hot space.
                chord_values[local_index, note.pitch] = 1.0
            chord_values[local_index, PITCH_FEATURES:] = chord.quantized_dur

        return {
            "hands": song.hands[frame_start:frame_stop],
            "hand_presence": song.presence[frame_start:frame_stop],
            "chords": chord_values,
            "chord_ids": torch.tensor(
                [chord.id for chord in selected_chords], dtype=torch.long
            ),
            "selected_chords": selected_chords,
        }
