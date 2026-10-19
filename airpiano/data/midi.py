"""Physical-note MIDI parsing and duration-bearing chord construction."""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import mido
import torch
import torch.nn.functional as F

from .types import Chord, EventSequence, MidiSequence, Note, NoteEvent


DURATION_BINS = 200
DURATION_COEFFICIENT = 4.0
DURATION_CLIP_SECONDS = 16.0


def quantize_duration_seconds(duration_seconds: torch.Tensor) -> torch.Tensor:
    """Return MIDI2ScoreTransformer one-hots for duration-like values."""

    denominator = math.log(
        DURATION_COEFFICIENT * DURATION_CLIP_SECONDS + 1.0
    )
    bucket_indices = torch.floor(
        DURATION_BINS
        * torch.log(DURATION_COEFFICIENT * duration_seconds + 1.0)
        / denominator
    ).to(torch.long)
    bucket_indices = bucket_indices.clamp(0, DURATION_BINS - 1)
    return F.one_hot(bucket_indices, num_classes=DURATION_BINS).to(torch.float32)


def quantize_duration_ticks(
    duration_ticks: torch.Tensor,
    *,
    num_bins: int = DURATION_BINS,
    minimum_tick: int = 0,
) -> torch.Tensor:
    """Return saturating one-hots for a bounded integer duration interval."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if bool((duration_ticks < 0).any()):
        raise ValueError("duration ticks must be non-negative")
    bounded_ticks = duration_ticks.to(torch.long).clamp(min=minimum_tick)
    bucket_indices = (bounded_ticks - minimum_tick).clamp(
        max=num_bins - 1
    )
    return F.one_hot(bucket_indices, num_classes=num_bins).to(torch.float32)


class MidoNoteReader:
    """Read tempo-aware physical notes from a standard MIDI file.

    ``mido.MidiFile`` iteration merges tracks and converts delta times to
    seconds. Repeated notes on the same channel and pitch are paired FIFO;
    sustain-pedal messages are deliberately ignored. Optional beat-grid
    quantization adds tick coordinates without replacing the seconds clock.
    """

    def __init__(
        self,
        quantization: bool = False,
        subdivision: int = 12,
        allowed_tick_offsets: Sequence[int] | None = None,
        out_of_bounds_mode: str = "extrapolate",
    ) -> None:
        self.quantization = quantization
        self.subdivision = int(subdivision)
        if out_of_bounds_mode not in {"extrapolate", "clamp"}:
            raise ValueError(
                "out_of_bounds_mode must be 'extrapolate' or 'clamp'"
            )
        self.out_of_bounds_mode = out_of_bounds_mode
        if allowed_tick_offsets is None:
            self.allowed_tick_offsets = None
        else:
            offsets = tuple(allowed_tick_offsets)
            if any(not isinstance(offset, int) for offset in offsets):
                raise ValueError("allowed_tick_offsets must contain integers")
            if len(offsets) != len(set(offsets)):
                raise ValueError("allowed_tick_offsets must be unique")
            if 0 not in offsets:
                raise ValueError("allowed_tick_offsets must include 0")
            if any(
                offset < 0 or offset >= self.subdivision
                for offset in offsets
            ):
                raise ValueError(
                    "allowed_tick_offsets must be in [0, subdivision)"
                )
            self.allowed_tick_offsets = tuple(sorted(offsets))

    def _read_ticks(
        self,
        path: str | Path,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows = tuple(
                tuple(float(value) for value in line.split())
                for line in handle
                if line.strip()
            )
        column_counts = {len(row) for row in rows}
        if column_counts not in ({1}, {2}):
            raise ValueError(
                "beat rows must consistently contain one or two numbers"
            )

        offsets = self.allowed_tick_offsets
        if offsets is None:
            offsets = tuple(range(self.subdivision))

        if column_counts == {1}:
            ticks = tuple(
                (row[0], tick)
                for tick, row in enumerate(rows)
                if tick % self.subdivision in offsets
            )
            return (
                tuple(time for time, _ in ticks),
                tuple(tick for _, tick in ticks),
            )

        beats = tuple(row[0] for row in rows)
        tick_times = tuple(
            start + (stop - start) * offset / self.subdivision
            for start, stop in zip(beats, beats[1:])
            for offset in offsets
        ) + (beats[-1],)
        tick_positions = tuple(
            beat_index * self.subdivision + offset
            for beat_index in range(len(beats) - 1)
            for offset in offsets
        ) + ((len(beats) - 1) * self.subdivision,)
        return tick_times, tick_positions

    def _nearest_allowed_tick(self, position: float) -> int:
        offsets = self.allowed_tick_offsets
        if offsets is None:
            offsets = tuple(range(self.subdivision))
        beat_index = math.floor(position / self.subdivision)
        candidates = (
            candidate_beat * self.subdivision + offset
            for candidate_beat in range(beat_index - 1, beat_index + 2)
            for offset in offsets
        )
        return min(candidates, key=lambda tick: (abs(tick - position), -tick))

    def _nearest_tick(
        self,
        time_seconds: float,
        tick_times: tuple[float, ...],
        tick_positions: tuple[int, ...],
    ) -> int:
        if time_seconds <= tick_times[0]:
            if self.out_of_bounds_mode == "clamp":
                return tick_positions[0]
            seconds_per_tick = (
                (tick_times[1] - tick_times[0])
                / (tick_positions[1] - tick_positions[0])
            )
            position = tick_positions[0] + (
                time_seconds - tick_times[0]
            ) / seconds_per_tick
            return self._nearest_allowed_tick(position)
        if time_seconds >= tick_times[-1]:
            if self.out_of_bounds_mode == "clamp":
                return tick_positions[-1]
            seconds_per_tick = (
                (tick_times[-1] - tick_times[-2])
                / (tick_positions[-1] - tick_positions[-2])
            )
            position = (
                tick_positions[-1]
                + (time_seconds - tick_times[-1]) / seconds_per_tick
            )
            return self._nearest_allowed_tick(position)

        later = bisect_left(tick_times, time_seconds)
        earlier = later - 1
        return (
            tick_positions[earlier]
            if time_seconds - tick_times[earlier]
            < tick_times[later] - time_seconds
            else tick_positions[later]
        )

    def read(
        self,
        path: str | Path,
        beat_path: str | Path | None = None,
    ) -> tuple[Note, ...]:
        ticks = self._read_ticks(beat_path) if self.quantization else None

        midi = mido.MidiFile(Path(path))
        active: dict[tuple[int, int], deque[tuple[float, int]]] = defaultdict(deque)
        parsed: list[tuple[float, float, int, int]] = []
        time_seconds = 0.0

        for message in midi:
            time_seconds += float(message.time)
            if message.type not in {"note_on", "note_off"}:
                continue
            pitch = int(message.note)
            channel = int(getattr(message, "channel", 0))
            velocity = int(message.velocity) if message.type == "note_on" else 0
            key = (channel, pitch)
            if message.type == "note_on" and velocity > 0:
                active[key].append((time_seconds, velocity))
                continue
            if not active[key]:
                continue
            onset, onset_velocity = active[key].popleft()
            if time_seconds > onset:
                parsed.append((onset, time_seconds, pitch, onset_velocity))

        parsed.sort(key=lambda note: (note[0], note[2], note[1] - note[0]))
        notes = []
        for note_id, (onset, offset, pitch, velocity) in enumerate(parsed):
            onset_tick = None
            offset_tick = None
            if ticks is not None:
                tick_times, tick_positions = ticks
                onset_tick = self._nearest_tick(
                    onset,
                    tick_times,
                    tick_positions,
                )
                offset_tick = self._nearest_tick(
                    offset,
                    tick_times,
                    tick_positions,
                )
            notes.append(
                Note(
                    id=note_id,
                    onset=onset,
                    offset=offset,
                    pitch=pitch,
                    velocity=velocity,
                    onset_tick=onset_tick,
                    offset_tick=offset_tick,
                )
            )
        return tuple(notes)


class InitializerChordMerger:
    """Group note events and encode seconds or beat-grid tick durations."""

    def __init__(
        self,
        tolerance_seconds: float = 0.05,
        duration_scale_range: Sequence[float] | None = None,
        duration_normalization: str = "none",
        time_mode: str = "first",
        split_notes: bool = False,
        duration_tick_bins: int = DURATION_BINS,
        minimum_duration_ticks: int = 0,
    ):
        if duration_tick_bins <= 0:
            raise ValueError("duration_tick_bins must be positive")
        self.duration_tick_bins = int(duration_tick_bins)
        self.minimum_duration_ticks = int(minimum_duration_ticks)
        self.tolerance_seconds = float(tolerance_seconds)
        if duration_scale_range is None:
            self.duration_scale_range = None
        else:
            minimum, maximum = duration_scale_range
            self.duration_scale_range = (float(minimum), float(maximum))
        if duration_normalization not in {"none", "song_mean"}:
            raise ValueError(
                "duration_normalization must be 'none' or 'song_mean'"
            )
        if (
            duration_normalization == "song_mean"
            and self.duration_scale_range is not None
        ):
            raise ValueError(
                "duration_scale_range cannot be combined with song_mean "
                "duration normalization"
            )
        self.duration_normalization = duration_normalization
        if time_mode not in {"first", "mean"}:
            raise ValueError("time_mode must be 'first' or 'mean'")
        self.time_mode = time_mode
        self.split_notes = bool(split_notes)

    def merge(
        self,
        event_sequence: EventSequence,
        quantization: bool = False,
        *,
        subdivision: int = 12,
    ) -> tuple[Chord, ...]:
        events = event_sequence.events
        groups: list[tuple[NoteEvent, ...]] = []
        start = 0
        while start < len(events):
            if self.split_notes:
                groups.append(events[start : start + 1])
                start += 1
                continue
            initializer = (
                events[start].tick if quantization else events[start].time
            )
            stop = start + 1
            while stop < len(events):
                action_time = (
                    events[stop].tick if quantization else events[stop].time
                )
                if quantization:
                    if action_time != initializer:
                        break
                    stop += 1
                    continue
                roundoff = 8.0 * max(
                    math.ulp(action_time),
                    math.ulp(initializer),
                    math.ulp(self.tolerance_seconds),
                )
                if action_time - initializer > self.tolerance_seconds + roundoff:
                    break
                stop += 1
            groups.append(events[start:stop])
            start = stop

        if not groups:
            return ()

        if self.time_mode == "mean":
            times = [
                sum(event.time for event in group) / len(group)
                for group in groups
            ]
        else:
            times = [group[0].time for group in groups]
        durations: list[float] = []
        duration_ticks: list[int | None] = []
        for index, group in enumerate(groups):
            if index + 1 < len(groups):
                duration = times[index + 1] - times[index]
            elif event_sequence.include_offsets:
                duration = 0.0
            else:
                duration = (
                    max(event.note.offset for event in group) - times[index]
                )
            durations.append(duration)

            ticks = None
            if quantization:
                action_tick = int(group[0].tick)
                if index + 1 < len(groups):
                    next_tick = int(groups[index + 1][0].tick)
                    ticks = next_tick - action_tick
                elif event_sequence.include_offsets:
                    # The final group contains the last note offset, so there
                    # is no following event gap to encode.
                    ticks = 0
                else:
                    ticks = (
                        max(int(event.note.offset_tick) for event in group)
                        - action_tick
                    )
            duration_ticks.append(ticks)

        if self.duration_normalization == "song_mean":
            duration_scale = len(durations) / sum(durations)
        elif self.duration_scale_range is not None:
            duration_scale = random.uniform(*self.duration_scale_range)
        else:
            duration_scale = 1.0

        chords = []
        for index, (group, time, duration, ticks) in enumerate(
            zip(groups, times, durations, duration_ticks)
        ):
            if quantization:
                quantized_dur = quantize_duration_ticks(
                    torch.tensor(ticks),
                    num_bins=self.duration_tick_bins,
                    minimum_tick=self.minimum_duration_ticks,
                )
            else:
                quantized_dur = quantize_duration_seconds(
                    torch.tensor(duration * duration_scale, dtype=torch.float64)
                )
            chords.append(
                Chord(
                    id=index,
                    events=group,
                    onset_notes=tuple(
                        event.note for event in group if not event.is_offset
                    ),
                    offset_notes=tuple(
                        event.note for event in group if event.is_offset
                    ),
                    time=time,
                    duration=duration,
                    duration_ticks=ticks,
                    quantized_dur=quantized_dur,
                    score_subdivision=(
                        int(group[0].tick) * 24.0 / subdivision
                        if quantization
                        else None
                    ),
                )
            )
        return tuple(chords)


class MidiLoader:
    """Compose note parsing and chord construction."""

    def __init__(
        self,
        reader: MidoNoteReader,
        chord_merger: InitializerChordMerger,
        include_offsets: bool = False,
    ):
        self.reader = reader
        self.chord_merger = chord_merger
        self.include_offsets = bool(include_offsets)

    def load(
        self,
        path: str | Path,
        beat_path: str | Path | None = None,
    ) -> MidiSequence:
        if beat_path is None:
            notes = self.reader.read(path)
        else:
            notes = self.reader.read(path, beat_path)
        events = EventSequence.from_notes(notes, self.include_offsets)
        return MidiSequence(
            notes=notes,
            chords=self.chord_merger.merge(
                events,
                quantization=self.reader.quantization,
                subdivision=self.reader.subdivision,
            ),
        )
