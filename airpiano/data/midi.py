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


def quantize_duration_ticks(duration_ticks: torch.Tensor) -> torch.Tensor:
    """Return direct one-hots for integer beat-grid durations."""

    bucket_indices = duration_ticks.to(torch.long).clamp(0, DURATION_BINS - 1)
    return F.one_hot(bucket_indices, num_classes=DURATION_BINS).to(torch.float32)


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
    ) -> None:
        self.quantization = quantization
        self.subdivision = subdivision

    def _read_ticks(self, path: str | Path) -> tuple[float, ...]:
        with Path(path).open("r", encoding="utf-8") as handle:
            beats = tuple(
                float(line.split()[0])
                for line in handle
                if line.strip()
            )
        return tuple(
            start + (stop - start) * index / self.subdivision
            for start, stop in zip(beats, beats[1:])
            for index in range(self.subdivision)
        ) + (beats[-1],)

    @staticmethod
    def _nearest_tick(time_seconds: float, ticks: tuple[float, ...]) -> int:
        if time_seconds <= ticks[0]:
            position = (time_seconds - ticks[0]) / (ticks[1] - ticks[0])
            return math.floor(position + 0.5)
        if time_seconds >= ticks[-1]:
            position = (
                len(ticks)
                - 1
                + (time_seconds - ticks[-1]) / (ticks[-1] - ticks[-2])
            )
            return math.floor(position + 0.5)

        later = bisect_left(ticks, time_seconds)
        earlier = later - 1
        return (
            earlier
            if time_seconds - ticks[earlier] < ticks[later] - time_seconds
            else later
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
                onset_tick = self._nearest_tick(onset, ticks)
                offset_tick = self._nearest_tick(offset, ticks)
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
    ):
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

    def merge(
        self,
        event_sequence: EventSequence,
        quantization: bool = False,
    ) -> tuple[Chord, ...]:
        events = event_sequence.events
        groups: list[tuple[NoteEvent, ...]] = []
        start = 0
        while start < len(events):
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
                quantized_dur = quantize_duration_ticks(torch.tensor(ticks))
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
            ),
        )
