"""Physical-note MIDI parsing and duration-bearing chord construction."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import mido
from .types import Chord, MidiSequence, Note


DEFAULT_TICK_OFFSETS = (0, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 21)


class MidoNoteReader:
    """Read tempo-aware physical notes from a standard MIDI file.

    ``mido.MidiFile`` iteration merges tracks and converts delta times to
    seconds. Repeated notes on the same channel and pitch are paired FIFO;
    sustain-pedal messages are deliberately ignored. Optional beat-grid
    quantization adds tick coordinates without replacing the seconds clock.
    """

    def __init__(
        self,
        subdivision: int = 24,
        allowed_tick_offsets: Sequence[int] = DEFAULT_TICK_OFFSETS,
    ) -> None:
        self.subdivision = int(subdivision)
        offsets = tuple(allowed_tick_offsets)
        if len(offsets) != len(set(offsets)):
            raise ValueError("allowed_tick_offsets must be unique")
        if 0 not in offsets:
            raise ValueError("allowed_tick_offsets must include 0")
        if any(offset < 0 or offset >= self.subdivision for offset in offsets):
            raise ValueError("allowed_tick_offsets must be in [0, subdivision)")
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
            seconds_per_tick = (
                (tick_times[1] - tick_times[0])
                / (tick_positions[1] - tick_positions[0])
            )
            position = tick_positions[0] + (
                time_seconds - tick_times[0]
            ) / seconds_per_tick
            return self._nearest_allowed_tick(position)
        if time_seconds >= tick_times[-1]:
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
        ticks = self._read_ticks(beat_path) if beat_path is not None else None

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
    """Group onset ticks, average physical onsets, and retain tick durations."""

    def merge(self, notes: tuple[Note, ...]) -> tuple[Chord, ...]:
        groups: list[tuple[Note, ...]] = []
        start = 0
        while start < len(notes):
            stop = start + 1
            while (
                stop < len(notes)
                and notes[stop].onset_tick == notes[start].onset_tick
            ):
                stop += 1
            groups.append(notes[start:stop])
            start = stop

        times = [sum(note.onset for note in group) / len(group) for group in groups]
        chords = []
        for index, group in enumerate(groups):
            time = times[index]
            onset_tick = int(group[0].onset_tick)
            if index + 1 < len(groups):
                duration = times[index + 1] - time
                ticks = int(groups[index + 1][0].onset_tick) - onset_tick
            else:
                duration = max(note.offset for note in group) - time
                ticks = max(int(note.offset_tick) for note in group) - onset_tick
            if ticks < 0:
                raise ValueError("duration ticks must be non-negative")
            chords.append(
                Chord(
                    id=index,
                    onset_notes=group,
                    time=time,
                    time_ticks=onset_tick,
                    duration=duration,
                    duration_ticks=ticks,
                )
            )
        return tuple(chords)


class MidiLoader:
    """Compose physical-note parsing and beat-quantized chord construction."""

    def __init__(
        self, reader: MidoNoteReader, chord_merger: InitializerChordMerger
    ):
        self.reader = reader
        self.chord_merger = chord_merger

    def load(
        self,
        path: str | Path,
        beat_path: str | Path,
    ) -> MidiSequence:
        notes = self.reader.read(path, beat_path)
        return MidiSequence(
            notes=notes,
            chords=self.chord_merger.merge(notes),
        )
