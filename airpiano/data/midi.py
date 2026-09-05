"""Physical-note MIDI parsing and pitch-only chord construction."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path

import mido

from .types import Chord, MidiSequence, Note


class MidoNoteReader:
    """Read tempo-aware physical notes from a standard MIDI file.

    ``mido.MidiFile`` iteration merges tracks and converts delta times to
    seconds. Repeated notes on the same channel and pitch are paired FIFO;
    sustain-pedal messages are deliberately ignored.
    """

    def read(self, path: str | Path) -> tuple[Note, ...]:
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
        return tuple(
            Note(
                id=note_id,
                onset=onset,
                offset=offset,
                pitch=pitch,
                velocity=velocity,
            )
            for note_id, (onset, offset, pitch, velocity) in enumerate(parsed)
        )


class InitializerChordMerger:
    """Group notes using an inclusive initializer-anchored onset tolerance."""

    def __init__(self, tolerance_seconds: float = 0.05):
        self.tolerance_seconds = float(tolerance_seconds)

    def merge(self, notes: tuple[Note, ...]) -> tuple[Chord, ...]:
        chords: list[Chord] = []
        start = 0
        while start < len(notes):
            initializer = notes[start].onset
            stop = start + 1
            while stop < len(notes):
                onset = notes[stop].onset
                roundoff = 8.0 * max(
                    math.ulp(onset),
                    math.ulp(initializer),
                    math.ulp(self.tolerance_seconds),
                )
                if onset - initializer > self.tolerance_seconds + roundoff:
                    break
                stop += 1
            chords.append(Chord(id=len(chords), notes=notes[start:stop]))
            start = stop
        return tuple(chords)


class MidiLoader:
    """Compose note parsing and chord construction."""

    def __init__(self, reader: MidoNoteReader, chord_merger: InitializerChordMerger):
        self.reader = reader
        self.chord_merger = chord_merger

    def load(self, path: str | Path) -> MidiSequence:
        notes = self.reader.read(path)
        return MidiSequence(notes=notes, chords=self.chord_merger.merge(notes))
