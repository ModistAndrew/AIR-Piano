"""Physical-note MIDI parsing and duration-bearing chord construction."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import mido
import torch
import torch.nn.functional as F

from .types import Chord, MidiSequence, Note


DURATION_BINS = 200
DURATION_COEFFICIENT = 4.0
DURATION_CLIP_SECONDS = 16.0


def quantize_duration_seconds(duration_seconds: torch.Tensor) -> torch.Tensor:
    """Return MIDI2ScoreTransformer duration one-hots for absolute seconds."""

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
    """Group notes and optionally scale duration encodings once per song."""

    def __init__(
        self,
        tolerance_seconds: float = 0.05,
        duration_scale_range: Sequence[float] | None = None,
    ):
        self.tolerance_seconds = float(tolerance_seconds)
        if duration_scale_range is None:
            self.duration_scale_range = None
        else:
            minimum, maximum = duration_scale_range
            self.duration_scale_range = (float(minimum), float(maximum))

    def merge(self, notes: tuple[Note, ...]) -> tuple[Chord, ...]:
        groups: list[tuple[Note, ...]] = []
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
            groups.append(notes[start:stop])
            start = stop

        duration_scale = (
            1.0
            if self.duration_scale_range is None
            else random.uniform(*self.duration_scale_range)
        )
        chords = []
        for index, group in enumerate(groups):
            onset = group[0].onset
            if index + 1 < len(groups):
                duration = groups[index + 1][0].onset - onset
            else:
                duration = max(note.offset for note in group) - onset
            quantized_dur = quantize_duration_seconds(
                torch.tensor(duration * duration_scale, dtype=torch.float64)
            )
            chords.append(
                Chord(
                    id=index,
                    notes=group,
                    duration=duration,
                    quantized_dur=quantized_dur,
                )
            )
        return tuple(chords)


class MidiLoader:
    """Compose note parsing and chord construction."""

    def __init__(self, reader: MidoNoteReader, chord_merger: InitializerChordMerger):
        self.reader = reader
        self.chord_merger = chord_merger

    def load(self, path: str | Path) -> MidiSequence:
        notes = self.reader.read(path)
        return MidiSequence(notes=notes, chords=self.chord_merger.merge(notes))
