"""Small immutable records shared by the AirPiano data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FilePair:
    """One MIDI/hand pair with an optional beat grid."""

    record_id: str
    midi_path: Path
    hand_path: Path
    beat_path: Path | None = None


@dataclass(frozen=True)
class Note:
    """One physical MIDI note in seconds, with optional beat-grid ticks."""

    id: int
    onset: float
    offset: float
    pitch: int
    velocity: int
    onset_tick: int | None = None
    offset_tick: int | None = None


@dataclass(frozen=True)
class NoteEvent:
    """One onset or offset action tied to its physical MIDI note."""

    note: Note
    is_offset: bool

    @property
    def pitch(self) -> int:
        return self.note.pitch

    @property
    def time(self) -> float:
        return self.note.offset if self.is_offset else self.note.onset

    @property
    def tick(self) -> int | None:
        return self.note.offset_tick if self.is_offset else self.note.onset_tick


@dataclass(frozen=True)
class EventSequence:
    """Chronological MIDI actions expanded from physical notes."""

    events: tuple[NoteEvent, ...]
    include_offsets: bool = False

    @classmethod
    def from_notes(
        cls,
        notes: tuple[Note, ...],
        include_offsets: bool = False,
    ) -> EventSequence:
        events = [NoteEvent(note, False) for note in notes]
        if include_offsets:
            events.extend(NoteEvent(note, True) for note in notes)
        events.sort(
            key=lambda event: (
                event.time,
                event.is_offset,
                event.pitch,
                event.note.id,
            )
        )
        return cls(tuple(events), bool(include_offsets))


@dataclass(frozen=True)
class Chord:
    """A group of simultaneous note actions and its duration encoding."""

    id: int
    events: tuple[NoteEvent, ...]
    onset_notes: tuple[Note, ...]
    offset_notes: tuple[Note, ...]
    time: float
    duration: float
    quantized_dur: torch.Tensor
    duration_ticks: int | None = None
    score_subdivision: float | None = None  # In units of 1/24 beat.


@dataclass(frozen=True)
class MidiSequence:
    """The complete ordered note and chord sequences for a recording."""

    notes: tuple[Note, ...]
    chords: tuple[Chord, ...]


@dataclass(frozen=True)
class HandSequence:
    """Canonical two-hand frames on a fixed-rate recording clock."""

    pose: torch.Tensor
    presence: torch.Tensor
    fps: float


@dataclass(frozen=True)
class Song:
    """One full in-memory MIDI/hand recording."""

    record_id: str
    notes: tuple[Note, ...]
    chords: tuple[Chord, ...]
    hands: torch.Tensor
    presence: torch.Tensor
    fps: float
