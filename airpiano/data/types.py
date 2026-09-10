"""Small immutable records shared by the AirPiano data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FilePair:
    """One MIDI file paired with one canonical hand cache."""

    record_id: str
    midi_path: Path
    hand_path: Path


@dataclass(frozen=True)
class Note:
    """One physical MIDI note."""

    id: int
    onset: float
    offset: float
    pitch: int
    velocity: int


@dataclass(frozen=True)
class Chord:
    """An onset group with raw duration and its possibly scaled encoding."""

    id: int
    notes: tuple[Note, ...]
    duration: float
    quantized_dur: torch.Tensor

    @property
    def onset(self) -> float:
        return self.notes[0].onset


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
