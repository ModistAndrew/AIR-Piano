"""Small immutable records shared by the AirPiano data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FilePair:
    """One MIDI/hand pair with its required beat grid."""

    record_id: str
    midi_path: Path
    hand_path: Path
    beat_path: Path


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
class Chord:
    """Grouped notes with a mean physical onset and native beat-grid ticks."""

    id: int
    onset_notes: tuple[Note, ...]
    time: float
    time_ticks: int
    duration: float
    duration_ticks: int


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


@dataclass(frozen=True)
class AlignmentWindow:
    """Raw aligned score and hand frames on the original recording clock."""

    hands: torch.Tensor
    hand_presence: torch.Tensor
    chords: tuple[Chord, ...]
    frame_start: int
