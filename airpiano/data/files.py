"""Pair and load prepared MIDI, canonical hand, and beat files.

Training and validation use FileLoader.load to load a split eagerly. This module
reports and skips missing files; inference-time cache preparation belongs to
song_input.iter_split_songs. Both paths share EagerSongReader for decoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .types import FilePair, Song


def read_record_stems(split_files: Iterable[str | Path]) -> list[str]:
    """Read unique path-like record stems in their first-seen order."""

    records: list[str] = []
    seen: set[str] = set()
    for split_file in split_files:
        with Path(split_file).open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                record_id = Path(line).as_posix()
                if record_id not in seen:
                    seen.add(record_id)
                    records.append(record_id)
    return records


class RelativeStemPairer:
    """Pair MIDI, hand, and beat files named by stems in a split list.

    Missing inputs are reported individually and omitted, so a partially
    materialized hand cache naturally resolves to its complete intersection.
    A stem is the whole stripped line: it may be nested, contain spaces, or
    have dots in its basename. Extensions are therefore appended rather than
    replacing a suffix.
    """

    def pair(
        self,
        split_file: str | Path,
        midi_root: str | Path,
        hand_root: str | Path,
        beat_root: str | Path,
    ) -> list[FilePair]:
        split_path = Path(split_file)
        midi_directory = Path(midi_root)
        hand_directory = Path(hand_root)
        beat_directory = Path(beat_root)
        pairs: list[FilePair] = []
        records = read_record_stems([split_path])

        for record_id in records:
            midi_path = midi_directory / f"{record_id}.mid"
            hand_path = hand_directory / f"{record_id}.npz"
            beat_path = beat_directory / f"{record_id}.txt"
            reasons: list[str] = []
            if not midi_path.is_file():
                reasons.append(f"missing MIDI {midi_path}")
            if not hand_path.is_file():
                reasons.append(f"missing hand cache {hand_path}")
            if not beat_path.is_file():
                reasons.append(f"missing beats {beat_path}")
            if reasons:
                print(f"[data] skip {record_id}: {'; '.join(reasons)}")
                continue
            pairs.append(
                FilePair(
                    record_id=record_id,
                    midi_path=midi_path,
                    hand_path=hand_path,
                    beat_path=beat_path,
                )
            )
        print(
            f"[data] pairing summary: requested={len(records)} "
            f"paired={len(pairs)} missing={len(records) - len(pairs)}"
        )
        return pairs


class EagerSongReader:
    """Load both modalities for one pair into a full in-memory song."""

    def read(self, pair: FilePair, midi_loader: Any, hand_loader: Any) -> Song:
        midi = midi_loader.load(pair.midi_path, pair.beat_path)
        hands = hand_loader.load(pair.hand_path)
        return Song(
            record_id=pair.record_id,
            notes=midi.notes,
            chords=midi.chords,
            hands=hands.pose,
            presence=hands.presence,
            fps=hands.fps,
        )


class FileLoader:
    """Compose pairing and eager song loading without dataset-specific logic."""

    def __init__(self, pairer: RelativeStemPairer, reader: EagerSongReader):
        self.pairer = pairer
        self.reader = reader

    def load(
        self,
        split_file: str | Path,
        midi_root: str | Path,
        hand_root: str | Path,
        midi_loader: Any,
        hand_loader: Any,
        beat_root: str | Path,
    ) -> list[Song]:
        pairs = self.pairer.pair(split_file, midi_root, hand_root, beat_root)
        return [
            self.reader.read(pair, midi_loader, hand_loader) for pair in pairs
        ]
