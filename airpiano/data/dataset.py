"""In-memory alignment dataset and fixed-shape collation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from .slicing import FutureChordSlicer
from .targets import TargetGenerator
from .types import Song


class AlignmentDataset(Dataset):
    """Slice eagerly loaded full songs into fixed hand/chord windows."""

    def __init__(
        self,
        songs: Sequence[Song],
        slicer: FutureChordSlicer,
        target_generator: TargetGenerator,
    ):
        self.songs = tuple(songs)
        self.slicer = slicer
        self.target_generator = target_generator
        samples: list[tuple[int, int, tuple[int, ...]]] = []
        for song_index, song in enumerate(self.songs):
            for frame_start in self.slicer.starts(len(song.hands)):
                positions = self.slicer.candidate_positions(
                    song.chords, frame_start
                )
                # A window after the score ends can contain real hand frames but
                # no chord candidates. Its chord mask would be entirely padding,
                # so it has no cells on which masked BCE can be defined.
                if positions:
                    samples.append((song_index, frame_start, positions))
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        song_index, frame_start, chord_positions = self.samples[int(index)]
        song = self.songs[song_index]
        window = self.slicer.slice(song, frame_start, chord_positions)
        selected_chords = window.pop("selected_chords")
        target = self.target_generator.generate(
            selected_chords,
            frame_start=frame_start,
            segment_frames=self.slicer.segment_frames,
            fps=self.slicer.fps,
            chord_count=self.slicer.chord_count,
        )
        target *= window["frame_mask"][:, None]
        return {
            **window,
            "target": target,
            "record_id": song.record_id,
            "frame_start": frame_start,
            "fps": song.fps,
        }


def collate_alignment_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed arrays while retaining source metadata."""

    tensor_fields = (
        "hands",
        "hand_presence",
        "frame_mask",
        "chords",
        "chord_mask",
        "target",
        "chord_ids",
    )
    return {
        **{field: torch.stack([sample[field] for sample in batch]) for field in tensor_fields},
        "record_id": [str(sample["record_id"]) for sample in batch],
        "frame_start": torch.tensor(
            [int(sample["frame_start"]) for sample in batch], dtype=torch.long
        ),
        "fps": torch.tensor(
            [float(sample["fps"]) for sample in batch], dtype=torch.float64
        ),
    }
