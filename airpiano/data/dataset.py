"""In-memory alignment dataset and padded batch collation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..utils import resolve_path
from .slicing import FutureChordSlicer
from .targets import TargetGenerator
from .types import Song


class AlignmentDataset(Dataset):
    """Slice eagerly loaded full songs into aligned hand/chord windows."""

    def __init__(
        self,
        songs: Sequence[Song],
        slicer: FutureChordSlicer,
        target_generator: TargetGenerator,
    ):
        self.songs = tuple(songs)
        self.slicer = slicer
        self.target_generator = target_generator
        samples: list[tuple[int, int, int, tuple[int, ...]]] = []
        for song_index, song in enumerate(self.songs):
            for frame_start, frame_stop, positions in self.slicer.windows(song):
                samples.append(
                    (song_index, frame_start, frame_stop, positions)
                )
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        song_index, frame_start, frame_stop, chord_positions = self.samples[
            int(index)
        ]
        song = self.songs[song_index]
        window = self.slicer.slice(
            song,
            frame_start,
            frame_stop,
            chord_positions,
        )
        selected_chords = window.pop("selected_chords")
        target = self.target_generator.generate(
            selected_chords,
            frame_start=frame_start,
            frame_count=len(window["hands"]),
            fps=self.slicer.fps,
        )
        return {
            **window,
            "target": target,
            "record_id": song.record_id,
            "frame_start": frame_start,
            "fps": song.fps,
        }


def collate_alignment_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad frame/chord axes and create masks from each sample's lengths."""

    frame_lengths = torch.tensor(
        [len(sample["hands"]) for sample in batch], dtype=torch.long
    )
    chord_lengths = torch.tensor(
        [len(sample["chords"]) for sample in batch], dtype=torch.long
    )
    max_frames = int(frame_lengths.max())
    max_chords = int(chord_lengths.max())
    target = torch.zeros(
        (len(batch), max_frames, max_chords),
        dtype=batch[0]["target"].dtype,
    )
    for index, sample in enumerate(batch):
        frame_count, chord_count = sample["target"].shape
        target[index, :frame_count, :chord_count] = sample["target"]

    return {
        "hands": pad_sequence(
            [sample["hands"] for sample in batch], batch_first=True
        ),
        "hand_presence": pad_sequence(
            [sample["hand_presence"] for sample in batch], batch_first=True
        ),
        "frame_mask": (
            torch.arange(max_frames)[None, :] < frame_lengths[:, None]
        ),
        "chords": pad_sequence(
            [sample["chords"] for sample in batch], batch_first=True
        ),
        "chord_mask": (
            torch.arange(max_chords)[None, :] < chord_lengths[:, None]
        ),
        "target": target,
        "chord_ids": pad_sequence(
            [sample["chord_ids"] for sample in batch],
            batch_first=True,
            padding_value=-1,
        ),
        "record_id": [str(sample["record_id"]) for sample in batch],
        "frame_start": torch.tensor(
            [int(sample["frame_start"]) for sample in batch], dtype=torch.long
        ),
        "fps": torch.tensor(
            [float(sample["fps"]) for sample in batch], dtype=torch.float64
        ),
    }


def build_data_components(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate the replaceable data components declared in YAML."""

    return {
        "file_loader": instantiate(cfg.dataset.file_loader),
        "midi_loader": instantiate(cfg.dataset.midi_loader),
        "hand_loader": instantiate(cfg.dataset.hand_loader),
        "slicer": instantiate(cfg.dataset.slicer),
        "target_generator": instantiate(cfg.dataset.target_generator),
    }


def build_dataset(
    cfg: DictConfig,
    split: str,
    *,
    components: dict[str, Any] | None = None,
) -> AlignmentDataset:
    """Eagerly load one split and precompute all slice/chord indices."""

    components = components or build_data_components(cfg)
    songs = components["file_loader"].load(
        split_file=resolve_path(cfg.data.splits[split]),
        midi_root=resolve_path(cfg.data.midi_root),
        hand_root=resolve_path(cfg.data.hand_root),
        midi_loader=components["midi_loader"],
        hand_loader=components["hand_loader"],
    )
    return AlignmentDataset(
        songs=songs,
        slicer=components["slicer"],
        target_generator=components["target_generator"],
    )


def make_loader(
    dataset: AlignmentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Build the deterministic, single-process baseline DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_alignment_batch,
        generator=generator,
    )
