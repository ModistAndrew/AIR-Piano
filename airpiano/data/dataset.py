"""In-memory alignment dataset and padded batch collation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..utils import resolve_path
from .augmentation import WindowAugmenter
from .slicing import FutureChordSlicer
from .targets import TargetGenerator
from .tokenization import ChordTokenizer
from .types import AlignmentWindow, Song


class AlignmentDataset(Dataset):
    """Compose alignment samples using one dataset-wide frame rate."""

    def __init__(
        self,
        songs: Sequence[Song],
        slicer: FutureChordSlicer,
        target_generator: TargetGenerator,
        *,
        fps: float,
        tokenizer: ChordTokenizer,
        augmenter: WindowAugmenter | None = None,
    ):
        self.fps = float(fps)
        self.songs = tuple(songs)
        if any(song.fps != self.fps for song in self.songs):
            raise ValueError("song and dataset frame rates must match")
        self.slicer = slicer
        self.target_generator = target_generator
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.samples = tuple(
            (song_index, window)
            for song_index, song in enumerate(self.songs)
            for window in self.slicer.windows(song)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        song_index, bounds = self.samples[int(index)]
        song = self.songs[song_index]
        frame_start, frame_stop, chord_positions = bounds
        window = AlignmentWindow(
            hands=song.hands[frame_start:frame_stop],
            hand_presence=song.presence[frame_start:frame_stop],
            chords=tuple(song.chords[position] for position in chord_positions),
            frame_start=frame_start,
        )
        if self.augmenter is not None:
            window = self.augmenter.augment(window, song)
        return {
            "hands": window.hands,
            "hand_presence": window.hand_presence,
            "chords": self.tokenizer.tokenize(window.chords),
            "chord_ids": torch.tensor(
                [chord.id for chord in window.chords], dtype=torch.long
            ),
            "target": self.target_generator.generate(
                window.chords,
                frame_start=window.frame_start,
                frame_count=len(window.hands),
                fps=self.fps,
            ),
            "record_id": song.record_id,
            "frame_start": window.frame_start,
            "fps": self.fps,
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
    """Construct components with the shared dataset frame rate."""

    fps = float(cfg.dataset.fps)
    return {
        "file_loader": instantiate(cfg.dataset.file_loader),
        "midi_loader": instantiate(cfg.dataset.midi_loader),
        "hand_loader": instantiate(cfg.dataset.hand_loader),
        "slicer": instantiate(cfg.dataset.slicer, fps=fps),
        "tokenizer": instantiate(cfg.dataset.tokenizer),
        "augmenter": instantiate(cfg.dataset.augmenter, fps=fps),
        "target_generator": instantiate(cfg.dataset.target_generator),
    }


def build_dataset(
    cfg: DictConfig,
    split: str,
    *,
    components: dict[str, Any] | None = None,
    split_file: str | Path | None = None,
) -> AlignmentDataset:
    """Eagerly load one split and precompute all slice/chord indices."""

    components = components or build_data_components(cfg)
    songs = components["file_loader"].load(
        split_file=resolve_path(
            cfg.data.splits[split] if split_file is None else split_file
        ),
        midi_root=resolve_path(cfg.data.midi_root),
        hand_root=resolve_path(cfg.data.hand_root),
        midi_loader=components["midi_loader"],
        hand_loader=components["hand_loader"],
        beat_root=resolve_path(cfg.data.beat_root),
    )
    return AlignmentDataset(
        songs=songs,
        fps=float(cfg.dataset.fps),
        slicer=components["slicer"],
        target_generator=components["target_generator"],
        tokenizer=components["tokenizer"],
        augmenter=components["augmenter"] if split == "train" else None,
    )


def make_loader(
    dataset: AlignmentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Build the deterministic, single-process DataLoader."""

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
