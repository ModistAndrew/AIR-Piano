"""In-memory alignment dataset and padded batch collation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
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
        random_hand_shift_seconds: float = 0.0,
        random_pitch_shift_semitones: int = 0,
        random_duration_scale_factors: Sequence[float] | None = None,
        minimum_duration_ticks: int = 0,
    ):
        if random_hand_shift_seconds < 0.0:
            raise ValueError("random_hand_shift_seconds must be non-negative")
        if random_pitch_shift_semitones < 0:
            raise ValueError(
                "random_pitch_shift_semitones must be non-negative"
            )
        if minimum_duration_ticks < 0:
            raise ValueError("minimum_duration_ticks must be non-negative")
        duration_scale_factors = (
            ()
            if random_duration_scale_factors is None
            else tuple(float(value) for value in random_duration_scale_factors)
        )
        if any(value <= 0.0 for value in duration_scale_factors):
            raise ValueError("random duration scale factors must be positive")
        self.songs = tuple(songs)
        self.slicer = slicer
        self.target_generator = target_generator
        self.random_hand_shift_seconds = float(random_hand_shift_seconds)
        self.random_pitch_shift_semitones = int(random_pitch_shift_semitones)
        self.random_duration_scale_factors = duration_scale_factors
        self.minimum_duration_ticks = int(minimum_duration_ticks)
        self.max_random_hand_shift_frames = round(
            self.random_hand_shift_seconds * self.slicer.fps
        )
        samples: list[tuple[int, int, int, tuple[int, ...]]] = []
        for song_index, song in enumerate(self.songs):
            for frame_start, frame_stop, positions in self.slicer.windows(song):
                samples.append(
                    (song_index, frame_start, frame_stop, positions)
                )
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _randomize_hand_frame_bounds(
        self,
        frame_start: int,
        frame_stop: int,
        frame_count: int,
    ) -> tuple[int, int]:
        if self.max_random_hand_shift_frames == 0:
            return frame_start, frame_stop

        shift = self.max_random_hand_shift_frames
        while True:
            shifted_start = max(
                0,
                min(frame_start + random.randint(-shift, shift), frame_count),
            )
            shifted_stop = max(
                0,
                min(frame_stop + random.randint(-shift, shift), frame_count),
            )
            if shifted_stop > shifted_start:
                return shifted_start, shifted_stop

    def _randomize_chord_pitches(
        self, chord_tokens: torch.Tensor
    ) -> torch.Tensor:
        max_shift = self.random_pitch_shift_semitones
        if max_shift == 0:
            return chord_tokens

        pitch_plane_count = 1 if self.slicer.compact_chord_tokens else 2
        pitch_width = 128 * pitch_plane_count
        pitch_values = chord_tokens[:, :pitch_width].reshape(
            len(chord_tokens),
            pitch_plane_count,
            128,
        )
        occupied = torch.nonzero(pitch_values, as_tuple=False)
        if len(occupied) == 0:
            return chord_tokens

        minimum_pitch = int(occupied[:, -1].min())
        maximum_pitch = int(occupied[:, -1].max())
        minimum_shift = max(-max_shift, -minimum_pitch)
        maximum_shift = min(max_shift, 127 - maximum_pitch)
        shift = random.randint(minimum_shift, maximum_shift)
        if shift == 0:
            return chord_tokens

        shifted_tokens = chord_tokens.clone()
        shifted_pitches = shifted_tokens[:, :pitch_width].reshape(
            len(chord_tokens),
            pitch_plane_count,
            128,
        )
        shifted_pitches.zero_()
        if shift > 0:
            shifted_pitches[:, :, shift:] = pitch_values[
                :, :, : 128 - shift
            ]
        else:
            shifted_pitches[:, :, : 128 + shift] = pitch_values[
                :, :, -shift:
            ]
        return shifted_tokens

    def _randomize_chord_durations(
        self, chord_tokens: torch.Tensor
    ) -> torch.Tensor:
        if not self.random_duration_scale_factors:
            return chord_tokens

        duration_start = 128 if self.slicer.compact_chord_tokens else 256
        duration_values = chord_tokens[:, duration_start:]
        duration_ticks = (
            duration_values.argmax(dim=-1) + self.minimum_duration_ticks
        )
        maximum_duration_ticks = (
            self.minimum_duration_ticks + self.slicer.chord_duration_bins - 1
        )
        valid = []
        for factor in self.random_duration_scale_factors:
            scaled_ticks = torch.ceil(
                duration_ticks.to(torch.float64) * factor
            ).to(torch.long)
            if bool(
                (
                    (scaled_ticks >= self.minimum_duration_ticks)
                    & (scaled_ticks <= maximum_duration_ticks)
                ).all()
            ):
                valid.append(factor)
        if not valid:
            raise ValueError(
                "no random duration scale factor preserves duration bounds"
            )

        factor = random.choice(valid)
        if factor == 1.0:
            return chord_tokens
        scaled_ticks = torch.ceil(
            duration_ticks.to(torch.float64) * factor
        ).to(torch.long)

        shifted_tokens = chord_tokens.clone()
        shifted_durations = shifted_tokens[:, duration_start:]
        shifted_durations.zero_()
        shifted_durations.scatter_(
            1,
            (scaled_ticks - self.minimum_duration_ticks).unsqueeze(-1),
            1.0,
        )
        return shifted_tokens

    def __getitem__(self, index: int) -> dict[str, Any]:
        song_index, frame_start, frame_stop, chord_positions = self.samples[
            int(index)
        ]
        song = self.songs[song_index]
        frame_start, frame_stop = self._randomize_hand_frame_bounds(
            frame_start,
            frame_stop,
            len(song.hands),
        )
        window = self.slicer.slice(
            song,
            frame_start,
            frame_stop,
            chord_positions,
        )
        selected_chords = window.pop("selected_chords")
        window["chords"] = self._randomize_chord_pitches(
            window["chords"]
        )
        window["chords"] = self._randomize_chord_durations(
            window["chords"]
        )
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
        beat_root=(
            resolve_path(cfg.data.beat_root)
            if cfg.data.get("beat_root") is not None
            else None
        ),
    )
    return AlignmentDataset(
        songs=songs,
        slicer=components["slicer"],
        target_generator=components["target_generator"],
        random_hand_shift_seconds=(
            float(cfg.dataset.random_hand_shift_seconds)
            if split == "train"
            else 0.0
        ),
        random_pitch_shift_semitones=(
            int(cfg.dataset.get("random_pitch_shift_semitones", 0))
            if split == "train"
            else 0
        ),
        random_duration_scale_factors=(
            tuple(
                float(value)
                for value in cfg.dataset.random_duration_scale_factors
            )
            if split == "train"
            and cfg.dataset.get("random_duration_scale_factors") is not None
            else None
        ),
        minimum_duration_ticks=int(
            cfg.dataset.midi_loader.chord_merger.minimum_duration_ticks
        ),
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
