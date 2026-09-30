"""Realtime preprocessing and inference for synchronized video/MIDI pairs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .data.files import read_record_stems
from .inference import infer_dataset
from .preprocessing.canonical import raw_cache_has_markers
from .utils import resolve_path, write_json


def iter_split_songs(
    cfg: DictConfig,
    split_file: str | Path,
    *,
    overwrite: bool,
    label: str = "realtime",
    components: dict[str, Any] | None = None,
) -> Iterator[tuple[int, int, data.Song, dict[str, Any]]]:
    """Preprocess and stream split records through configured modality roots."""

    split_path = resolve_path(split_file)
    records = read_record_stems([split_path])
    if not records:
        print(
            f"[{label}] no records in {split_path}",
            flush=True,
        )
        return

    video_root = resolve_path(cfg.data.video_root)
    midi_root = resolve_path(cfg.data.midi_root)
    raw_root = resolve_path(cfg.data.raw_wilor_root)
    canonical_root = resolve_path(cfg.data.hand_root)
    beat_root = (
        resolve_path(cfg.data.beat_root)
        if cfg.data.get("beat_root") is not None
        else None
    )
    video_to_ragged = None
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    components = components or data.build_data_components(cfg)
    for index, record_id in enumerate(records, 1):
        video_path = video_root / f"{record_id}.mp4"
        midi_path = midi_root / f"{record_id}.mid"
        if not midi_path.is_file():
            print(f"[{label} skip] {record_id}: missing MIDI {midi_path}")
            continue

        print(
            f"{label} {index}/{len(records)}: preprocess {record_id}",
            flush=True,
        )
        raw_dir = raw_root / record_id
        canonical_path = canonical_root / f"{record_id}.npz"
        reuse_canonical = canonical_path.exists() and not overwrite
        if reuse_canonical or raw_cache_has_markers(raw_dir):
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
                overwrite=overwrite,
            )
        else:
            if not video_path.is_file():
                print(
                    f"[{label} skip] {record_id}: missing video {video_path}",
                    flush=True,
                )
                continue
            if video_to_ragged is None:
                video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
            video_to_ragged.run(video_path, raw_dir, overwrite)
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
                overwrite=overwrite,
            )
        candidate_beat_path = (
            beat_root / f"{record_id}.txt" if beat_root is not None else None
        )
        pair = data.FilePair(
            record_id=record_id,
            midi_path=midi_path,
            hand_path=canonical_path,
            beat_path=(
                candidate_beat_path
                if candidate_beat_path is not None
                and candidate_beat_path.is_file()
                else None
            ),
        )
        song = components["file_loader"].reader.read(
            pair,
            components["midi_loader"],
            components["hand_loader"],
        )
        yield index, len(records), song, components


def realtime(cfg: DictConfig) -> None:
    """Preprocess and infer every record in the configured split."""

    if cfg.realtime.split_file is None:
        raise ValueError("realtime.split_file must be set")
    output_root = resolve_path(cfg.realtime.output_dir)
    checkpoint = resolve_path(cfg.realtime.checkpoint)
    overwrite = bool(cfg.realtime.overwrite)

    for index, record_count, song, components in iter_split_songs(
        cfg,
        cfg.realtime.split_file,
        overwrite=overwrite,
    ):
        output_dir = output_root / song.record_id
        dataset = data.AlignmentDataset(
            songs=[song],
            slicer=components["slicer"],
            target_generator=components["target_generator"],
        )
        loss = infer_dataset(cfg, dataset, checkpoint, output_dir)
        write_json(output_dir / "realtime_loss.json", {"loss": loss})
        print(
            f"realtime {index}/{record_count}: {song.record_id} loss={loss:.8f}",
            flush=True,
        )
