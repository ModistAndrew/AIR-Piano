"""Split-driven preparation of synchronized MIDI and hand data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .data.files import read_record_stems
from .preprocessing.canonical import raw_cache_has_markers
from .utils import resolve_path


def iter_split_songs(
    cfg: DictConfig,
    split_file: str | Path,
    *,
    overwrite: bool,
    label: str,
    components: dict[str, Any] | None = None,
) -> Iterator[tuple[int, int, data.Song, dict[str, Any]]]:
    """Prepare and load split records through configured modality roots."""

    split_path = resolve_path(split_file)
    records = read_record_stems([split_path])
    if not records:
        print(f"[{label}] no records in {split_path}", flush=True)
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

        beat_path = None
        if beat_root is not None:
            beat_path = beat_root / f"{record_id}.txt"
            if not beat_path.is_file():
                print(
                    f"[{label} skip] {record_id}: missing beats {beat_path}",
                    flush=True,
                )
                continue

        print(
            f"{label} {index}/{len(records)}: prepare {record_id}",
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

        pair = data.FilePair(
            record_id=record_id,
            midi_path=midi_path,
            hand_path=canonical_path,
            beat_path=beat_path,
        )
        song = components["file_loader"].reader.read(
            pair,
            components["midi_loader"],
            components["hand_loader"],
        )
        yield index, len(records), song, components


__all__ = ["iter_split_songs"]
