"""Realtime preprocessing and inference for synchronized video/MIDI pairs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .inference import infer_dataset
from .utils import resolve_path, write_json


def iter_realtime_songs(
    cfg: DictConfig,
    root: Path,
    *,
    overwrite: bool,
    label: str = "realtime",
    components: dict[str, Any] | None = None,
) -> Iterator[tuple[int, int, data.Song, dict[str, Any]]]:
    """Preprocess paired files from root video/MIDI directories."""

    video_root = root / "video"
    midi_root = root / "midi"
    videos = {
        path.name[: -len(".mp4")]: path
        for path in video_root.glob("*.mp4")
        if path.is_file()
    }
    midis = {
        path.name[: -len(".mid")]: path
        for path in midi_root.glob("*.mid")
        if path.is_file()
    }
    for record_id in sorted(videos.keys() - midis.keys()):
        print(f"[{label} missing] {record_id}: {midi_root / f'{record_id}.mid'}")
    for record_id in sorted(midis.keys() - videos.keys()):
        print(f"[{label} missing] {record_id}: {video_root / f'{record_id}.mp4'}")
    record_ids = sorted(videos.keys() & midis.keys())
    if not record_ids:
        print(
            f"[{label}] no matching pairs under {video_root} and {midi_root}",
            flush=True,
        )
        return

    raw_root = root / resolve_path(cfg.data.raw_wilor_root).name
    canonical_root = root / resolve_path(cfg.data.hand_root).name

    # The expensive WiLoR component is constructed only after canonicalization
    # shows that an existing canonical or ragged cache cannot be reused.
    video_to_ragged = None
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    components = components or data.build_data_components(cfg)
    for index, record_id in enumerate(record_ids, 1):
        print(
            f"{label} {index}/{len(record_ids)}: preprocess {record_id}",
            flush=True,
        )
        raw_dir = raw_root / record_id
        canonical_path = canonical_root / f"{record_id}.npz"
        try:
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
                overwrite=overwrite,
            )
        except FileNotFoundError:
            if video_to_ragged is None:
                video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
            video_to_ragged.run(videos[record_id], raw_dir, overwrite)
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
                overwrite=overwrite,
            )
        pair = data.FilePair(
            record_id=record_id,
            midi_path=midis[record_id],
            hand_path=canonical_path,
        )
        song = components["file_loader"].reader.read(
            pair,
            components["midi_loader"],
            components["hand_loader"],
        )
        yield index, len(record_ids), song, components


def realtime(cfg: DictConfig) -> None:
    """Preprocess and infer every direct video/MIDI pair in one root."""

    root = resolve_path(cfg.realtime.root)
    output_root = resolve_path(cfg.realtime.output_dir)
    checkpoint = resolve_path(cfg.realtime.checkpoint)
    overwrite = bool(cfg.realtime.overwrite)

    for index, record_count, song, components in iter_realtime_songs(
        cfg,
        root,
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
