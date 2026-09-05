"""Realtime preprocessing and inference for synchronized video/MIDI pairs."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .inference import infer_dataset
from .utils import resolve_output_path, write_json


def realtime(cfg: DictConfig) -> None:
    """Preprocess and infer every direct video/MIDI pair in one root."""

    root = resolve_output_path(cfg, cfg.realtime.root)
    videos = {
        path.name[: -len(".mp4")]: path
        for path in root.glob("*.mp4")
        if path.is_file()
    }
    midis = {
        path.name[: -len(".mid")]: path
        for path in root.glob("*.mid")
        if path.is_file()
    }
    for record_id in sorted(videos.keys() - midis.keys()):
        print(f"[realtime missing] {record_id}: {root / f'{record_id}.mid'}")
    for record_id in sorted(midis.keys() - videos.keys()):
        print(f"[realtime missing] {record_id}: {root / f'{record_id}.mp4'}")
    record_ids = sorted(videos.keys() & midis.keys())
    if not record_ids:
        return

    output_root = resolve_output_path(cfg, cfg.realtime.output_dir)
    checkpoint = resolve_output_path(cfg, cfg.realtime.checkpoint)
    overwrite = bool(cfg.realtime.overwrite)

    # Components are shared across records so the lazy WiLoR models load once.
    video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    components = data.build_data_components(cfg)
    for index, record_id in enumerate(record_ids, 1):
        raw_dir = root / "raw_wilor" / record_id
        canonical_path = root / "canonical" / f"{record_id}.npz"
        output_dir = output_root / record_id
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
        dataset = data.AlignmentDataset(
            songs=[song],
            slicer=components["slicer"],
            target_generator=components["target_generator"],
        )
        loss = infer_dataset(cfg, dataset, checkpoint, output_dir)
        write_json(output_dir / "realtime_loss.json", {"loss": loss})
        print(
            f"realtime {index}/{len(record_ids)}: {record_id} loss={loss:.8f}",
            flush=True,
        )
