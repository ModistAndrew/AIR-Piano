"""Realtime preprocessing and inference for one synchronized video/MIDI pair."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from .data import AlignmentDataset, FilePair
from .train import build_data_components, infer_dataset
from .utils import resolve_output_path, write_json


def realtime(cfg: DictConfig) -> None:
    """Preprocess one synchronized video/MIDI pair and infer every slice."""

    root = resolve_output_path(cfg, cfg.realtime.root)
    stem = Path(str(cfg.realtime.stem))
    record_id = stem.as_posix()
    video_path = root / "video" / f"{record_id}.mp4"
    midi_path = root / "midi" / f"{record_id}.mid"
    raw_dir = root / "raw_wilor" / stem
    canonical_path = root / "canonical" / f"{record_id}.npz"
    output_dir = resolve_output_path(cfg, cfg.realtime.output_dir) / stem
    checkpoint = resolve_output_path(cfg, cfg.realtime.checkpoint)
    overwrite = bool(cfg.realtime.overwrite)

    # Realtime uses the exact offline preprocessing components before entering
    # the shared eager dataset and checkpoint-inference path.
    video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    video_to_ragged.run(video_path, raw_dir, overwrite)
    ragged_to_canonical.run(
        raw_record_dir=raw_dir,
        output_path=canonical_path,
        overwrite=overwrite,
    )

    components = build_data_components(cfg)
    pair = FilePair(
        record_id=record_id,
        midi_path=midi_path,
        hand_path=canonical_path,
    )
    song = components["file_loader"].reader.read(
        pair,
        components["midi_loader"],
        components["hand_loader"],
    )
    dataset = AlignmentDataset(
        songs=[song],
        slicer=components["slicer"],
        target_generator=components["target_generator"],
    )
    loss = infer_dataset(cfg, dataset, checkpoint, output_dir)
    write_json(output_dir / "realtime_loss.json", {"loss": loss})
    print(f"realtime loss: {loss:.8f}", flush=True)
