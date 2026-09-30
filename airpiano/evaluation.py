"""Full-song inference and alignment metrics over a configured split."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .data.files import read_record_stems
from .full_song import infer_full_song
from .inference import load_model_checkpoint
from .render.artifact import InferenceArtifact, load_artifact
from .render.sonify import decode_chord_onsets
from .utils import resolve_path, safe_record_name, write_json


def iter_evaluation_songs(
    cfg: DictConfig,
    records: Sequence[str],
    *,
    overwrite: bool,
    components: dict[str, Any],
) -> Iterator[tuple[int, int, data.Song]]:
    """Preprocess and stream split records using full-song semantics."""

    video_root = resolve_path(cfg.data.video_root)
    midi_root = resolve_path(cfg.data.midi_root)
    raw_root = resolve_path(cfg.data.raw_wilor_root)
    hand_root = resolve_path(cfg.data.hand_root)
    beat_root = (
        resolve_path(cfg.data.beat_root)
        if cfg.data.get("beat_root") is not None
        else None
    )
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    video_to_ragged = None

    for index, record_id in enumerate(records, 1):
        video_path = video_root / f"{record_id}.mp4"
        midi_path = midi_root / f"{record_id}.mid"
        reasons = []
        if not video_path.is_file():
            reasons.append(f"missing video {video_path}")
        if not midi_path.is_file():
            reasons.append(f"missing MIDI {midi_path}")
        if reasons:
            print(f"[evaluation skip] {record_id}: {'; '.join(reasons)}")
            continue

        print(
            f"evaluation {index}/{len(records)}: preprocess {record_id}",
            flush=True,
        )
        raw_dir = raw_root / record_id
        hand_path = hand_root / f"{record_id}.npz"
        try:
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=hand_path,
                overwrite=overwrite,
            )
        except FileNotFoundError:
            if video_to_ragged is None:
                video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
            video_to_ragged.run(video_path, raw_dir, overwrite)
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=hand_path,
                overwrite=overwrite,
            )

        candidate_beat_path = (
            beat_root / f"{record_id}.txt" if beat_root is not None else None
        )
        pair = data.FilePair(
            record_id=record_id,
            midi_path=midi_path,
            hand_path=hand_path,
            beat_path=(
                candidate_beat_path
                if candidate_beat_path is not None
                and candidate_beat_path.is_file()
                else None
            ),
        )
        yield (
            index,
            len(records),
            components["file_loader"].reader.read(
                pair,
                components["midi_loader"],
                components["hand_loader"],
            ),
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float | None]) -> tuple[float | None, int]:
    present = [float(value) for value in values if value is not None]
    return (sum(present) / len(present), len(present)) if present else (None, 0)


def _active_chords(
    frames: np.ndarray,
    onset_frames: dict[int, int],
) -> np.ndarray:
    active = np.full(len(frames), -1, dtype=np.int64)
    for chord_slot, onset_frame in onset_frames.items():
        begun = frames >= onset_frame
        active[begun] = np.maximum(active[begun], chord_slot)
    return active


def calculate_alignment_metrics(
    artifact: InferenceArtifact,
    ground_truth_onset_seconds: Sequence[float],
    *,
    onset_threshold: float,
    tolerances_frames: Sequence[int],
) -> dict[str, object]:
    """Calculate frame and chord alignment metrics for one full-song artifact."""

    tolerances = tuple(int(value) for value in tolerances_frames)
    valid_frames = np.flatnonzero(artifact.frame_mask)
    valid_chords = np.flatnonzero(artifact.chord_mask)
    if len(ground_truth_onset_seconds) != len(valid_chords):
        raise ValueError(
            "ground-truth onset count must match valid artifact chords"
        )

    if valid_frames.size == 0:
        return {
            "chord_acc": None,
            "c_ar": {str(value): None for value in tolerances},
            "onset_mae_frames": None,
            "onset_mae_seconds": None,
            "chord_coverage": None,
            "correct_frame_count": 0,
            "evaluated_frame_count": 0,
            "correct_chord_counts": {str(value): 0 for value in tolerances},
            "detected_chord_count": 0,
            "ground_truth_chord_count": 0,
            "absolute_error_frames_sum": 0.0,
            "absolute_error_seconds_sum": 0.0,
        }

    global_frames = artifact.frame_start + valid_frames
    timeline_start = float(global_frames[0]) / artifact.fps
    timeline_end = float(global_frames[-1] + 1) / artifact.fps
    ground_truth_frames: dict[int, int] = {}
    frame_times = global_frames.astype(np.float64) / artifact.fps
    for chord_slot, onset_seconds in zip(
        valid_chords, ground_truth_onset_seconds
    ):
        onset_seconds = float(onset_seconds)
        if not timeline_start <= onset_seconds < timeline_end:
            continue
        closest = int(np.argmin(np.abs(frame_times - onset_seconds)))
        ground_truth_frames[int(chord_slot)] = int(global_frames[closest])

    predicted_local_frames, _ = decode_chord_onsets(
        artifact,
        threshold=onset_threshold,
    )
    predicted_frames = {
        int(chord_slot): artifact.frame_start + int(predicted_local_frames[chord_slot])
        for chord_slot in valid_chords
        if predicted_local_frames[chord_slot] >= 0
    }

    if ground_truth_frames:
        first_onset = min(ground_truth_frames.values())
        evaluated_frames = global_frames[global_frames >= first_onset]
        real_active = _active_chords(evaluated_frames, ground_truth_frames)
        predicted_active = _active_chords(evaluated_frames, predicted_frames)
        correct_frame_count = int((real_active == predicted_active).sum())
        evaluated_frame_count = len(evaluated_frames)
    else:
        correct_frame_count = 0
        evaluated_frame_count = 0

    absolute_errors = [
        abs(predicted_frames[chord_slot] - ground_truth_frame)
        for chord_slot, ground_truth_frame in ground_truth_frames.items()
        if chord_slot in predicted_frames
    ]
    chord_count = len(ground_truth_frames)
    detected_count = len(absolute_errors)
    correct_chord_counts = {
        str(tolerance): sum(error <= tolerance for error in absolute_errors)
        for tolerance in tolerances
    }
    absolute_error_frames_sum = float(sum(absolute_errors))
    absolute_error_seconds_sum = absolute_error_frames_sum / artifact.fps

    return {
        "chord_acc": _ratio(correct_frame_count, evaluated_frame_count),
        "c_ar": {
            str(tolerance): _ratio(
                correct_chord_counts[str(tolerance)], chord_count
            )
            for tolerance in tolerances
        },
        "onset_mae_frames": (
            absolute_error_frames_sum / detected_count
            if detected_count
            else None
        ),
        "onset_mae_seconds": (
            absolute_error_seconds_sum / detected_count
            if detected_count
            else None
        ),
        "chord_coverage": _ratio(detected_count, chord_count),
        "correct_frame_count": correct_frame_count,
        "evaluated_frame_count": evaluated_frame_count,
        "correct_chord_counts": correct_chord_counts,
        "detected_chord_count": detected_count,
        "ground_truth_chord_count": chord_count,
        "absolute_error_frames_sum": absolute_error_frames_sum,
        "absolute_error_seconds_sum": absolute_error_seconds_sum,
    }


def aggregate_alignment_metrics(
    metrics: Sequence[dict[str, object]],
    tolerances_frames: Sequence[int],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return pooled micro metrics and equal-record macro metrics."""

    tolerances = tuple(int(value) for value in tolerances_frames)
    correct_frames = sum(int(item["correct_frame_count"]) for item in metrics)
    evaluated_frames = sum(int(item["evaluated_frame_count"]) for item in metrics)
    detected_chords = sum(int(item["detected_chord_count"]) for item in metrics)
    ground_truth_chords = sum(
        int(item["ground_truth_chord_count"]) for item in metrics
    )
    error_frames = sum(float(item["absolute_error_frames_sum"]) for item in metrics)
    error_seconds = sum(
        float(item["absolute_error_seconds_sum"]) for item in metrics
    )
    correct_chords = {
        str(tolerance): sum(
            int(item["correct_chord_counts"][str(tolerance)])
            for item in metrics
        )
        for tolerance in tolerances
    }
    micro = {
        "chord_acc": _ratio(correct_frames, evaluated_frames),
        "c_ar": {
            str(tolerance): _ratio(
                correct_chords[str(tolerance)], ground_truth_chords
            )
            for tolerance in tolerances
        },
        "onset_mae_frames": (
            error_frames / detected_chords if detected_chords else None
        ),
        "onset_mae_seconds": (
            error_seconds / detected_chords if detected_chords else None
        ),
        "chord_coverage": _ratio(detected_chords, ground_truth_chords),
        "correct_frame_count": correct_frames,
        "evaluated_frame_count": evaluated_frames,
        "correct_chord_counts": correct_chords,
        "detected_chord_count": detected_chords,
        "ground_truth_chord_count": ground_truth_chords,
        "absolute_error_frames_sum": error_frames,
        "absolute_error_seconds_sum": error_seconds,
    }

    chord_acc, chord_acc_records = _mean(
        item["chord_acc"] for item in metrics
    )
    onset_mae_frames, onset_mae_records = _mean(
        item["onset_mae_frames"] for item in metrics
    )
    onset_mae_seconds, _ = _mean(
        item["onset_mae_seconds"] for item in metrics
    )
    chord_coverage, chord_records = _mean(
        item["chord_coverage"] for item in metrics
    )
    macro_c_ar = {}
    for tolerance in tolerances:
        value, _ = _mean(
            item["c_ar"][str(tolerance)] for item in metrics
        )
        macro_c_ar[str(tolerance)] = value
    macro = {
        "chord_acc": chord_acc,
        "c_ar": macro_c_ar,
        "onset_mae_frames": onset_mae_frames,
        "onset_mae_seconds": onset_mae_seconds,
        "chord_coverage": chord_coverage,
        "record_count": len(metrics),
        "chord_acc_record_count": chord_acc_records,
        "chord_metric_record_count": chord_records,
        "onset_mae_record_count": onset_mae_records,
    }
    return micro, macro


def _remove_artifact_directory(path: Path, output_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_output = output_dir.resolve()
    if resolved_path.parent != resolved_output or resolved_path == resolved_output:
        raise ValueError(f"unsafe evaluation artifact directory: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def evaluation(cfg: DictConfig) -> None:
    """Run full-song inference and alignment metrics for one configured split."""

    split_file = resolve_path(cfg.evaluation.split_file)
    split_name = split_file.stem
    output_dir = resolve_path(cfg.evaluation.output_dir)
    artifact_dir = output_dir / split_name
    metrics_path = output_dir / f"{split_name}.json"
    checkpoint = resolve_path(cfg.evaluation.checkpoint)
    tolerances = tuple(int(value) for value in cfg.evaluation.tolerances_frames)
    onset_threshold = float(cfg.evaluation.onset_threshold)
    remove_artifacts = bool(cfg.evaluation.remove_inference_artifacts)
    records = read_record_stems([split_file])

    components = data.build_data_components(cfg)
    device = torch.device(str(cfg.device))
    model = load_model_checkpoint(checkpoint, device)
    alignment_loss = str(cfg.get("alignment_loss", "bce"))
    reports = []
    total_loss = 0.0
    total_weight = 0

    for index, record_count, song in iter_evaluation_songs(
        cfg,
        records,
        overwrite=bool(cfg.evaluation.overwrite),
        components=components,
    ):
        print(
            f"evaluation {index}/{record_count}: infer {song.record_id} "
            f"({len(song.hands)} frames, {len(song.chords)} chords)",
            flush=True,
        )
        artifact_path = artifact_dir / safe_record_name(song.record_id, 0)
        loss, loss_weight, shift_count = infer_full_song(
            model,
            song,
            components["slicer"],
            components["target_generator"],
            device,
            artifact_path,
            bf16=bool(cfg.train.bf16),
            alignment_threshold=float(cfg.evaluation.alignment_threshold),
            advance_after_chords=int(cfg.evaluation.advance_after_chords),
            alignment_loss=alignment_loss,
        )
        if loss_weight:
            total_loss += loss * loss_weight
            total_weight += loss_weight
        artifact = load_artifact(artifact_path)
        item_metrics = calculate_alignment_metrics(
            artifact,
            [chord.time for chord in song.chords],
            onset_threshold=onset_threshold,
            tolerances_frames=tolerances,
        )
        reports.append(
            {
                "record_id": song.record_id,
                "artifact_name": artifact_path.name,
                "fps": song.fps,
                "frame_count": len(song.hands),
                "chord_count": len(song.chords),
                "full_song_loss": loss,
                "loss_weight": loss_weight,
                "shift_count": shift_count,
                "metrics": item_metrics,
            }
        )
        print(
            f"evaluation {index}/{record_count}: {song.record_id} "
            f"ChordAcc={item_metrics['chord_acc']} "
            f"coverage={item_metrics['chord_coverage']}",
            flush=True,
        )

    item_metrics = [report["metrics"] for report in reports]
    micro, macro = aggregate_alignment_metrics(item_metrics, tolerances)
    summary = {
        "schema_version": "airpiano-evaluation-v1",
        "split_name": split_name,
        "split_file": str(split_file),
        "checkpoint": str(checkpoint),
        "artifact_directory": str(artifact_dir.resolve()),
        "inference_artifacts_removed": remove_artifacts,
        "onset_threshold": onset_threshold,
        "tolerances_frames": list(tolerances),
        "requested_record_count": len(records),
        "evaluated_record_count": len(reports),
        "skipped_record_count": len(records) - len(reports),
        "full_song_loss": total_loss / total_weight if total_weight else None,
        "micro": micro,
        "macro": macro,
        "records": reports,
    }
    write_json(metrics_path, summary)
    if remove_artifacts:
        try:
            _remove_artifact_directory(artifact_dir, output_dir)
        except OSError:
            summary["inference_artifacts_removed"] = False
            write_json(metrics_path, summary)
            raise

    print(f"wrote evaluation metrics {metrics_path}", flush=True)
    if remove_artifacts:
        print(f"removed inference artifacts {artifact_dir}", flush=True)


__all__ = [
    "aggregate_alignment_metrics",
    "calculate_alignment_metrics",
    "evaluation",
    "iter_evaluation_songs",
]
