"""Alignment metrics for stored inference artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from .render.artifact import InferenceArtifact, load_artifact
from .render.sonify import decode_chord_onsets
from .utils import resolve_path, write_json


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


def _empty_metrics(tolerances: Sequence[int]) -> dict[str, object]:
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


def calculate_alignment_metrics(
    artifact: InferenceArtifact,
    *,
    onset_threshold: float,
    tolerances_frames: Sequence[int],
) -> dict[str, object]:
    """Calculate frame and chord alignment metrics for one artifact."""

    tolerances = tuple(int(value) for value in tolerances_frames)
    valid_frames = np.flatnonzero(artifact.frame_mask)
    if valid_frames.size == 0:
        return _empty_metrics(tolerances)

    valid_chords = np.flatnonzero(artifact.chord_mask)
    global_frames = artifact.frame_start + valid_frames
    ground_truth_frames = {}
    for chord_slot in valid_chords:
        chord_slot = int(chord_slot)
        scores = artifact.target[valid_frames, chord_slot]
        if not np.any(scores > 0.0):
            continue
        target_frame = int(valid_frames[int(np.argmax(scores))])
        ground_truth_frames[chord_slot] = artifact.frame_start + target_frame

    if not ground_truth_frames:
        return _empty_metrics(tolerances)

    predicted_local_frames, _ = decode_chord_onsets(
        artifact,
        threshold=onset_threshold,
    )
    predicted_frames = {
        chord_slot: artifact.frame_start
        + int(predicted_local_frames[chord_slot])
        for chord_slot in ground_truth_frames
        if predicted_local_frames[chord_slot] >= 0
    }

    first_onset = min(ground_truth_frames.values())
    evaluated_frames = global_frames[global_frames >= first_onset]
    real_active = _active_chords(evaluated_frames, ground_truth_frames)
    predicted_active = _active_chords(evaluated_frames, predicted_frames)
    correct_frame_count = int((real_active == predicted_active).sum())
    evaluated_frame_count = len(evaluated_frames)

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
    """Return pooled micro metrics and equal-artifact macro metrics."""

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

    chord_acc, chord_acc_artifacts = _mean(
        item["chord_acc"] for item in metrics
    )
    onset_mae_frames, onset_mae_artifacts = _mean(
        item["onset_mae_frames"] for item in metrics
    )
    onset_mae_seconds, _ = _mean(
        item["onset_mae_seconds"] for item in metrics
    )
    chord_coverage, chord_artifacts = _mean(
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
        "artifact_count": len(metrics),
        "chord_acc_artifact_count": chord_acc_artifacts,
        "chord_metric_artifact_count": chord_artifacts,
        "onset_mae_artifact_count": onset_mae_artifacts,
    }
    return micro, macro


def evaluation(cfg: DictConfig) -> None:
    """Calculate alignment metrics for stored inference artifacts."""

    artifact_root = resolve_path(cfg.evaluation.root)
    if not artifact_root.is_dir():
        raise FileNotFoundError(
            f"evaluation artifact root not found: {artifact_root}"
        )
    metrics_path = resolve_path(cfg.evaluation.output_path)
    tolerances = tuple(int(value) for value in cfg.evaluation.tolerances_frames)
    onset_threshold = float(cfg.evaluation.onset_threshold)
    remove_artifacts = bool(cfg.evaluation.remove_inference_artifacts)
    artifact_paths = sorted(
        path for path in artifact_root.rglob("*.npz") if path.is_file()
    )

    reports = []
    for index, artifact_path in enumerate(artifact_paths, 1):
        artifact = load_artifact(artifact_path)
        item_metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=onset_threshold,
            tolerances_frames=tolerances,
        )
        reports.append(
            {
                "artifact_path": artifact_path.relative_to(
                    artifact_root
                ).as_posix(),
                "record_id": artifact.record_id,
                "frame_start": artifact.frame_start,
                "fps": artifact.fps,
                "frame_count": int(artifact.frame_mask.sum()),
                "chord_count": int(artifact.chord_mask.sum()),
                "metrics": item_metrics,
            }
        )
        print(
            f"evaluation {index}/{len(artifact_paths)}: "
            f"{artifact.record_id} frame_start={artifact.frame_start} "
            f"ChordAcc={item_metrics['chord_acc']} "
            f"coverage={item_metrics['chord_coverage']}",
            flush=True,
        )

    item_metrics = [report["metrics"] for report in reports]
    micro, macro = aggregate_alignment_metrics(item_metrics, tolerances)
    unique_record_ids = sorted({report["record_id"] for report in reports})
    summary = {
        "schema_version": "airpiano-evaluation-v2",
        "artifact_root": str(artifact_root),
        "onset_threshold": onset_threshold,
        "tolerances_frames": list(tolerances),
        "artifact_count": len(reports),
        "unique_record_count": len(unique_record_ids),
        "unique_record_ids": unique_record_ids,
        "inference_artifacts_removed": remove_artifacts,
        "micro": micro,
        "macro": macro,
        "artifacts": reports,
    }
    write_json(metrics_path, summary)

    if remove_artifacts:
        try:
            for artifact_path in artifact_paths:
                artifact_path.unlink()
        except OSError:
            summary["inference_artifacts_removed"] = False
            write_json(metrics_path, summary)
            raise

    print(f"wrote evaluation metrics {metrics_path}", flush=True)
    if remove_artifacts:
        print(
            f"removed {len(artifact_paths)} inference artifacts "
            f"from {artifact_root}",
            flush=True,
        )


__all__ = [
    "aggregate_alignment_metrics",
    "calculate_alignment_metrics",
    "evaluation",
]
