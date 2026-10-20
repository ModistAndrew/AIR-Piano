"""Alignment metrics for recorded real-time releases."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from .render.artifact import InferenceArtifact, load_artifact
from .utils import resolve_path, write_json


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float | None]) -> tuple[float | None, int]:
    present = [float(value) for value in values if value is not None]
    return (sum(present) / len(present), len(present)) if present else (None, 0)


def _population_variance(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    mean = sum(present) / len(present)
    return sum((value - mean) ** 2 for value in present) / len(present)


def _active_chords(
    frames: np.ndarray,
    onset_frames: dict[int, float],
) -> np.ndarray:
    active = np.full(len(frames), -1, dtype=np.int64)
    for chord_slot, onset_frame in onset_frames.items():
        begun = frames >= onset_frame
        active[begun] = np.maximum(active[begun], chord_slot)
    return active


def _alignment_observations(
    artifact: InferenceArtifact,
) -> tuple[np.ndarray, dict[int, float | None]]:
    """Return frame correctness and fractional-frame onset errors."""

    valid_frames = np.flatnonzero(artifact.frame_mask)
    if valid_frames.size == 0:
        return np.empty(0, dtype=np.bool_), {}

    ground_truth_frames = {}
    for chord_slot in np.flatnonzero(artifact.chord_mask):
        chord_slot = int(chord_slot)
        scores = artifact.target[valid_frames, chord_slot]
        if not np.any(scores > 0.0):
            continue
        target_frame = int(valid_frames[int(np.argmax(scores))])
        ground_truth_frames[chord_slot] = artifact.frame_start + target_frame

    if not ground_truth_frames:
        return np.empty(0, dtype=np.bool_), {}

    # Release times are absolute seconds and may fall between frames.
    predicted_frames = {
        chord_slot: float(artifact.release_time[chord_slot]) * artifact.fps
        for chord_slot in ground_truth_frames
        if not np.isnan(artifact.release_time[chord_slot])
    }
    global_frames = artifact.frame_start + valid_frames
    first_onset = min(ground_truth_frames.values())
    evaluated_frames = global_frames[global_frames >= first_onset]
    real_active = _active_chords(evaluated_frames, ground_truth_frames)
    predicted_active = _active_chords(evaluated_frames, predicted_frames)
    chord_errors = {
        chord_slot: (
            predicted_frames[chord_slot] - ground_truth_frame
            if chord_slot in predicted_frames
            else None
        )
        for chord_slot, ground_truth_frame in ground_truth_frames.items()
    }
    return real_active == predicted_active, chord_errors


def _empty_metrics(tolerances: Sequence[int]) -> dict[str, object]:
    return {
        "chord_acc": None,
        "c_ar": {str(value): None for value in tolerances},
        "onset_mae_frames": None,
        "onset_mae_seconds": None,
        "onset_mean_delta_frames": None,
        "onset_mean_delta_seconds": None,
        "chord_coverage": None,
        "correct_frame_count": 0,
        "evaluated_frame_count": 0,
        "correct_chord_counts": {str(value): 0 for value in tolerances},
        "detected_chord_count": 0,
        "ground_truth_chord_count": 0,
        "absolute_error_frames_sum": 0.0,
        "absolute_error_seconds_sum": 0.0,
        "onset_delta_frames_sum": 0.0,
        "onset_delta_seconds_sum": 0.0,
    }


def _metrics_from_observations(
    artifact: InferenceArtifact,
    frame_correct: np.ndarray,
    chord_errors: dict[int, float | None],
    tolerances: Sequence[int],
) -> dict[str, object]:
    if not chord_errors:
        return _empty_metrics(tolerances)

    onset_deltas = [
        float(error) for error in chord_errors.values() if error is not None
    ]
    absolute_errors = [abs(delta) for delta in onset_deltas]
    correct_frame_count = int(frame_correct.sum())
    evaluated_frame_count = len(frame_correct)
    chord_count = len(chord_errors)
    detected_count = len(absolute_errors)
    correct_chord_counts = {
        str(tolerance): sum(error <= tolerance for error in absolute_errors)
        for tolerance in tolerances
    }
    absolute_error_frames_sum = float(sum(absolute_errors))
    absolute_error_seconds_sum = absolute_error_frames_sum / artifact.fps
    onset_delta_frames_sum = float(sum(onset_deltas))
    onset_delta_seconds_sum = onset_delta_frames_sum / artifact.fps

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
        "onset_mean_delta_frames": (
            onset_delta_frames_sum / detected_count
            if detected_count
            else None
        ),
        "onset_mean_delta_seconds": (
            onset_delta_seconds_sum / detected_count
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
        "onset_delta_frames_sum": onset_delta_frames_sum,
        "onset_delta_seconds_sum": onset_delta_seconds_sum,
    }


def calculate_alignment_metrics(
    artifact: InferenceArtifact,
    *,
    tolerances_frames: Sequence[int],
) -> dict[str, object]:
    """Compare one artifact's recorded releases with its target peaks."""

    tolerances = tuple(int(value) for value in tolerances_frames)
    frame_correct, chord_errors = _alignment_observations(artifact)
    return _metrics_from_observations(
        artifact, frame_correct, chord_errors, tolerances
    )


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
    onset_delta_frames = sum(
        float(item["onset_delta_frames_sum"]) for item in metrics
    )
    onset_delta_seconds = sum(
        float(item["onset_delta_seconds_sum"]) for item in metrics
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
        "onset_mean_delta_frames": (
            onset_delta_frames / detected_chords if detected_chords else None
        ),
        "onset_mean_delta_seconds": (
            onset_delta_seconds / detected_chords if detected_chords else None
        ),
        "chord_coverage": _ratio(detected_chords, ground_truth_chords),
        "correct_frame_count": correct_frames,
        "evaluated_frame_count": evaluated_frames,
        "correct_chord_counts": correct_chords,
        "detected_chord_count": detected_chords,
        "ground_truth_chord_count": ground_truth_chords,
        "absolute_error_frames_sum": error_frames,
        "absolute_error_seconds_sum": error_seconds,
        "onset_delta_frames_sum": onset_delta_frames,
        "onset_delta_seconds_sum": onset_delta_seconds,
    }

    chord_acc_values = [item["chord_acc"] for item in metrics]
    chord_acc, chord_acc_artifacts = _mean(chord_acc_values)
    chord_acc_variance = _population_variance(chord_acc_values)
    onset_mae_frames, onset_mae_artifacts = _mean(
        item["onset_mae_frames"] for item in metrics
    )
    onset_mae_seconds, _ = _mean(
        item["onset_mae_seconds"] for item in metrics
    )
    onset_mean_delta_frames, onset_mean_delta_artifacts = _mean(
        item["onset_mean_delta_frames"] for item in metrics
    )
    onset_mean_delta_seconds, _ = _mean(
        item["onset_mean_delta_seconds"] for item in metrics
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
        "chord_acc_variance": chord_acc_variance,
        "c_ar": macro_c_ar,
        "onset_mae_frames": onset_mae_frames,
        "onset_mae_seconds": onset_mae_seconds,
        "onset_mean_delta_frames": onset_mean_delta_frames,
        "onset_mean_delta_seconds": onset_mean_delta_seconds,
        "chord_coverage": chord_coverage,
        "artifact_count": len(metrics),
        "chord_acc_artifact_count": chord_acc_artifacts,
        "chord_metric_artifact_count": chord_artifacts,
        "onset_mae_artifact_count": onset_mae_artifacts,
        "onset_mean_delta_artifact_count": onset_mean_delta_artifacts,
    }
    return micro, macro


def aggregate_indexed_alignment_metrics(
    observations: Sequence[tuple[np.ndarray, dict[int, float | None]]],
    tolerances_frames: Sequence[int],
) -> dict[str, object]:
    """Average frame and chord correctness at each relative index."""

    tolerances = tuple(int(value) for value in tolerances_frames)
    frame_count = max((len(frames) for frames, _ in observations), default=0)
    chord_count = max(
        (max(errors, default=-1) + 1 for _, errors in observations),
        default=0,
    )
    frame_observation_counts = np.zeros(frame_count, dtype=np.int64)
    frame_correct_counts = np.zeros(frame_count, dtype=np.int64)
    chord_observation_counts = np.zeros(chord_count, dtype=np.int64)
    chord_correct_counts = {
        str(tolerance): np.zeros(chord_count, dtype=np.int64)
        for tolerance in tolerances
    }

    for frame_correct, chord_errors in observations:
        count = len(frame_correct)
        frame_observation_counts[:count] += 1
        frame_correct_counts[:count] += frame_correct.astype(np.int64)
        for chord_slot, error in chord_errors.items():
            chord_observation_counts[chord_slot] += 1
            if error is None:
                continue
            for tolerance in tolerances:
                if abs(error) <= tolerance:
                    chord_correct_counts[str(tolerance)][chord_slot] += 1

    frame_rates = np.divide(
        frame_correct_counts,
        frame_observation_counts,
        out=np.full(frame_count, np.nan, dtype=np.float64),
        where=frame_observation_counts > 0,
    )
    chord_rates = {
        key: np.divide(
            counts,
            chord_observation_counts,
            out=np.full(chord_count, np.nan, dtype=np.float64),
            where=chord_observation_counts > 0,
        )
        for key, counts in chord_correct_counts.items()
    }
    return {
        "frame_chord_acc": frame_rates,
        "frame_observation_count": frame_observation_counts,
        "chord_c_ar": chord_rates,
        "chord_observation_count": chord_observation_counts,
    }


def plot_indexed_alignment_summary(
    indexed_metrics: dict[str, object],
    tolerances_frames: Sequence[int],
    output_path: str | Path,
    *,
    c_ar_values: Sequence[float] = (),
) -> Path:
    """Draw indexed alignment metrics and per-artifact C-AR."""

    import os
    import tempfile

    matplotlib_config = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    frame_rates = np.asarray(indexed_metrics["frame_chord_acc"])
    chord_rates = indexed_metrics["chord_c_ar"]
    chord_count = len(indexed_metrics["chord_observation_count"])
    artifact_c_ar = np.asarray(c_ar_values, dtype=np.float64)
    max_tolerance = max(int(value) for value in tolerances_frames)
    c_ar_label = f"C-AR (delta={max_tolerance} frames)"
    width = max(
        8.0,
        min(18.0, max(len(frame_rates) / 60.0, chord_count * 0.5)),
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(width, 14.0),
        constrained_layout=True,
    )

    frame_indices = np.arange(len(frame_rates))
    if len(frame_rates):
        axes[0].plot(frame_indices, frame_rates, color="#4c72b0")
    else:
        axes[0].text(0.5, 0.5, "No evaluated frames", ha="center", va="center")
    axes[0].set_xlabel("Frame index from first ground-truth onset")
    axes[0].set_ylabel("ChordAcc")
    axes[0].set_title("ChordAcc by relative evaluated frame index")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=0.25)

    chord_indices = np.arange(chord_count)
    plotted = False
    for tolerance in tolerances_frames:
        rates = np.asarray(chord_rates[str(int(tolerance))])
        if not len(rates):
            continue
        axes[1].plot(
            chord_indices,
            rates,
            marker="o",
            markersize=3.0,
            label=f"delta={int(tolerance)} frames",
        )
        plotted = True
    if plotted:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No evaluated chords", ha="center", va="center")
    axes[1].set_xlabel("Relative chord index")
    axes[1].set_ylabel("C-AR")
    axes[1].set_title("C-AR by relative chord index")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(alpha=0.25)

    if len(artifact_c_ar):
        axes[2].hist(
            artifact_c_ar,
            bins=np.linspace(0.0, 1.0, 51),
            color="#55a868",
            edgecolor="white",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            f"No artifact {c_ar_label} values",
            ha="center",
            va="center",
        )
    axes[2].set_xlabel(c_ar_label)
    axes[2].set_ylabel("Artifact count")
    axes[2].set_title(f"Per-artifact {c_ar_label} distribution")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].grid(axis="y", alpha=0.25)

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def evaluation(cfg: DictConfig) -> None:
    """Calculate alignment metrics from saved real-time release records."""

    artifact_root = resolve_path(cfg.evaluation.root)
    if not artifact_root.is_dir():
        raise FileNotFoundError(
            f"evaluation artifact root not found: {artifact_root}"
        )
    metrics_path = resolve_path(cfg.evaluation.output_path)
    plot_path = metrics_path.with_suffix(".png")
    tolerances = tuple(int(value) for value in cfg.evaluation.tolerances_frames)
    remove_artifacts = bool(cfg.evaluation.remove_inference_artifacts)
    summary_only = bool(cfg.evaluation.summary_only)
    artifact_paths = sorted(
        path for path in artifact_root.rglob("*.npz") if path.is_file()
    )

    reports = []
    indexed_observations = []
    for index, artifact_path in enumerate(artifact_paths, 1):
        artifact = load_artifact(artifact_path, require_releases=True)
        frame_correct, chord_errors = _alignment_observations(artifact)
        indexed_observations.append((frame_correct, chord_errors))
        item_metrics = _metrics_from_observations(
            artifact, frame_correct, chord_errors, tolerances
        )
        reports.append(
            {
                "artifact_path": artifact_path.relative_to(
                    artifact_root
                ).as_posix(),
                "record_id": artifact.record_id,
                "frame_start": artifact.frame_start,
                "fps": artifact.fps,
                "onset_mode": "release_time",
                "frame_count": int(artifact.frame_mask.sum()),
                "duration_seconds": (
                    int(artifact.frame_mask.sum()) / artifact.fps
                ),
                "chord_count": int(artifact.chord_mask.sum()),
                "metrics": item_metrics,
            }
        )
        if not summary_only:
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
    total_duration_seconds = sum(
        float(report["duration_seconds"]) for report in reports
    )
    average_duration_seconds = (
        total_duration_seconds / len(reports) if reports else None
    )
    indexed_metrics = aggregate_indexed_alignment_metrics(
        indexed_observations, tolerances
    )
    max_tolerance_key = str(max(tolerances))
    c_ar_values = [
        float(item["c_ar"][max_tolerance_key])
        for item in item_metrics
        if item["c_ar"][max_tolerance_key] is not None
    ]
    summary_plot = plot_indexed_alignment_summary(
        indexed_metrics,
        tolerances,
        plot_path,
        c_ar_values=c_ar_values,
    )
    summary = {
        "schema_version": "airpiano-evaluation-v9",
        "artifact_root": str(artifact_root),
        "summary_plot_path": str(summary_plot),
        "onset_mode": "release_time",
        "tolerances_frames": list(tolerances),
        "artifact_count": len(reports),
        "average_duration_seconds": average_duration_seconds,
        "total_duration_seconds": total_duration_seconds,
        "summary_only": summary_only,
        "unique_record_count": len(unique_record_ids),
        "inference_artifacts_removed": remove_artifacts,
        "micro": micro,
        "macro": macro,
    }
    if not summary_only:
        summary["unique_record_ids"] = unique_record_ids
        summary["artifacts"] = reports
    write_json(metrics_path, summary)

    if remove_artifacts:
        try:
            for artifact_path in artifact_paths:
                artifact_path.unlink()
        except OSError:
            summary["inference_artifacts_removed"] = False
            write_json(metrics_path, summary)
            raise

    print(f"evaluation micro: {micro}", flush=True)
    print(f"evaluation macro: {macro}", flush=True)
    print(
        f"evaluation average_duration_seconds={average_duration_seconds}",
        flush=True,
    )
    print(f"wrote evaluation metrics {metrics_path}", flush=True)
    if remove_artifacts:
        print(
            f"removed {len(artifact_paths)} inference artifacts "
            f"from {artifact_root}",
            flush=True,
        )
    print(f"wrote evaluation summary chart {summary_plot}", flush=True)


__all__ = [
    "aggregate_alignment_metrics",
    "aggregate_indexed_alignment_metrics",
    "calculate_alignment_metrics",
    "evaluation",
    "plot_indexed_alignment_summary",
]
