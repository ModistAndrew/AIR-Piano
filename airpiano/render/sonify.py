"""Decode an inference artifact into a simple MIDI performance."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..data.midi import MidoNoteReader
from ..utils import resolve_path, write_json
from .artifact import InferenceArtifact, load_artifact


def sonify(cfg: DictConfig) -> None:
    """Recursively sonify every inference artifact below the configured root."""

    root = resolve_path(cfg.sonify.root)
    midi_root = resolve_path(cfg.sonify.midi_root)
    onset_mode = str(cfg.sonify.onset_mode)
    threshold = float(cfg.sonify.threshold)
    target_radius_seconds = float(cfg.sonify.target_radius_seconds)
    legacy_validated = False
    onset_reports = []
    for input_path in sorted(root.rglob("*.npz")):
        if not input_path.is_file():
            continue
        artifact = load_artifact(input_path)
        if artifact.release_time is None and not legacy_validated:
            validate_onset_target_config(cfg, onset_mode)
            onset_prediction_horizon_seconds(
                threshold, target_radius_seconds, onset_mode
            )
            legacy_validated = True
        prediction_output = sonify_artifact(
            input_path,
            input_path.with_suffix(".mid"),
            threshold=threshold,
            onset_mode=onset_mode,
            target_radius_seconds=target_radius_seconds,
            duration_seconds=float(cfg.sonify.duration_seconds),
            velocity=int(cfg.sonify.velocity),
        )
        midi_path = midi_root / f"{artifact.record_id}.mid"
        ground_truth_output = sonify_ground_truth(
            input_path,
            midi_path,
            input_path.with_suffix(".gt.mid"),
        )
        onset_report = calculate_onset_deltas(
            input_path,
            midi_path,
            threshold=threshold,
            onset_mode=onset_mode,
            target_radius_seconds=target_radius_seconds,
            chord_tolerance_seconds=float(
                cfg.sonify.chord_tolerance_seconds
            ),
        )
        onset_reports.append(onset_report)
        print(f"wrote prediction {prediction_output}", flush=True)
        print(f"wrote ground truth {ground_truth_output}", flush=True)

    if onset_reports:
        summary_output = write_onset_delta_summary(
            onset_reports,
            root / "onset_delta_summary.json",
        )
        chart_output = plot_onset_delta_summary(
            summary_output,
            root / "onset_delta_summary.png",
        )
        print(f"wrote onset delta summary {summary_output}", flush=True)
        print(f"wrote onset delta chart {chart_output}", flush=True)


def _seconds_to_tick(seconds: float, *, ticks_per_beat: int, tempo: int) -> int:
    ticks_per_second = ticks_per_beat * 1_000_000 / tempo
    return int(round(seconds * ticks_per_second))


def _write_note_events(
    events: list[tuple[int, int, int, int]], output_path: str | Path
) -> Path:
    """Write sorted note events as one constant-tempo MIDI track."""

    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    previous_tick = 0
    # At equal ticks, priority zero puts note-offs before note-ons so repeated
    # pitches retrigger instead of becoming one continuous note.
    for tick, priority, pitch, event_velocity in sorted(events):
        message_type = "note_off" if priority == 0 else "note_on"
        track.append(
            mido.Message(
                message_type,
                note=pitch,
                velocity=event_velocity,
                time=tick - previous_tick,
            )
        )
        previous_tick = tick

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    return output


def onset_prediction_horizon_seconds(
    threshold: float,
    target_radius_seconds: float,
    onset_mode: str = "first_threshold",
) -> float:
    """Return the future scheduling horizon implied by onset decoding."""

    if onset_mode not in {"first_threshold", "peak"}:
        raise ValueError(f"unsupported onset mode: {onset_mode}")
    if onset_mode == "peak":
        return 0.0
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "first_threshold requires threshold between 0 and 1 inclusive"
        )
    if target_radius_seconds < 0.0:
        raise ValueError(
            "first_threshold requires non-negative target_radius_seconds"
        )
    return target_radius_seconds * (1.0 - threshold)


def validate_onset_target_config(cfg: DictConfig, onset_mode: str) -> None:
    """Reject first-threshold compensation for incompatible targets."""

    if onset_mode not in {"first_threshold", "peak"}:
        raise ValueError(f"unsupported onset mode: {onset_mode}")
    if onset_mode == "peak":
        return

    target_generator = cfg.dataset.target_generator
    softener_target = str(target_generator.softener._target_)
    if (
        str(target_generator.target_mode) != "onset"
        or bool(target_generator.get("velocity_target", False))
        or softener_target != "airpiano.data.targets.TriangularSoftener"
    ):
        raise ValueError(
            "first_threshold onset mode requires an unscaled triangular "
            "onset target (target_mode=onset, velocity_target=false, "
            "softener=TriangularSoftener)"
        )


def decode_chord_onsets(
    artifact: InferenceArtifact,
    threshold: float,
    onset_mode: str = "first_threshold",
    target_radius_seconds: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-slot scheduled onset frames and selection probabilities.

    Undetected and padded chord slots receive frame -1 and probability NaN.
    Threshold comparisons are inclusive. Peak ties select the earliest valid
    frame. First-threshold detection adds the radius-derived future horizon;
    peak timing remains unshifted.
    """

    horizon_seconds = onset_prediction_horizon_seconds(
        threshold, target_radius_seconds, onset_mode
    )
    horizon_frames = int(round(horizon_seconds * artifact.fps))

    onset_frames = np.full(len(artifact.chord_mask), -1, dtype=np.int64)
    onset_probabilities = np.full(
        len(artifact.chord_mask), np.nan, dtype=np.float64
    )
    valid_frames = np.flatnonzero(artifact.frame_mask)
    if valid_frames.size == 0:
        return onset_frames, onset_probabilities

    for chord_slot in np.flatnonzero(artifact.chord_mask):
        chord_slot = int(chord_slot)
        scores = artifact.probability[valid_frames, chord_slot]
        if onset_mode == "first_threshold":
            selected = next(
                (
                    index for index, score in enumerate(scores)
                    if score >= threshold
                ),
                None,
            )
            if selected is None:
                continue
        else:
            selected = int(np.argmax(scores))
            if scores[selected] < threshold:
                continue
        onset_frames[chord_slot] = (
            int(valid_frames[selected]) + horizon_frames
        )
        onset_probabilities[chord_slot] = float(scores[selected])
    return onset_frames, onset_probabilities


def sonify_artifact(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    duration_seconds: float = 1.0,
    velocity: int = 64,
    onset_mode: str = "first_threshold",
    target_radius_seconds: float = 0.2,
) -> Path:
    """Emit every pitch in each confidently aligned chord as a MIDI note.

    First-threshold notes use the radius-compensated future frame; peak notes
    use the earliest global maximum. Threshold equality is included, and all
    notes use the requested fixed duration and velocity. Real-time artifacts
    instead replay their recorded release times, duration, and velocity.
    """

    artifact = load_artifact(input_path)
    ticks_per_beat = 480
    tempo = 500_000
    events: list[tuple[int, int, int, int]] = []
    if artifact.release_time is not None:
        onset_times = artifact.release_time
        duration_seconds = artifact.note_duration
        velocity = artifact.note_velocity
    else:
        onset_frames, _ = decode_chord_onsets(
            artifact,
            threshold,
            onset_mode=onset_mode,
            target_radius_seconds=target_radius_seconds,
        )
        onset_times = onset_frames / artifact.fps
        onset_times[onset_frames < 0] = np.nan

    for chord in np.flatnonzero(artifact.chord_mask):
        onset_seconds = float(onset_times[chord])
        if np.isnan(onset_seconds):
            continue

        start_tick = _seconds_to_tick(
            onset_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        end_tick = _seconds_to_tick(
            onset_seconds + duration_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        for pitch in np.flatnonzero(artifact.chord_pitches[chord]):
            events.append((start_tick, 1, int(pitch), velocity))
            events.append((end_tick, 0, int(pitch), 0))

    return _write_note_events(events, output_path)


def calculate_onset_deltas(
    input_path: str | Path,
    midi_path: str | Path,
    threshold: float = 0.5,
    chord_tolerance_seconds: float = 0.05,
    onset_mode: str = "first_threshold",
    target_radius_seconds: float = 0.2,
) -> dict[str, object]:
    """Compare target peaks with decoded onsets or recorded MIDI releases."""

    del midi_path, chord_tolerance_seconds
    artifact = load_artifact(input_path)
    valid_frames = np.flatnonzero(artifact.frame_mask)
    recorded_releases = artifact.release_time is not None
    if recorded_releases:
        onset_times = artifact.release_time
        onset_mode = "release_time"
        prediction_horizon_seconds = 0.0
        target_radius_seconds = 0.0
        threshold = 0.5
    else:
        onset_frames, onset_probabilities = decode_chord_onsets(
            artifact,
            threshold,
            onset_mode=onset_mode,
            target_radius_seconds=target_radius_seconds,
        )
        onset_times = (artifact.frame_start + onset_frames) / artifact.fps
        onset_times[onset_frames < 0] = np.nan
        prediction_horizon_seconds = onset_prediction_horizon_seconds(
            threshold, target_radius_seconds, onset_mode
        )
    chords = []

    for chord_slot in np.flatnonzero(artifact.chord_mask):
        chord_slot = int(chord_slot)
        source_chord_id = int(artifact.chord_ids[chord_slot])
        target_frame = int(
            valid_frames[
                np.argmax(artifact.target[valid_frames, chord_slot])
            ]
        )
        real_onset = (
            artifact.frame_start + target_frame
        ) / artifact.fps
        detected = not np.isnan(onset_times[chord_slot])
        predicted_onset = (
            float(onset_times[chord_slot]) if detected else None
        )
        delta = (
            float(predicted_onset - real_onset)
            if predicted_onset is not None
            else None
        )
        chord_report = {
            "absolute_delta_seconds": abs(delta) if delta is not None else None,
            "chord_id": chord_slot,
            "delta_seconds": delta,
            "detected": detected,
            "onset_probability": (
                float(onset_probabilities[chord_slot])
                if detected and not recorded_releases
                else None
            ),
            "predicted_onset_seconds": predicted_onset,
            "real_onset_seconds": real_onset,
            "source_chord_id": source_chord_id,
        }
        if recorded_releases:
            chord_report["estimated_onset_seconds"] = (
                float(artifact.predicted_onset[chord_slot])
                if detected else None
            )
        chords.append(chord_report)

    report = {
        "chords": chords,
        "fps": artifact.fps,
        "frame_start": artifact.frame_start,
        "onset_mode": onset_mode,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "record_id": artifact.record_id,
        "target_radius_seconds": target_radius_seconds,
        "threshold": threshold,
    }
    if recorded_releases:
        report["onset_prediction_offset_seconds"] = (
            artifact.onset_prediction_offset
        )
    return report


def write_onset_delta_summary(
    reports: Iterable[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Average slice deltas for each chunk-relative chord ID."""

    observation_counts: dict[int, int] = {}
    detected_counts: dict[int, int] = {}
    deltas: dict[int, list[float]] = {}
    absolute_deltas: dict[int, list[float]] = {}
    artifact_count = 0
    onset_mode = None
    prediction_horizon_seconds = None
    target_radius_seconds = None
    threshold = None

    for report in reports:
        artifact_count += 1
        if onset_mode is None:
            onset_mode = str(report["onset_mode"])
            prediction_horizon_seconds = float(
                report["prediction_horizon_seconds"]
            )
            target_radius_seconds = float(report["target_radius_seconds"])
            threshold = float(report["threshold"])
        for chord in report["chords"]:
            chord_id = int(chord["chord_id"])
            observation_counts[chord_id] = (
                observation_counts.get(chord_id, 0) + 1
            )
            if not chord["detected"]:
                continue
            detected_counts[chord_id] = detected_counts.get(chord_id, 0) + 1
            deltas.setdefault(chord_id, []).append(float(chord["delta_seconds"]))
            absolute_deltas.setdefault(chord_id, []).append(
                float(chord["absolute_delta_seconds"])
            )

    chords = []
    for chord_id in sorted(observation_counts):
        detected_count = detected_counts.get(chord_id, 0)
        mean_delta = (
            sum(deltas[chord_id]) / detected_count if detected_count else None
        )
        mean_absolute_delta = (
            sum(absolute_deltas[chord_id]) / detected_count
            if detected_count
            else None
        )
        observation_count = observation_counts[chord_id]
        chords.append(
            {
                "chord_id": chord_id,
                "detected_count": detected_count,
                "mean_absolute_delta_seconds": mean_absolute_delta,
                "mean_delta_seconds": mean_delta,
                "missed_count": observation_count - detected_count,
                "observation_count": observation_count,
            }
        )

    output = resolve_path(output_path)
    write_json(
        output,
        {
            "artifact_count": artifact_count,
            "chords": chords,
            "delta_definition": (
                "predicted_onset_seconds - real_onset_seconds"
            ),
            "onset_mode": onset_mode,
            "prediction_horizon_seconds": prediction_horizon_seconds,
            "target_radius_seconds": target_radius_seconds,
            "threshold": threshold,
        },
    )
    return output


def plot_onset_delta_summary(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Draw mean absolute onset delta and misses by relative chord ID."""

    report = json.loads(resolve_path(input_path).read_text(encoding="utf-8"))
    chords = report["chords"]
    chord_ids = [int(chord["chord_id"]) for chord in chords]
    detected = [
        chord for chord in chords if chord["mean_absolute_delta_seconds"] is not None
    ]
    detected_ids = [int(chord["chord_id"]) for chord in detected]
    mean_absolute_deltas = [
        float(chord["mean_absolute_delta_seconds"]) for chord in detected
    ]
    missed_counts = [int(chord["missed_count"]) for chord in chords]

    import os
    import tempfile

    matplotlib_config = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = max(8.0, min(18.0, len(chord_ids) * 0.5))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(width, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].bar(detected_ids, mean_absolute_deltas, color="#4c72b0")
    axes[0].set_ylabel("Mean absolute delta (seconds)")
    axes[0].set_title("Mean absolute onset delta by relative chord ID")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(chord_ids, missed_counts, color="#c44e52")
    axes[1].set_xticks(chord_ids)
    axes[1].set_xlabel("Relative chord ID")
    axes[1].set_ylabel("Missing count")
    axes[1].set_title("Missing predictions by relative chord ID")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(axis="y", alpha=0.25)

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def sonify_ground_truth(
    input_path: str | Path,
    midi_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write source notes whose onsets fall within the artifact's real frames."""

    artifact = load_artifact(input_path)
    slice_start = artifact.frame_start / artifact.fps
    slice_end = (
        artifact.frame_start + int(artifact.frame_mask.sum())
    ) / artifact.fps
    ticks_per_beat = 480
    tempo = 500_000
    events: list[tuple[int, int, int, int]] = []

    for note in MidoNoteReader().read(midi_path):
        if note.onset < slice_start or note.onset >= slice_end:
            continue
        onset = note.onset - slice_start
        offset = min(note.offset, slice_end) - slice_start
        events.append(
            (
                _seconds_to_tick(
                    onset,
                    ticks_per_beat=ticks_per_beat,
                    tempo=tempo,
                ),
                1,
                note.pitch,
                note.velocity,
            )
        )
        events.append(
            (
                _seconds_to_tick(
                    offset,
                    ticks_per_beat=ticks_per_beat,
                    tempo=tempo,
                ),
                0,
                note.pitch,
                0,
            )
        )

    return _write_note_events(events, output_path)
