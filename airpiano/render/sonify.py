"""Decode an inference artifact into a simple MIDI performance."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..data.midi import MidoNoteReader
from ..utils import resolve_path, write_json
from .artifact import load_artifact


def sonify(cfg: DictConfig) -> None:
    """Recursively sonify every inference artifact below the configured root."""

    root = resolve_path(cfg.sonify.root)
    midi_root = resolve_path(cfg.sonify.midi_root)
    onset_reports = []
    for input_path in sorted(root.rglob("*.npz")):
        if not input_path.is_file():
            continue
        prediction_output = sonify_artifact(
            input_path,
            input_path.with_suffix(".mid"),
            threshold=float(cfg.sonify.threshold),
            duration_seconds=float(cfg.sonify.duration_seconds),
            velocity=int(cfg.sonify.velocity),
        )
        artifact = load_artifact(input_path)
        midi_path = midi_root / f"{artifact.record_id}.mid"
        ground_truth_output = sonify_ground_truth(
            input_path,
            midi_path,
            input_path.with_suffix(".gt.mid"),
        )
        onset_report = calculate_onset_deltas(
            input_path,
            midi_path,
            threshold=float(cfg.sonify.threshold),
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


def _chord_peak(
    probability: np.ndarray,
    valid_frames: np.ndarray,
    chord_slot: int,
) -> tuple[int, float]:
    scores = probability[valid_frames, chord_slot]
    peak_score = float(scores.max())
    # valid_frames is sorted and argmax returns the first occurrence of a tie.
    peak_frame = int(valid_frames[int(np.argmax(scores))])
    return peak_frame, peak_score


def sonify_artifact(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.2,
    duration_seconds: float = 1.0,
    velocity: int = 64,
) -> Path:
    """Emit every pitch in each confidently aligned chord as a MIDI note.

    Each valid chord is placed at its earliest maximum over real frames. A peak
    equal to ``threshold`` is included, and all notes use the requested fixed
    duration and velocity.
    """

    artifact = load_artifact(input_path)
    ticks_per_beat = 480
    tempo = 500_000
    events: list[tuple[int, int, int, int]] = []
    valid_frames = np.flatnonzero(artifact.frame_mask)

    for chord in np.flatnonzero(artifact.chord_mask):
        if valid_frames.size == 0:
            break
        peak_frame, peak_score = _chord_peak(
            artifact.probability,
            valid_frames,
            int(chord),
        )
        if peak_score < threshold:
            continue

        onset_seconds = peak_frame / artifact.fps
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
    threshold: float = 0.2,
    chord_tolerance_seconds: float = 0.05,
) -> dict[str, object]:
    """Calculate frame-level target-to-prediction deltas for valid chords."""

    del midi_path, chord_tolerance_seconds
    artifact = load_artifact(input_path)
    valid_frames = np.flatnonzero(artifact.frame_mask)
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
        peak_frame, peak_score = _chord_peak(
            artifact.probability,
            valid_frames,
            chord_slot,
        )
        detected = peak_score >= threshold
        predicted_onset = (
            (artifact.frame_start + peak_frame) / artifact.fps
            if detected
            else None
        )
        delta = (
            float(predicted_onset - real_onset)
            if predicted_onset is not None
            else None
        )
        chords.append(
            {
                "absolute_delta_seconds": abs(delta) if delta is not None else None,
                "chord_id": chord_slot,
                "delta_seconds": delta,
                "detected": detected,
                "peak_probability": peak_score,
                "predicted_onset_seconds": predicted_onset,
                "real_onset_seconds": real_onset,
                "source_chord_id": source_chord_id,
            }
        )

    return {
        "chords": chords,
        "fps": artifact.fps,
        "frame_start": artifact.frame_start,
        "record_id": artifact.record_id,
        "threshold": threshold,
    }


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

    for report in reports:
        artifact_count += 1
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
