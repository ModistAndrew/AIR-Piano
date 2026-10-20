"""Replay saved real-time chord releases and physical ground truth as MIDI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..data.midi import MidoNoteReader
from ..utils import resolve_path
from .artifact import load_artifact


def sonify(cfg: DictConfig) -> None:
    """Write prediction and ground-truth MIDI for each real-time artifact."""

    root = resolve_path(cfg.sonify.root)
    midi_root = resolve_path(cfg.sonify.midi_root)
    for input_path in sorted(root.rglob("*.npz")):
        if not input_path.is_file():
            continue
        artifact = load_artifact(input_path, require_releases=True)
        prediction_output = sonify_artifact(
            input_path, input_path.with_suffix(".mid")
        )
        midi_path = midi_root / f"{artifact.record_id}.mid"
        ground_truth_output = sonify_ground_truth(
            input_path, midi_path, input_path.with_suffix(".gt.mid")
        )
        print(f"wrote prediction {prediction_output}", flush=True)
        print(f"wrote ground truth {ground_truth_output}", flush=True)


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


def sonify_artifact(input_path: str | Path, output_path: str | Path) -> Path:
    """Replay recorded chord releases with their saved duration and velocity."""

    artifact = load_artifact(input_path, require_releases=True)
    ticks_per_beat = 480
    tempo = 500_000
    events: list[tuple[int, int, int, int]] = []
    for chord in np.flatnonzero(artifact.chord_mask):
        onset_seconds = float(artifact.release_time[chord])
        if np.isnan(onset_seconds):
            continue

        start_tick = _seconds_to_tick(
            onset_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        end_tick = _seconds_to_tick(
            onset_seconds + artifact.note_duration,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        for pitch in np.flatnonzero(artifact.chord_pitches[chord]):
            events.append((start_tick, 1, int(pitch), artifact.note_velocity))
            events.append((end_tick, 0, int(pitch), 0))

    return _write_note_events(events, output_path)


def sonify_ground_truth(
    input_path: str | Path,
    midi_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write source notes whose onsets fall within the artifact's real frames."""

    artifact = load_artifact(input_path, require_releases=True)
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
