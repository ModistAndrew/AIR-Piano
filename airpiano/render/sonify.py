"""Decode an inference artifact into a simple MIDI performance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..data.midi import MidoNoteReader
from ..utils import resolve_output_path
from .artifact import load_artifact


def sonify(cfg: DictConfig) -> None:
    """Write prediction and source-MIDI excerpts for one inference slice."""

    input_path = resolve_output_path(cfg, cfg.sonify.input)
    prediction_output = sonify_artifact(
        input_path,
        resolve_output_path(cfg, cfg.sonify.output),
        threshold=float(cfg.sonify.threshold),
        duration_seconds=float(cfg.sonify.duration_seconds),
        velocity=int(cfg.sonify.velocity),
    )
    artifact = load_artifact(input_path)
    midi_root = Path(str(cfg.sonify.midi_root)).expanduser().resolve()
    midi_path = midi_root / f"{artifact.record_id}.mid"
    ground_truth_output = sonify_ground_truth(
        input_path,
        midi_path,
        resolve_output_path(cfg, cfg.sonify.ground_truth_output),
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

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    return output


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
        scores = artifact.probability[valid_frames, chord]
        peak_score = float(scores.max())
        if peak_score < threshold:
            continue

        # valid_frames is sorted and argmax returns the first occurrence of a tie.
        peak_frame = int(valid_frames[int(np.argmax(scores))])
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
