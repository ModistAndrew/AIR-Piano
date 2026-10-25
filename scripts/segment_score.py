"""Export physical score events and matching alignment images for one slice."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mido
import numpy as np

from airpiano.data.types import Chord, Song
from airpiano.render.artifact import load_artifact


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_physical_midi(notes: list[dict[str, Any]], duration: float, path: Path) -> None:
    """Keep physical seconds to within one 1/60000-second MIDI tick."""

    midi = mido.MidiFile(ticks_per_beat=30_000)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    end_tick = round(duration * 60_000)
    events = []
    # Separate simultaneous voices of a repeated pitch so nested note durations
    # remain unambiguous; channel 9 is reserved for percussion in General MIDI.
    channels = tuple(channel for channel in range(16) if channel != 9)
    voice_ends: dict[int, dict[int, int]] = {}
    for note in sorted(notes, key=lambda item: (item["onset"], item["pitch"], item["id"])):
        start = min(round(note["onset"] * 60_000), end_tick - 1)
        stop = min(end_tick, max(start + 1, round(note["offset"] * 60_000)))
        pitch = note["pitch"]
        active = voice_ends.setdefault(pitch, {})
        channel = next((value for value in channels if active.get(value, -1) <= start), None)
        if channel is None:
            raise ValueError(f"more than 15 overlapping physical notes at pitch {pitch}")
        active[channel] = stop
        events.append((start, 1, pitch, channel, note["velocity"]))
        events.append((stop, 0, pitch, channel, 0))
    previous = 0
    for tick, priority, pitch, channel, velocity in sorted(events):
        track.append(mido.Message(
            "note_on" if priority else "note_off", note=pitch,
            velocity=velocity, channel=channel, time=tick - previous,
        ))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - previous)))
    midi.save(path)


def _plot_notes(notes: list[dict[str, Any]], duration: float, path: Path) -> None:
    plt = _pyplot()
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    if notes:
        patches = [Rectangle((note["onset"], note["pitch"] - 0.4),
                             note["offset"] - note["onset"], 0.8) for note in notes]
        collection = PatchCollection(patches, cmap="viridis", edgecolor="none")
        collection.set_array(np.asarray([note["velocity"] for note in notes]))
        collection.set_clim(0, 127)
        axis.add_collection(collection)
        axis.set_ylim(min(note["pitch"] for note in notes) - 2,
                      max(note["pitch"] for note in notes) + 2)
        figure.colorbar(collection, ax=axis, label="MIDI velocity", pad=0.01)
    else:
        axis.set_ylim(20, 109)
    axis.set_xlim(0, duration)
    axis.set_xlabel("Time in slice (seconds)")
    axis.set_ylabel("MIDI pitch")
    axis.set_title("Ground-truth physical MIDI notes")
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _export_beats(beat_path: Path, start: float, stop: float, output_dir: Path) -> dict:
    values = np.loadtxt(beat_path, ndmin=2)
    if values.shape[1] not in (1, 2):
        raise ValueError("beat rows must contain one or two numbers")
    indices = np.flatnonzero((values[:, 0] >= start) & (values[:, 0] < stop))
    rows = []
    for index in indices:
        position = float(values[index, 1]) if values.shape[1] == 2 else None
        rows.append({"source_index": int(index), "source_seconds": float(values[index, 0]),
                     "local_seconds": float(values[index, 0] - start),
                     "source_position": position, "downbeat": position == 1})
    with (output_dir / "beats.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Cropped source markers; full source grid defines score quantization.\n")
        handle.write("# One-column source files contain subdivision ticks, without downbeat labels.\n")
        handle.write("source_index\tsource_seconds\tlocal_seconds\tsource_position\tdownbeat\n")
        for row in rows:
            position = "NA" if row["source_position"] is None else f'{row["source_position"]:g}'
            handle.write(f'{row["source_index"]}\t{row["source_seconds"]:.9f}\t'
                         f'{row["local_seconds"]:.9f}\t{position}\t{int(row["downbeat"])}\n')
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(14, 2.8), constrained_layout=True)
    for row in rows:
        downbeat = row["downbeat"]
        axis.vlines(row["local_seconds"], 0, 1 if downbeat else 0.55,
                    color="#d62728" if downbeat else "#1976d2",
                    linewidth=2.6 if downbeat else 1.5)
        if row["source_position"] is not None:
            axis.text(row["local_seconds"], 1.06 if downbeat else 0.61,
                      f'{row["source_position"]:g}', ha="center", fontsize=9)
    axis.set(xlim=(0, stop - start), ylim=(-0.05, 1.3), yticks=[],
             xlabel="Time in slice (seconds)",
             title="Source beat markers (red = downbeat; numbers = source beat position)")
    figure.savefig(output_dir / "beats.png", dpi=180, facecolor="white")
    plt.close(figure)
    return {"marker_count": len(rows), "downbeat_count": sum(row["downbeat"] for row in rows),
            "source_columns": values.shape[1], "markers": rows}


def export_score_bundle(
    *, song: Song, chords: tuple[Chord, ...], sample: dict,
    beat_path: Path, output_dir: Path,
) -> dict:
    """Export clipped physical notes, source beats, and exact model score tokens.

    All exported event times use the sample's unaugmented frame clock. Physical
    note overlaps are retained even when their chord falls outside the slice.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_start = int(sample["frame_start"])
    frame_count = len(sample["hands"])
    fps = float(sample["fps"])
    start, stop = frame_start / fps, (frame_start + frame_count) / fps
    if stop <= start:
        raise ValueError("score export requires a nonempty frame interval")
    tokens = _array(sample["chords"])
    chord_ids = _array(sample["chord_ids"])
    if tokens.ndim != 2 or tokens.shape[0] != len(chords) or tokens.shape[1] <= 128:
        raise ValueError("sample tokens must match the requested chords")
    if not np.array_equal(chord_ids, [chord.id for chord in chords]):
        raise ValueError("sample chord IDs must preserve the requested chord order")
    duration_bins = tokens.shape[1] - 128
    notes = [{"id": note.id, "onset": max(note.onset, start) - start,
              "offset": min(note.offset, stop) - start, "pitch": note.pitch,
              "velocity": note.velocity, "source_onset": note.onset,
              "source_offset": note.offset}
             for note in song.notes if note.onset < stop and note.offset > start]
    _write_physical_midi(notes, stop - start, output_dir / "gt_notes.mid")
    _plot_notes(notes, stop - start, output_dir / "gt_notes.png")
    beats = _export_beats(Path(beat_path), start, stop, output_dir)

    with (output_dir / "score_tokens.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"# Exact sample token vectors: {len(chords)} rows x {tokens.shape[1]} values.\n")
        handle.write(f"# Indices 0..127: MIDI pitch multi-hot; 128..{tokens.shape[1]-1}: "
                     f"one-hot duration ticks clamped to [1,{duration_bins}].\n")
        handle.write("slot\tchord_id\tpitches\tduration_ticks\tduration_bin\tactive_indices\ttoken\n")
        for slot, (chord, token) in enumerate(zip(chords, tokens)):
            pitches = sorted({note.pitch for note in chord.onset_notes})
            active = np.flatnonzero(token).tolist()
            vector = "[" + ",".join(format(float(value), ".9g") for value in token) + "]"
            handle.write(f"{slot}\t{chord.id}\t{json.dumps(pitches)}\t{chord.duration_ticks}\t"
                         f"{min(max(chord.duration_ticks, 1), duration_bins)}\t"
                         f"{json.dumps(active)}\t{vector}\n")
    with (output_dir / "chord_onsets.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Mean physical onsets are averaged over constituent note onsets in seconds.\n")
        handle.write("# Local seconds subtract frame_start/fps; beat ticks do not replace physical time.\n")
        handle.write("slot\tchord_id\tmean_source_seconds\tmean_local_seconds\tscore_tick\t"
                     "duration_ticks\tnote_ids\tpitches\tnote_source_onsets\tnote_local_onsets\n")
        for slot, chord in enumerate(chords):
            fields = [str(slot), str(chord.id), f"{chord.time:.12f}", f"{chord.time-start:.12f}",
                      str(chord.time_ticks), str(chord.duration_ticks),
                      json.dumps([note.id for note in chord.onset_notes]),
                      json.dumps([note.pitch for note in chord.onset_notes]),
                      json.dumps([note.onset for note in chord.onset_notes]),
                      json.dumps([note.onset - start for note in chord.onset_notes])]
            handle.write("\t".join(fields) + "\n")
    return {"frame_start": frame_start, "frame_count": frame_count, "fps": fps,
            "start_seconds": start, "stop_seconds": stop, "duration_seconds": stop - start,
            "physical_note_count": len(notes),
            "left_clipped_note_count": sum(note["source_onset"] < start for note in notes),
            "right_clipped_note_count": sum(note["source_offset"] > stop for note in notes),
            "midi_ticks_per_second": 60_000,
            "chord_count": len(chords), "token_shape": list(tokens.shape), "beats": beats}


def render_alignment_images(artifact_path: Path, output_dir: Path) -> dict:
    """Save separate target and prediction panels with identical axes and scale."""

    artifact = load_artifact(artifact_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    frames, chords = artifact.probability.shape
    if frames == 0 or chords == 0:
        raise ValueError("alignment images require nonempty frame and chord axes")
    support = artifact.support.T
    cmap = plt.colormaps["magma"].with_extremes(bad="#d5d8dc")
    width = max(8.0, min(18.0, frames / 15.0))
    ticks = np.unique(np.linspace(0, frames - 1, min(9, frames), dtype=int))
    for values, title, filename in (
        (artifact.target, "Alignment target", "gt_alignment.png"),
        (artifact.probability, "Prediction probability", "predicted_alignment.png"),
    ):
        # Fixed subplot positions keep pixels, axes, and colorbar identical
        # between panels even when title strings differ.
        figure = plt.figure(figsize=(width, 5.5))
        axis = figure.add_axes([0.07, 0.13, 0.84, 0.78])
        color_axis = figure.add_axes([0.93, 0.13, 0.015, 0.78])
        image = axis.imshow(np.ma.masked_where(~support, values.T),
                            aspect="auto", interpolation="nearest", origin="lower",
                            cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_ylabel("Chord slot")
        axis.set_xlabel("Time in slice (seconds)")
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{frame / artifact.fps:.2f}" for frame in ticks])
        figure.colorbar(image, cax=color_axis)
        figure.savefig(output_dir / filename, dpi=180, facecolor="white")
        plt.close(figure)
    return {"shape": [frames, chords], "valid_cells": int(artifact.support.sum()),
            "fps": artifact.fps, "frame_start": artifact.frame_start,
            "colormap": "magma", "color_range": [0.0, 1.0], "dpi": 180}
