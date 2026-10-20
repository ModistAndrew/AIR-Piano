"""Prepare split inputs for window testing and real-time inference.

Reuse canonical hands, or build them from raw WiLoR caches or source videos.
Generate missing beats from MIDI metadata, then delegate song loading to data.files.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import tempfile
from typing import Any

import mido
from hydra.utils import instantiate
from omegaconf import DictConfig

from . import data
from .data.files import read_record_stems
from .preprocessing.canonical import raw_cache_has_markers
from .utils import resolve_path


def _midi_beats(midi_path: Path) -> list[tuple[float, int]]:
    """Return MIDI beat times and bar positions, including a boundary at the end."""
    midi = mido.MidiFile(midi_path)
    tempo = 500_000
    beat_ticks = midi.ticks_per_beat
    beats_per_bar = 4
    beat_number = 1
    next_beat_tick = 0
    tick = 0
    seconds = 0.0
    beats = []

    for message in midi.merged_track:
        event_tick = tick + message.time
        # Defer a boundary at the event tick until all simultaneous metadata
        # has been applied, so meter changes cannot create duplicate rows.
        while next_beat_tick < event_tick:
            beat_time = seconds + mido.tick2second(
                next_beat_tick - tick, midi.ticks_per_beat, tempo
            )
            beats.append((beat_time, beat_number))
            next_beat_tick += beat_ticks
            beat_number = beat_number % beats_per_bar + 1

        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        tick = event_tick
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "time_signature":
            compound = message.numerator > 3 and message.numerator % 3 == 0
            pulse = 3 if compound else 1
            beat_ticks = midi.ticks_per_beat * 4 * pulse / message.denominator
            beats_per_bar = message.numerator // pulse
            beat_number = 1
            next_beat_tick = tick

    # Cover end-of-track, even when it extends beyond the last note. Very short
    # scores still need two rows for the reader's subdivision interpolation.
    beat_time = seconds + mido.tick2second(
        next_beat_tick - tick, midi.ticks_per_beat, tempo
    )
    beats.append((beat_time, beat_number))
    if len(beats) < 2:
        beats.append((
            beat_time + mido.tick2second(beat_ticks, midi.ticks_per_beat, tempo),
            beat_number % beats_per_bar + 1,
        ))
    return beats


def iter_split_songs(
    cfg: DictConfig,
    split_file: str | Path,
    *,
    label: str,
    components: dict[str, Any] | None = None,
) -> Iterator[tuple[int, int, data.Song, dict[str, Any]]]:
    """Yield prepared songs one at a time through configured modality roots.

    Existing preprocessing caches and beat files are reused. Missing beats are
    derived from MIDI only here; training still requires existing beat files.
    After preparation, use the same FilePair reader as FileLoader.load.
    """

    split_path = resolve_path(split_file)
    records = read_record_stems([split_path])
    if not records:
        print(f"[{label}] no records in {split_path}", flush=True)
        return

    video_root = resolve_path(cfg.data.video_root)
    midi_root = resolve_path(cfg.data.midi_root)
    raw_root = resolve_path(cfg.data.raw_wilor_root)
    canonical_root = resolve_path(cfg.data.hand_root)
    beat_root = resolve_path(cfg.data.beat_root)
    video_to_ragged = None
    ragged_to_canonical = instantiate(cfg.preprocess.ragged_to_canonical)
    components = components or data.build_data_components(cfg)
    for index, record_id in enumerate(records, 1):
        video_path = video_root / f"{record_id}.mp4"
        midi_path = midi_root / f"{record_id}.mid"
        if not midi_path.is_file():
            print(f"[{label} skip] {record_id}: missing MIDI {midi_path}")
            continue

        beat_path = beat_root / f"{record_id}.txt"

        print(
            f"{label} {index}/{len(records)}: prepare {record_id}",
            flush=True,
        )
        raw_dir = raw_root / record_id
        canonical_path = canonical_root / f"{record_id}.npz"
        if not canonical_path.exists():
            if not raw_cache_has_markers(raw_dir):
                if not video_path.is_file():
                    print(
                        f"[{label} skip] {record_id}: missing video {video_path}",
                        flush=True,
                    )
                    continue
                if video_to_ragged is None:
                    video_to_ragged = instantiate(cfg.preprocess.video_to_ragged)
                video_to_ragged.run(video_path, raw_dir)
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
            )

        if not beat_path.is_file():
            beats = _midi_beats(midi_path)
            beat_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{beat_path.name}.", suffix=".tmp", dir=beat_path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.writelines(
                        f"{time:.9f} {number}\n" for time, number in beats
                    )
                os.replace(temporary, beat_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            print(f"[{label}] generated MIDI beats: {beat_path}", flush=True)

        pair = data.FilePair(
            record_id=record_id,
            midi_path=midi_path,
            hand_path=canonical_path,
            beat_path=beat_path,
        )
        song = components["file_loader"].reader.read(
            pair,
            components["midi_loader"],
            components["hand_loader"],
        )
        yield index, len(records), song, components


__all__ = ["iter_split_songs"]
