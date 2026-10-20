"""Simulated frame-by-frame inference with full-score DP and MIDI releases."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from . import data
from .inference import load_model_checkpoint, write_inference_artifact
from .model.loss import alignment_probabilities, alignment_probability_loss
from .model.transformer import AirPianoTransformer
from .render.sonify import sonify_artifact
from .song_input import iter_split_songs
from .train import _autocast, _forward
from .utils import resolve_path, resolve_split_output_dir


@dataclass
class RuntimeChord:
    """One score occurrence; released means its MIDI note-on was emitted."""

    id: int
    pitches: tuple[int, ...]
    time_ticks: int
    triggered: bool = False
    released: bool = False
    anchor_time: float | None = None


@dataclass(frozen=True)
class Event:
    id: int
    time: float
    score: float


class RealTimeController:
    """Maintain occurrence latches, full-score alignment, and release history."""

    processing_latency = 0.050
    note_duration = 1.0

    def __init__(
        self,
        score_chords: Sequence[data.Chord],
        *,
        window_size: int = 50,
        fps: float = 30.0,
        target_radius_seconds: float = 0.2,
    ) -> None:
        self.onset_prediction_offset = float(target_radius_seconds) * (1.0 - 0.5)
        self.window_size = int(window_size)
        self.fps = float(fps)
        self.chord_count = len(score_chords)
        if self.chord_count < self.window_size:
            raise ValueError(
                f"full_song_real_time requires at least {self.window_size} chords"
            )
        positions = [chord.time_ticks for chord in score_chords]
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError(
                "full_song_real_time requires strictly increasing score positions"
            )

        # The sentinel keeps all runtime lookups in 1-based occurrence IDs.
        self.chords = [None] + [
            RuntimeChord(
                id=index + 1,
                pitches=tuple(sorted({note.pitch for note in chord.onset_notes})),
                time_ticks=positions[index],
            )
            for index, chord in enumerate(score_chords)
        ]
        self.chord_window_start = 1
        self.video_history_start = 0
        self.prev = np.zeros(self.chord_count + 1)
        self.cur = np.zeros(self.chord_count + 1)
        self.best_score = 0.0
        self.pending_releases: dict[int, float] = {}
        self.event_records: list[dict[str, Any]] = []
        self.release_records: list[dict[str, Any]] = []

    def collect_events(
        self, scores: Sequence[float], window_start: int, frame_index: int
    ) -> list[Event]:
        """Latch all observations using the window saved before this forward."""

        events = []
        event_time = frame_index / self.fps + self.onset_prediction_offset
        for slot, value in enumerate(scores):
            score = float(value)
            chord = self.chords[window_start + slot]
            if score > 0.5 and not chord.triggered:
                chord.triggered = True
                events.append(Event(chord.id, event_time, score))
            elif score < 0.5:
                chord.triggered = False
        return events

    def process_event(self, event: Event, frame_index: int) -> bool:
        """Advance one full DP column, then update the window and history."""

        old_best = self.best_score
        match_score = self.prev[event.id - 1] + 1
        self.cur[0] = 0
        for index in range(1, self.chord_count + 1):
            self.cur[index] = max(self.prev[index], self.cur[index - 1] - 1)
            if index == event.id:
                self.cur[index] = max(self.cur[index], match_score)
        success = bool(match_score > old_best)
        if success:
            self.best_score = float(match_score)
        self.prev, self.cur = self.cur, self.prev

        chord = self.chords[event.id]
        if not chord.released:
            self.pending_releases.setdefault(event.id, event.time)

        final_start = self.chord_count - self.window_size + 1
        if success:
            chord.anchor_time = event.time
            self.chord_window_start = max(
                1, min(event.id - self.window_size // 2, final_start)
            )
        else:
            self.chord_window_start = min(
                self.chord_window_start + 1, final_start
            )

        start_time = self.estimate_time(
            self.chords[self.chord_window_start].time_ticks
        )
        self.video_history_start = (
            0 if math.isinf(start_time)
            else max(0, min(math.floor(start_time * self.fps), frame_index))
        )
        self.event_records.append({
            "chord_id": event.id,
            "frame_index": frame_index,
            "time": event.time,
            "score": event.score,
            "match_score": float(match_score),
            "old_best": old_best,
            "best_score": self.best_score,
            "success": success,
            "chord_window_start": self.chord_window_start,
            "video_history_start": self.video_history_start,
        })
        return success

    def estimate_time(self, time_ticks: int) -> float:
        """Estimate time from stored anchors for video-history selection."""

        anchors = sorted(
            (chord.time_ticks, chord.anchor_time)
            for chord in self.chords[1:]
            if chord.anchor_time is not None
        )
        positions = [position for position, _ in anchors]
        right = bisect_left(positions, time_ticks)
        if right < len(anchors) and positions[right] == time_ticks:
            return anchors[right][1]
        if len(anchors) < 2:
            return math.inf
        right = max(1, min(right, len(anchors) - 1))
        s_a, t_a = anchors[right - 1]
        s_b, t_b = anchors[right]
        return t_a + (time_ticks - s_a) * (t_b - t_a) / (s_b - s_a)

    def release_due(self, frame_index: int) -> list[dict[str, Any]]:
        """Emit due triggers at this frame's simulated result time."""

        frame_time = frame_index / self.fps
        now = frame_time + self.processing_latency
        records = []
        for chord_id, target_time in list(self.pending_releases.items()):
            if target_time > now:
                continue
            chord = self.chords[chord_id]
            record = {
                "chord_id": chord.id,
                "pitches": chord.pitches,
                "frame_index": frame_index,
                "frame_time": frame_time,
                "predicted_onset": target_time,
                "release_time": now,
                "duration": self.note_duration,
            }
            self.release_records.append(record)
            records.append(record)
            chord.released = True
            del self.pending_releases[chord_id]
        return records


def _controller_arrays(controller: RealTimeController) -> dict[str, np.ndarray]:
    """Keep debug traces numeric so artifacts load with allow_pickle=False."""

    chords = controller.chords[1:]
    arrays = {
        "time_ticks": np.asarray([c.time_ticks for c in chords], dtype=np.int64),
        "anchor_time": np.asarray([
            np.nan if c.anchor_time is None else c.anchor_time for c in chords
        ]),
        "triggered": np.asarray([c.triggered for c in chords]),
        "released": np.asarray([c.released for c in chords]),
        "release_time": np.full(controller.chord_count, np.nan),
        "predicted_onset": np.full(controller.chord_count, np.nan),
        "release_frame_index": np.full(
            controller.chord_count, -1, dtype=np.int64
        ),
        "release_frame_time": np.full(controller.chord_count, np.nan),
        "note_duration": np.asarray(controller.note_duration),
        "note_velocity": np.asarray(64, dtype=np.int64),
        "processing_latency": np.asarray(controller.processing_latency),
        "onset_prediction_offset": np.asarray(
            controller.onset_prediction_offset
        ),
        "window_size": np.asarray(controller.window_size),
        "best_score": np.asarray(controller.best_score),
        "final_chord_window_start": np.asarray(controller.chord_window_start),
        "final_video_history_start": np.asarray(controller.video_history_start),
        "dp_prev": controller.prev,
        "dp_cur": controller.cur,
    }
    for record in controller.release_records:
        index = record["chord_id"] - 1
        arrays["release_time"][index] = record["release_time"]
        arrays["predicted_onset"][index] = record["predicted_onset"]
        arrays["release_frame_index"][index] = record["frame_index"]
        arrays["release_frame_time"][index] = record["frame_time"]
    for name, dtype in (
        ("chord_id", np.int64), ("frame_index", np.int64),
        ("time", np.float64), ("score", np.float64),
        ("match_score", np.float64), ("old_best", np.float64),
        ("best_score", np.float64), ("success", bool),
        ("chord_window_start", np.int64), ("video_history_start", np.int64),
    ):
        arrays[f"event_{name}"] = np.asarray(
            [record[name] for record in controller.event_records], dtype=dtype
        )
    return arrays


@torch.no_grad()
def infer_full_song_real_time(
    model: AirPianoTransformer,
    song: data.Song,
    slicer: data.FutureChordSlicer,
    target_generator: data.TargetGenerator,
    device: torch.device,
    output_path: str | Path,
    *,
    tokenizer: data.ChordTokenizer,
    fps: float,
    bf16: bool,
) -> RealTimeController:
    """Receive one frame at a time and write both the release MIDI and NPZ."""

    if song.fps != fps:
        raise ValueError("song and dataset frame rates must match")
    controller = RealTimeController(
        song.chords,
        window_size=slicer.segment_frames,
        fps=fps,
        target_radius_seconds=target_generator.softener.radius_seconds,
    )
    model.eval()
    frame_count = len(song.hands)
    chord_count = len(song.chords)
    chord_tokens = tokenizer.tokenize(song.chords)
    probability = np.zeros((frame_count, chord_count), dtype=np.float32)
    window_starts = np.empty(frame_count, dtype=np.int64)
    history_starts = np.empty(frame_count, dtype=np.int64)
    video_frames = []
    presence_frames = []

    for frame_index in range(frame_count):
        video_frames.append(song.hands[frame_index])
        presence_frames.append(song.presence[frame_index])
        window_start = controller.chord_window_start
        history_start = controller.video_history_start
        window_starts[frame_index] = window_start
        history_starts[frame_index] = history_start
        start = window_start - 1
        stop = start + controller.window_size
        batch = {
            "hands": torch.stack(video_frames[history_start:]).unsqueeze(0),
            "hand_presence": torch.stack(
                presence_frames[history_start:]
            ).unsqueeze(0),
            "chords": chord_tokens[start:stop].unsqueeze(0),
            "frame_mask": torch.ones(
                (1, frame_index + 1 - history_start), dtype=torch.bool
            ),
            "chord_mask": torch.ones(
                (1, controller.window_size), dtype=torch.bool
            ),
        }
        with _autocast(device, bf16):
            logits = _forward(model, batch, device)
        scores = alignment_probabilities(logits[:, -1:])[0, 0].cpu().numpy()
        probability[frame_index, start:stop] = scores
        events = controller.collect_events(scores, window_start, frame_index)
        for event in events:
            controller.process_event(event, frame_index)
        controller.release_due(frame_index)

    # Targets are only diagnostic; the controller never observes score times.
    target = target_generator.generate(
        song.chords, frame_start=0, frame_count=frame_count, fps=controller.fps
    )
    loss, _ = alignment_probability_loss(
        torch.from_numpy(probability), target
    )
    extra_arrays = _controller_arrays(controller)
    frame_times = np.arange(frame_count) / controller.fps
    extra_arrays.update(
        frame_chord_window_start=window_starts,
        frame_video_history_start=history_starts,
        frame_time=frame_times,
        processing_time=frame_times + controller.processing_latency,
    )
    write_inference_artifact(
        output_path,
        probability=probability,
        target=target.numpy(),
        frame_mask=np.ones(frame_count, dtype=bool),
        chord_mask=np.ones(chord_count, dtype=bool),
        chord_pitches=chord_tokens[:, :128].numpy(),
        chord_ids=np.arange(1, chord_count + 1),
        record_id=song.record_id,
        frame_start=0,
        fps=controller.fps,
        loss=float(loss),
        extra_arrays=extra_arrays,
    )
    sonify_artifact(output_path, Path(output_path).with_suffix(".mid"))
    return controller


def full_song_real_time(cfg: DictConfig) -> None:
    """Reuse split preparation and infer one simulated performance per song."""

    settings = cfg.full_song_real_time
    components = data.build_data_components(cfg)
    output_dir = resolve_split_output_dir(settings.output_dir, settings.split_file)
    settings.output_dir = str(output_dir)
    device = torch.device(str(cfg.device))
    model = load_model_checkpoint(resolve_path(settings.checkpoint), device)
    for index, record_count, song, _ in iter_split_songs(
        cfg, settings.split_file,
        label="full-song-real-time", components=components,
    ):
        print(
            f"full-song-real-time {index}/{record_count}: infer {song.record_id} "
            f"({len(song.hands)} frames, {len(song.chords)} chords)",
            flush=True,
        )
        controller = infer_full_song_real_time(
            model, song, components["slicer"], components["target_generator"],
            device, output_dir / f"{song.record_id}.npz",
            tokenizer=components["tokenizer"],
            fps=float(cfg.dataset.fps),
            bf16=bool(cfg.train.bf16),
        )
        released = len(controller.release_records)
        print(
            f"full-song-real-time {index}/{record_count}: {song.record_id} "
            f"events={len(controller.event_records)} "
            f"accepted={sum(r['success'] for r in controller.event_records)} "
            f"anchors={sum(c.anchor_time is not None for c in controller.chords[1:])} "
            f"released={released} unreleased={controller.chord_count - released}",
            flush=True,
        )
