"""Shared causal frame conversion and score-following inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .. import data
from ..full_song_real_time import RealTimeController
from ..model.loss import alignment_probabilities
from ..preprocessing.canonical import WiLoRToCanonical
from ..train import _autocast, _forward


class CanonicalFrameAdapter:
    """Convert one WiLoR frame on the canonical clock, resetting across gaps.

    Inputs are one-frame ragged chunks. The caller assigns canonical frame
    indices; this adapter does not interpolate frames from other source cadences.
    """

    def __init__(self, target_fps: float = 30.0) -> None:
        self.converter = WiLoRToCanonical(target_fps)
        self.previous_pose: np.ndarray | None = None
        self.previous_presence: np.ndarray | None = None
        self.last_frame_index = -1

    def step(
        self, arrays: Mapping[str, np.ndarray], *, frame_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if frame_index is None:
            frame_index = self.last_frame_index + 1
        if frame_index <= self.last_frame_index:
            raise ValueError("canonical frame indices must increase")
        if frame_index != self.last_frame_index + 1:
            self.previous_pose = None
            self.previous_presence = None
        pose = np.zeros((1, 2, 21, 3), dtype=np.float32)
        presence = np.zeros((1, 2), dtype=np.bool_)
        self.converter._select_chunk(arrays, pose, presence)
        if self.previous_pose is None:
            history_pose = pose
            history_presence = presence
        else:
            history_pose = np.concatenate((self.previous_pose, pose))
            history_presence = np.concatenate((self.previous_presence, presence))
        processed_pose, processed_presence = self.converter.postprocessor(
            history_pose, history_presence, self.converter.target_fps
        )
        self.previous_pose = pose
        self.previous_presence = presence
        self.last_frame_index = frame_index
        return processed_pose[-1], processed_presence[-1]


class AirPianoStepper:
    """Run the shared frame iteration on already transformed CPU hand tensors.

    Logical frame indices preserve the canonical clock when observations drop.
    Missing slots remain real time steps with absent hands, and only the supplied
    observation produces events. Full CPU history is retained for controller
    rewinds. Set release=False when a separate scheduler owns release timing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        chords: Sequence[data.Chord],
        components: Mapping[str, Any],
        device: torch.device,
        *,
        fps: float = 30.0,
        bf16: bool = False,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.bf16 = bf16
        self.chord_tokens = components["tokenizer"].tokenize(chords)
        target_generator = components["target_generator"]
        self.controller = RealTimeController(
            chords,
            window_size=components["slicer"].segment_frames,
            fps=fps,
            target_radius_seconds=target_generator.softener.radius_seconds,
            target_delay_seconds=target_generator.event_generator.delay_seconds,
        )
        self.video_frames: list[torch.Tensor] = []
        self.presence_frames: list[torch.Tensor] = []
        self.last_frame_index = -1
        self.last_source_time = 0.0

    @torch.no_grad()
    def step(
        self,
        hands: torch.Tensor,
        presence: torch.Tensor,
        *,
        frame_index: int | None = None,
        source_time: float | None = None,
        release: bool = True,
        now: float | None = None,
        return_scores: bool = False,
    ) -> dict[str, Any]:
        if frame_index is None:
            frame_index = self.last_frame_index + 1
        if frame_index <= self.last_frame_index:
            raise ValueError("inference frame indices must increase")
        missing_count = frame_index - self.last_frame_index - 1
        if missing_count:
            self.video_frames.extend([torch.zeros_like(hands)] * missing_count)
            self.presence_frames.extend([torch.zeros_like(presence)] * missing_count)
        self.video_frames.append(hands)
        self.presence_frames.append(presence)
        controller = self.controller
        frame_time = frame_index / controller.fps if source_time is None else source_time
        with controller.lock:
            window_start = controller.chord_window_start
            history_start = controller.video_history_start
        start = window_start - 1
        stop = start + controller.window_size
        batch = {
            "hands": torch.stack(self.video_frames[history_start:]).unsqueeze(0),
            "hand_presence": torch.stack(
                self.presence_frames[history_start:]
            ).unsqueeze(0),
            "chords": self.chord_tokens[start:stop].unsqueeze(0),
            "frame_mask": torch.ones(
                (1, frame_index + 1 - history_start), dtype=torch.bool
            ),
            "chord_mask": torch.ones(
                (1, controller.window_size), dtype=torch.bool
            ),
        }
        with _autocast(self.device, self.bf16):
            logits = _forward(self.model, batch, self.device)
        scores = alignment_probabilities(logits[:, -1:])[0, 0].cpu().numpy()
        with controller.lock:
            events = controller.collect_events(
                scores, window_start, frame_index, source_time=source_time
            )
            for event in events:
                controller.process_event(event, frame_index)
            self.last_frame_index = frame_index
            self.last_source_time = frame_time
            releases = (
                controller.release_due(frame_index, now=now, frame_time=frame_time)
                if release else []
            )
        result = {
            "frame_index": frame_index,
            "source_time": frame_time,
            "history_start": history_start,
            "history_frames": frame_index + 1 - history_start,
            "window_start": window_start,
            "missing_frames": missing_count,
            "events": len(events),
            "releases": len(releases),
        }
        if return_scores:
            result["scores"] = scores
        return result
