"""Causal adapters for timing the existing WiLoR and AirPiano frame operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from airpiano import data
from airpiano.full_song_real_time import RealTimeController
from airpiano.model.loss import alignment_probabilities
from airpiano.preprocessing.canonical import WiLoRToCanonical
from airpiano.train import _autocast, _forward


class CanonicalFrameAdapter:
    """Convert ordered native-30-Hz WiLoR frames using production components.

    The source cadence must equal ``target_fps``. This adapter does not perform
    interpolation, which would require future frames at other source cadences.
    Each input is a one-frame ragged chunk, with detection offsets ``[0, n]``.
    """

    def __init__(self, target_fps: float = 30.0) -> None:
        self.converter = WiLoRToCanonical(target_fps)
        self.previous_pose: np.ndarray | None = None
        self.previous_presence: np.ndarray | None = None

    def step(
        self, arrays: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return processed_pose[-1], processed_presence[-1]


class AirPianoStepper:
    """Run one causal iteration of ``infer_full_song_real_time``.

    The caller loads the model, score, and data components once. Input poses use
    canonical cache coordinates; the configured hand-loader transform is applied
    here. Controller release times retain production's simulated clock, so the
    benchmark must measure wall-clock completion separately.
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
        self.transform = components["hand_loader"].transform
        self.chord_tokens = components["tokenizer"].tokenize(chords)
        self.controller = RealTimeController(
            chords,
            window_size=components["slicer"].segment_frames,
            fps=fps,
            target_radius_seconds=components[
                "target_generator"
            ].softener.radius_seconds,
        )
        self.video_frames: list[torch.Tensor] = []
        self.presence_frames: list[torch.Tensor] = []

    @torch.no_grad()
    def step(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        *,
        return_scores: bool = False,
    ) -> dict[str, Any]:
        frame_index = len(self.video_frames)
        presence_tensor = torch.from_numpy(presence)
        self.video_frames.append(
            self.transform(torch.from_numpy(pose), presence_tensor)
        )
        self.presence_frames.append(presence_tensor)
        controller = self.controller
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
        events = controller.collect_events(scores, window_start, frame_index)
        for event in events:
            controller.process_event(event, frame_index)
        releases = controller.release_due(frame_index)
        result = {
            "frame_index": frame_index,
            "history_start": history_start,
            "history_frames": frame_index + 1 - history_start,
            "window_start": window_start,
            "events": len(events),
            "releases": len(releases),
        }
        if return_scores:
            result["scores"] = scores
        return result
