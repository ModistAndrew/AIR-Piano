"""Compatibility adapters for benchmarks using untransformed NumPy hand poses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from airpiano import data
from airpiano.streaming.core import AirPianoStepper as TensorAirPianoStepper
from airpiano.streaming.core import CanonicalFrameAdapter


class AirPianoStepper(TensorAirPianoStepper):
    """Apply the configured hand transform once before shared frame inference."""

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
        super().__init__(model, chords, components, device, fps=fps, bf16=bf16)
        self.transform = components["hand_loader"].transform

    def step(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        *,
        return_scores: bool = False,
    ) -> dict[str, Any]:
        presence_tensor = torch.from_numpy(presence)
        return super().step(
            self.transform(torch.from_numpy(pose), presence_tensor),
            presence_tensor,
            return_scores=return_scores,
        )
