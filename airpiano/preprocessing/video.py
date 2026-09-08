"""Public path-based component for producing ragged WiLoR caches."""

from __future__ import annotations

import math
from pathlib import Path

from airpiano.preprocessing.wilor.backend import WiLoRBackend
from airpiano.preprocessing.wilor.cache import (
    completed_output_is_valid,
    process_video,
)
from airpiano.utils import resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VideoToWiLoR:
    """Extract videos with one lazily shared, headless WiLoR model bundle."""

    def __init__(
        self,
        wilor_repo: str | Path = PROJECT_ROOT / "third_party/WiLoR",
        checkpoint: str | Path = PROJECT_ROOT / "assets/wilor/wilor_final.ckpt",
        model_config: str | Path | None = None,
        detector_checkpoint: str | Path = PROJECT_ROOT / "assets/wilor/detector.pt",
        mano_model: str | Path = PROJECT_ROOT / "assets/wilor/MANO_RIGHT.pkl",
        mano_mean_params: str | Path | None = None,
        device: str = "cuda:1",
        detector_confidence: float = 0.3,
        rescale_factor: float = 2.0,
        frame_batch_size: int = 8,
        hand_batch_size: int = 16,
        chunk_frames: int = 512,
        fast: bool = False,
        compile: bool = False,
        target_fps: float | None = None,
    ) -> None:
        repository = resolve_path(wilor_repo)
        self.chunk_frames = int(chunk_frames)
        if target_fps is not None and (
            not math.isfinite(target_fps) or target_fps <= 0
        ):
            raise ValueError("target_fps must be positive or None")
        self.target_fps = target_fps
        self._backend = WiLoRBackend(
            wilor_repo=repository,
            checkpoint=resolve_path(checkpoint),
            model_config=(
                resolve_path(model_config)
                if model_config is not None
                else repository / "pretrained_models/model_config.yaml"
            ),
            detector_checkpoint=resolve_path(detector_checkpoint),
            mano_model=resolve_path(mano_model),
            mano_mean_params=(
                resolve_path(mano_mean_params)
                if mano_mean_params is not None
                else repository / "mano_data/mano_mean_params.npz"
            ),
            device=device,
            detector_confidence=detector_confidence,
            rescale_factor=rescale_factor,
            frame_batch_size=frame_batch_size,
            hand_batch_size=hand_batch_size,
            fast=fast,
            compile=compile,
        )

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Extract one video or reuse a complete current-schema raw cache."""

        output = resolve_path(output_dir)

        # Reuse precedes source lookup so retained raw caches remain useful after
        # the large input video has been removed. Schema mismatches raise here.
        if not overwrite and completed_output_is_valid(output):
            return output

        video = resolve_path(video_path)
        if not video.is_file():
            raise FileNotFoundError(video)
        process_video(
            video,
            output,
            self._backend,
            chunk_frames=self.chunk_frames,
            overwrite=overwrite,
            target_fps=self.target_fps,
        )
        if not completed_output_is_valid(output):
            raise RuntimeError(f"WiLoR extraction did not complete: {output}")
        return output
