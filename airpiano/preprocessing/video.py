"""Public path-based component for producing ragged WiLoR caches."""

from __future__ import annotations

from pathlib import Path

from airpiano.preprocessing.wilor.backend import WiLoRBackend
from airpiano.preprocessing.wilor.cache import (
    completed_output_is_valid,
    process_video,
)


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
    ) -> None:
        repository = Path(wilor_repo).expanduser().resolve()
        self.chunk_frames = int(chunk_frames)
        self._backend = WiLoRBackend(
            wilor_repo=repository,
            checkpoint=checkpoint,
            model_config=(
                model_config
                or repository / "pretrained_models/model_config.yaml"
            ),
            detector_checkpoint=detector_checkpoint,
            mano_model=mano_model,
            mano_mean_params=(
                mano_mean_params or repository / "mano_data/mano_mean_params.npz"
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

        output = Path(output_dir).expanduser().resolve()

        # Reuse precedes source lookup so retained raw caches remain useful after
        # the large input video has been removed. Schema mismatches raise here.
        if not overwrite and completed_output_is_valid(output):
            return output

        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        process_video(
            video,
            output,
            self._backend,
            chunk_frames=self.chunk_frames,
            overwrite=overwrite,
        )
        if not completed_output_is_valid(output):
            raise RuntimeError(f"WiLoR extraction did not complete: {output}")
        return output
