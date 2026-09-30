"""Convert production ragged WiLoR caches to canonical 30 Hz hand poses."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import savgol_filter

from airpiano.preprocessing.wilor.selection import select_best_hand_indices
from airpiano.utils import resolve_path, write_npz


RAW_SCHEMA_VERSION = "airpiano-wilor-ragged-v1"
LEGACY_RAW_SCHEMA_VERSION = "pianomotion-wilor-v1"
SUPPORTED_RAW_SCHEMA_VERSIONS = frozenset(
    {RAW_SCHEMA_VERSION, LEGACY_RAW_SCHEMA_VERSION}
)
CANONICAL_SCHEMA_VERSION = "airpiano-wilor-canonical-v1"
TARGET_FPS = 30.0


PosePostprocessor = Callable[
    [np.ndarray, np.ndarray, float],
    tuple[np.ndarray, np.ndarray],
]
HandSelector = Callable[[Mapping[str, np.ndarray]], np.ndarray]


class InsufficientHandPresenceError(RuntimeError):
    """Raised when a canonical record fails its configured presence gate."""

    def __init__(
        self,
        raw_record_dir: Path,
        presence: float,
        minimum_presence: float,
    ) -> None:
        self.raw_record_dir = raw_record_dir
        self.presence = presence
        self.minimum_presence = minimum_presence
        super().__init__(
            f"{raw_record_dir}: hand presence {presence:.6f} is below "
            f"the required {minimum_presence:.6f}"
        )


def raw_cache_has_markers(path: str | Path) -> bool:
    """Return whether a raw record advertises a completed cache.

    This is intentionally the only preflight check. Once all markers exist,
    malformed manifests and chunks fail at their native JSON or NumPy operation.
    """

    directory = resolve_path(path)
    return (
        directory.is_dir()
        and (directory / "manifest.json").is_file()
        and (directory / "_SUCCESS").is_file()
    )


class WiLoRCoordinateConverter:
    """Build one canonical hand from WiLoR 2D and local 3D joints.

    ``xy_source`` selects either full-image normalized coordinates or local
    joints relative to that hand's wrist. Depth remains local and
    wrist-relative in both currently supported representations.
    ``wrist_source="camera_3d"`` replaces joint zero with the camera-space
    wrist while leaving every other joint in the selected representation.
    """

    def __init__(
        self,
        xy_source: str = "normalized_2d",
        z_source: str = "wrist_relative_3d",
        wrist_source: str = "converted",
    ) -> None:
        self.xy_source = xy_source
        self.z_source = z_source
        self.wrist_source = wrist_source

    def __call__(
        self, arrays: Mapping[str, np.ndarray], detection: int
    ) -> np.ndarray:
        joints_3d = arrays["pred_keypoints_3d"][detection].astype(
            np.float64, copy=False
        )
        wrist_relative = joints_3d - joints_3d[:1]

        if self.xy_source == "normalized_2d":
            image_size = arrays["image_size_wh"][detection].astype(
                np.float64, copy=False
            )
            xy = arrays["pred_keypoints_2d_full"][detection].astype(
                np.float64, copy=False
            ) / image_size
        elif self.xy_source == "wrist_relative_3d":
            xy = wrist_relative[:, :2]
        else:
            raise ValueError(f"unknown WiLoR XY source: {self.xy_source}")

        if self.z_source != "wrist_relative_3d":
            raise ValueError(f"unknown WiLoR Z source: {self.z_source}")

        converted = np.empty((21, 3), dtype=np.float32)
        converted[:, :2] = xy
        converted[:, 2] = wrist_relative[:, 2]
        if self.wrist_source == "camera_3d":
            converted[0] = (
                joints_3d[0] + arrays["pred_cam_t_full"][detection]
            )
        elif self.wrist_source != "converted":
            raise ValueError(
                f"unknown WiLoR wrist source: {self.wrist_source}"
            )
        return converted


class LowestWristHandSelector:
    """Select the lowest wrist per handedness, then apply confidence."""

    def __init__(self, minimum_detector_confidence: float = 0.75) -> None:
        if not 0.0 <= minimum_detector_confidence <= 1.0:
            raise ValueError(
                "minimum_detector_confidence must be between zero and one"
            )
        self.minimum_detector_confidence = float(minimum_detector_confidence)

    def __call__(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        offsets = arrays["frame_detection_offsets"]
        confidence = arrays["detector_confidence"]
        handedness = arrays["is_right"]
        wrist_v = arrays["pred_keypoints_2d_full"][:, 0, 1]
        selected = np.full((len(offsets) - 1, 2), -1, np.int64)

        for frame in range(len(selected)):
            start = int(offsets[frame])
            end = int(offsets[frame + 1])
            for slot in range(2):
                candidates = np.flatnonzero(handedness[start:end] == slot) + start
                if not candidates.size:
                    continue
                lowest_v = np.max(wrist_v[candidates])
                lowest = candidates[wrist_v[candidates] == lowest_v]
                winner = int(lowest[np.argmax(confidence[lowest])])
                if confidence[winner] >= self.minimum_detector_confidence:
                    selected[frame, slot] = winner
        return selected


class WristDeltaPostprocessor:
    """Replace the camera-space wrist with its canonical-frame displacement."""

    def __call__(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        processed_pose = np.array(pose, copy=True)
        processed_presence = np.array(presence, dtype=np.bool_, copy=True)
        wrist_delta = np.zeros_like(processed_pose[:, :, 0])
        valid_delta = processed_presence[1:] & processed_presence[:-1]
        wrist_delta[1:] = np.where(
            valid_delta[..., None],
            processed_pose[1:, :, 0] - processed_pose[:-1, :, 0],
            0.0,
        )
        processed_pose[:, :, 0] = wrist_delta
        processed_pose[~processed_presence] = 0.0
        return processed_pose, processed_presence


class PianoMotion10MPostprocessor:
    """Clean and smooth canonical 30 Hz poses using the paper's stages."""

    def __init__(
        self,
        hampel_window_frames: int = 20,
        hampel_n_sigma: float = 3.0,
        max_missing_run_frames: int = 29,
        min_presence_run_frames: int = 15,
        savgol_window_frames: int = 11,
        savgol_polyorder: int = 3,
    ) -> None:
        if hampel_window_frames <= 0:
            raise ValueError("hampel_window_frames must be positive")
        if hampel_n_sigma <= 0.0:
            raise ValueError("hampel_n_sigma must be positive")
        if max_missing_run_frames < 0:
            raise ValueError("max_missing_run_frames must be non-negative")
        if min_presence_run_frames <= 0:
            raise ValueError("min_presence_run_frames must be positive")
        if savgol_window_frames <= 0 or savgol_window_frames % 2 == 0:
            raise ValueError("savgol_window_frames must be positive and odd")
        if savgol_polyorder < 0:
            raise ValueError("savgol_polyorder must be non-negative")
        if savgol_window_frames <= savgol_polyorder:
            raise ValueError(
                "savgol_window_frames must exceed savgol_polyorder"
            )
        self.hampel_window_frames = int(hampel_window_frames)
        self.hampel_n_sigma = float(hampel_n_sigma)
        self.max_missing_run_frames = int(max_missing_run_frames)
        self.min_presence_run_frames = int(min_presence_run_frames)
        self.savgol_window_frames = int(savgol_window_frames)
        self.savgol_polyorder = int(savgol_polyorder)

    def __call__(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return independently cleaned and smoothed left/right poses."""

        if not math.isclose(fps, TARGET_FPS, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"PianoMotion10MPostprocessor requires {TARGET_FPS} fps, "
                f"received {fps}"
            )

        processed_pose = np.array(pose, dtype=np.float32, copy=True)
        processed_presence = np.array(presence, dtype=np.bool_, copy=True)

        # Compute every Hampel decision against the same pre-Hampel sequence.
        # Blocking bounds the temporary rolling-window arrays for long videos.
        outliers = np.zeros_like(processed_presence)
        left_context = self.hampel_window_frames // 2
        right_context = self.hampel_window_frames - left_context - 1
        for slot in range(processed_presence.shape[1]):
            slot_presence = processed_presence[:, slot]
            present_frames = np.flatnonzero(slot_presence)
            if not present_frames.size:
                continue
            flattened_pose = processed_pose[:, slot].reshape(
                processed_pose.shape[0], -1
            )
            observed_pose = np.where(
                slot_presence[:, None], flattened_pose, np.nan
            )
            padded_pose = np.pad(
                observed_pose,
                ((left_context, right_context), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                padded_pose,
                self.hampel_window_frames,
                axis=0,
            )
            for block_start in range(0, len(present_frames), 4096):
                frame_indices = present_frames[block_start : block_start + 4096]
                block = windows[frame_indices]
                local_median = np.nanmedian(block, axis=-1)
                local_mad = np.nanmedian(
                    np.abs(block - local_median[..., None]),
                    axis=-1,
                )
                residual = np.abs(
                    flattened_pose[frame_indices] - local_median
                )
                scaled_mad = np.float32(1.4826) * local_mad
                standardized = np.zeros_like(residual)
                nonzero_mad = scaled_mad > 0.0
                np.divide(
                    residual,
                    scaled_mad,
                    out=standardized,
                    where=nonzero_mad,
                )
                standardized[~nonzero_mad & (residual > 0.0)] = np.inf
                score = np.median(standardized, axis=1)
                outliers[frame_indices, slot] = (
                    score > self.hampel_n_sigma
                )
        processed_presence[outliers] = False

        for slot in range(processed_presence.shape[1]):
            detected = np.flatnonzero(processed_presence[:, slot])
            for left, right in zip(detected[:-1], detected[1:]):
                gap_frames = int(right - left - 1)
                if not 0 < gap_frames <= self.max_missing_run_frames:
                    continue
                alpha = (
                    np.arange(1, gap_frames + 1, dtype=np.float32)
                    / np.float32(gap_frames + 1)
                )[:, None, None]
                processed_pose[left + 1 : right, slot] = (
                    processed_pose[left, slot] * (np.float32(1.0) - alpha)
                    + processed_pose[right, slot] * alpha
                )
                processed_presence[left + 1 : right, slot] = True

        for slot in range(processed_presence.shape[1]):
            mask = processed_presence[:, slot]
            padded = np.concatenate(
                (np.asarray([False]), mask, np.asarray([False]))
            )
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            stops = np.flatnonzero(padded[:-1] & ~padded[1:])
            for start, stop in zip(starts, stops):
                if stop - start < self.min_presence_run_frames:
                    processed_presence[start:stop, slot] = False

        for slot in range(processed_presence.shape[1]):
            mask = processed_presence[:, slot]
            padded = np.concatenate(
                (np.asarray([False]), mask, np.asarray([False]))
            )
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            stops = np.flatnonzero(padded[:-1] & ~padded[1:])
            for start, stop in zip(starts, stops):
                if stop - start < self.savgol_window_frames:
                    continue
                processed_pose[start:stop, slot] = savgol_filter(
                    processed_pose[start:stop, slot],
                    window_length=self.savgol_window_frames,
                    polyorder=self.savgol_polyorder,
                    axis=0,
                    mode="interp",
                )

        processed_pose[~processed_presence] = 0.0
        return (
            processed_pose.astype(np.float32, copy=False),
            processed_presence.astype(np.bool_, copy=False),
        )


class WiLoRToCanonical:
    """Select two hands from ragged detections and resample them to fixed FPS."""

    def __init__(
        self,
        target_fps: float = TARGET_FPS,
        coordinates: WiLoRCoordinateConverter | None = None,
        selector: HandSelector | None = None,
        postprocessor: PosePostprocessor | None = None,
        minimum_presence: float | None = None,
    ) -> None:
        if minimum_presence is not None and not 0.0 <= minimum_presence <= 1.0:
            raise ValueError("minimum_presence must be between zero and one")
        self.target_fps = float(target_fps)
        self.coordinates = coordinates or WiLoRCoordinateConverter()
        self.selector = selector
        self.postprocessor = postprocessor
        self.minimum_presence = (
            None if minimum_presence is None else float(minimum_presence)
        )

    def run(
        self,
        raw_record_dir: str | Path,
        output_path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Convert one raw record to ``pose``, ``presence``, ``fps`` metadata."""

        output = resolve_path(output_path)
        if output.exists() and not overwrite:
            return output

        raw_directory = resolve_path(raw_record_dir)
        pose, presence, source_fps = self._read_raw_sequence(raw_directory)
        pose, presence = self._resample(pose, presence, source_fps)
        if self.postprocessor is not None:
            pose, presence = self.postprocessor(
                pose, presence, self.target_fps
            )
        if self.minimum_presence is not None:
            presence_rate = float(np.mean(presence))
            if presence_rate < self.minimum_presence:
                raise InsufficientHandPresenceError(
                    raw_directory,
                    presence_rate,
                    self.minimum_presence,
                )
        write_npz(
            output,
            pose=pose,
            presence=presence,
            fps=np.asarray(self.target_fps, dtype=np.float64),
            schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
        )
        return output

    def _read_raw_sequence(
        self, raw_directory: Path
    ) -> tuple[np.ndarray, np.ndarray, float]:
        manifest_path = raw_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = manifest["schema_version"]
        if schema_version not in SUPPORTED_RAW_SCHEMA_VERSIONS:
            raise ValueError(
                f"raw WiLoR cache schema mismatch at {manifest_path}: expected "
                f"one of {sorted(SUPPORTED_RAW_SCHEMA_VERSIONS)!r}, found "
                f"{schema_version!r}"
            )
        frame_count = int(manifest["decoded_frame_count"])
        source_fps = float(manifest["video"]["fps"])

        pose = np.zeros((frame_count, 2, 21, 3), dtype=np.float32)
        presence = np.zeros((frame_count, 2), dtype=np.bool_)
        for chunk in manifest["chunks"]:
            frame_start = int(chunk["frame_start"])
            frame_end = int(chunk["frame_end"])
            with np.load(raw_directory / chunk["file"], allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in (
                    "frame_detection_offsets",
                    "detector_confidence",
                    "is_right",
                    "pred_keypoints_3d",
                    "pred_cam_t_full",
                    "pred_keypoints_2d_full",
                    "image_size_wh",
                )}
            self._select_chunk(
                arrays,
                pose[frame_start:frame_end],
                presence[frame_start:frame_end],
            )
        return pose, presence, source_fps

    def _select_chunk(
        self,
        arrays: Mapping[str, np.ndarray],
        destination_pose: np.ndarray,
        destination_presence: np.ndarray,
    ) -> None:
        """Choose one left and right detection per frame."""

        offsets = arrays["frame_detection_offsets"]
        if self.selector is None:
            selected = select_best_hand_indices(
                offsets,
                arrays["detector_confidence"],
                arrays["is_right"],
            )
        else:
            selected = self.selector(arrays)
        for frame in range(destination_pose.shape[0]):
            for slot, winner in enumerate(selected[frame]):
                if winner < 0:
                    continue
                destination_pose[frame, slot] = self.coordinates(
                    arrays, int(winner)
                )
                destination_presence[frame, slot] = True

    def _resample(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        source_fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly sample source frame centers at canonical frame centers.

        A hand is usable only when it exists at both interpolation endpoints;
        otherwise its entire canonical slot is exactly zero.
        """

        source_frames = pose.shape[0]
        target_frames = math.floor(source_frames * self.target_fps / source_fps)
        coordinate = (
            (np.arange(target_frames, dtype=np.float64) + 0.5)
            * source_fps
            / self.target_fps
            - 0.5
        )
        coordinate = np.clip(coordinate, 0.0, source_frames - 1)
        left = np.floor(coordinate + 1e-12).astype(np.int64)
        right = np.minimum(left + 1, source_frames - 1)
        alpha = coordinate - left
        exact = np.isclose(alpha, 0.0, rtol=0.0, atol=1e-12)
        alpha[exact] = 0.0
        right[exact] = left[exact]

        target_presence = presence[left] & presence[right]
        blend = alpha.astype(np.float32)[:, None, None, None]
        target_pose = (
            pose[left] * (np.float32(1.0) - blend) + pose[right] * blend
        ).astype(np.float32, copy=False)
        target_pose[~target_presence] = 0.0
        return target_pose, target_presence.astype(np.bool_, copy=False)
