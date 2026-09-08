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
CANONICAL_SCHEMA_VERSION = "airpiano-wilor-canonical-v1"
TARGET_FPS = 30.0


PosePostprocessor = Callable[
    [np.ndarray, np.ndarray, float],
    tuple[np.ndarray, np.ndarray],
]


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
    """

    def __init__(
        self,
        xy_source: str = "normalized_2d",
        z_source: str = "wrist_relative_3d",
    ) -> None:
        self.xy_source = xy_source
        self.z_source = z_source

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
        return converted


class GapAwareSavitzkyGolayPostprocessor:
    """Fill short dropouts and smooth valid source-rate pose runs."""

    def __init__(
        self,
        max_gap_seconds: float = 0.09,
        window_seconds: float = 0.17,
        polyorder: int = 2,
    ) -> None:
        if max_gap_seconds < 0.0:
            raise ValueError("max_gap_seconds must be non-negative")
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if polyorder < 0:
            raise ValueError("polyorder must be non-negative")
        self.max_gap_seconds = float(max_gap_seconds)
        self.window_seconds = float(window_seconds)
        self.polyorder = int(polyorder)

    def __call__(
        self,
        pose: np.ndarray,
        presence: np.ndarray,
        source_fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return independently repaired and smoothed left/right poses."""

        processed_pose = np.array(pose, dtype=np.float32, copy=True)
        processed_presence = np.array(presence, dtype=np.bool_, copy=True)
        max_gap_frames = math.floor(self.max_gap_seconds * source_fps + 1e-12)

        for slot in range(processed_presence.shape[1]):
            detected = np.flatnonzero(processed_presence[:, slot])
            for left, right in zip(detected[:-1], detected[1:]):
                gap_frames = int(right - left - 1)
                if not 0 < gap_frames <= max_gap_frames:
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

        requested_window = self.window_seconds * source_fps
        window_length = max(1, 2 * math.floor(requested_window / 2.0) + 1)
        if window_length <= self.polyorder:
            raise ValueError(
                f"window_seconds={self.window_seconds} resolves to "
                f"{window_length} samples at {source_fps} fps, but must exceed "
                f"polyorder={self.polyorder}"
            )

        for slot in range(processed_presence.shape[1]):
            mask = processed_presence[:, slot]
            padded = np.concatenate(
                (np.asarray([False]), mask, np.asarray([False]))
            )
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            stops = np.flatnonzero(padded[:-1] & ~padded[1:])
            for start, stop in zip(starts, stops):
                if stop - start < window_length:
                    continue
                processed_pose[start:stop, slot] = savgol_filter(
                    processed_pose[start:stop, slot],
                    window_length=window_length,
                    polyorder=self.polyorder,
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
        postprocessor: PosePostprocessor | None = None,
    ) -> None:
        self.target_fps = float(target_fps)
        self.coordinates = coordinates or WiLoRCoordinateConverter()
        self.postprocessor = postprocessor

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

        pose, presence, source_fps = self._read_raw_sequence(
            resolve_path(raw_record_dir)
        )
        if self.postprocessor is not None:
            pose, presence = self.postprocessor(pose, presence, source_fps)
        pose, presence = self._resample(pose, presence, source_fps)
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
        manifest = json.loads(
            (raw_directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["schema_version"] != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"raw WiLoR cache schema mismatch at "
                f"{raw_directory / 'manifest.json'}: expected "
                f"{RAW_SCHEMA_VERSION!r}, found {manifest['schema_version']!r}"
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
        """Choose the highest-confidence left and right detection per frame."""

        offsets = arrays["frame_detection_offsets"]
        selected = select_best_hand_indices(
            offsets,
            arrays["detector_confidence"],
            arrays["is_right"],
        )
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
