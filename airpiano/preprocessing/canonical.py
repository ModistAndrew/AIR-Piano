"""Convert production ragged WiLoR caches to canonical 30 Hz hand poses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from airpiano.utils import resolve_path, write_npz


RAW_SCHEMA_VERSION = "airpiano-wilor-ragged-v1"
CANONICAL_SCHEMA_VERSION = "airpiano-wilor-canonical-v1"
TARGET_FPS = 30.0
HAND_REPRESENTATIONS = ("wrist_relative", "absolute_joint_xy", "absolute_joint_wrist_xy")


def validate_canonical_representation(payload, expected: str) -> None:
    # Existing caches without this field use the original relative representation.
    actual = str(payload["representation"].item()) if "representation" in payload else "wrist_relative"
    if actual != expected:
        raise ValueError(f"canonical representation mismatch: expected {expected}, found {actual}")


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
    """Build relative joints or camera XY with relative Z; retain camera wrist."""

    def __init__(self, representation: str = "wrist_relative") -> None:
        if representation not in HAND_REPRESENTATIONS:
            raise ValueError(f"unknown hand representation: {representation}")
        self.representation = representation

    def __call__(
        self, arrays: Mapping[str, np.ndarray], detection: int
    ) -> np.ndarray:
        joints_3d = arrays["pred_keypoints_3d"][detection].astype(
            np.float64, copy=False
        )
        converted = (joints_3d - joints_3d[:1]).astype(np.float32)
        if self.representation != "wrist_relative":
            converted[1:, :2] = joints_3d[1:, :2] + arrays["pred_cam_t_full"][detection, :2]
        converted[0] = joints_3d[0] + arrays["pred_cam_t_full"][detection]
        return converted


def select_lowest_wrist_indices(
    frame_detection_offsets: np.ndarray,
    detector_confidence: np.ndarray,
    is_right: np.ndarray,
    wrist_y: np.ndarray,
) -> np.ndarray:
    """Return each frame's lowest anatomical Left/Right wrist indices.

    Larger image-space Y is lower. Confidence breaks equal-Y ties, and
    ``np.argmax`` preserves detection order when confidence also ties.
    The raw detector already applies its configurable confidence threshold.
    """

    selected = np.full((len(frame_detection_offsets) - 1, 2), -1, np.int64)
    for frame in range(len(selected)):
        start = int(frame_detection_offsets[frame])
        end = int(frame_detection_offsets[frame + 1])
        for slot in range(2):
            candidates = np.flatnonzero(is_right[start:end] == slot) + start
            if candidates.size:
                lowest_y = np.max(wrist_y[candidates])
                lowest = candidates[wrist_y[candidates] == lowest_y]
                selected[frame, slot] = int(
                    lowest[np.argmax(detector_confidence[lowest])]
                )
    return selected


class LowestWristHandSelector:
    """Choose the lowest wrist per hand, breaking ties by confidence and order."""

    def __call__(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return select_lowest_wrist_indices(
            arrays["frame_detection_offsets"],
            arrays["detector_confidence"],
            arrays["is_right"],
            arrays["pred_keypoints_2d_full"][:, 0, 1],
        )


class WristDeltaPostprocessor:
    """Replace camera wrist with displacement, optionally retaining absolute XY."""

    def __init__(self, absolute_wrist_xy: bool = False) -> None:
        self.absolute_wrist_xy = absolute_wrist_xy

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
        if self.absolute_wrist_xy:
            processed_pose[:, :, 0, :2] = pose[:, :, 0, :2]
        processed_pose[~processed_presence] = 0.0
        return processed_pose, processed_presence


class WiLoRToCanonical:
    """Select two hands from ragged detections and resample them to fixed FPS."""

    def __init__(
        self, target_fps: float = TARGET_FPS, representation: str = "wrist_relative"
    ) -> None:
        self.target_fps = float(target_fps)
        self.representation = representation
        self.coordinates = WiLoRCoordinateConverter(representation)
        self.selector = LowestWristHandSelector()
        self.postprocessor = WristDeltaPostprocessor(representation == "absolute_joint_wrist_xy")

    def run(
        self,
        raw_record_dir: str | Path,
        output_path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Convert one raw record to ``pose``, ``presence``, ``fps`` metadata."""

        output = resolve_path(output_path)
        if output.exists() and not overwrite:
            with np.load(output, allow_pickle=False) as payload:
                validate_canonical_representation(payload, self.representation)
            return output

        pose, presence = self.read_resampled(raw_record_dir)
        pose, presence = self.postprocessor(pose, presence, self.target_fps)
        canonicalization_config = {
            "selector": "lowest_wrist",
            "minimum_detector_confidence": 0.0,
            "coordinates": {
                "joints": ("wrist_relative_3d" if self.representation == "wrist_relative"
                           else "camera_xy_wrist_relative_z"),
                "wrist": "camera_3d",
            },
            "postprocessor": ("wrist_delta_except_absolute_xy"
                              if self.postprocessor.absolute_wrist_xy else "wrist_delta"),
            "target_fps": self.target_fps,
        }
        write_npz(
            output,
            pose=pose,
            presence=presence,
            fps=np.asarray(self.target_fps, dtype=np.float64),
            schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
            representation=np.asarray(self.representation),
            canonicalization_config=np.asarray(json.dumps(canonicalization_config)),
        )
        return output

    def read_resampled(self, raw_record_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Expose camera wrists before differencing for training-only mean fitting."""
        pose, presence, source_fps = self._read_raw_sequence(resolve_path(raw_record_dir))
        return self._resample(pose, presence, source_fps)

    def _read_raw_sequence(
        self, raw_directory: Path
    ) -> tuple[np.ndarray, np.ndarray, float]:
        manifest_path = raw_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = manifest["schema_version"]
        if schema_version != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"raw WiLoR cache schema mismatch at {manifest_path}: expected "
                f"{RAW_SCHEMA_VERSION!r}, found "
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
