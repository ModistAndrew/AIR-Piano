"""Convert production ragged WiLoR caches to canonical 30 Hz hand poses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from airpiano.preprocessing.wilor.selection import select_best_hand_indices
from airpiano.utils import resolve_path, write_npz


RAW_SCHEMA_VERSION = "airpiano-wilor-ragged-v1"
CANONICAL_SCHEMA_VERSION = "airpiano-wilor-canonical-v1"
TARGET_FPS = 30.0


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


class WiLoRToCanonical:
    """Select two hands from ragged detections and resample them to fixed FPS."""

    def __init__(
        self,
        target_fps: float = TARGET_FPS,
        coordinates: WiLoRCoordinateConverter | None = None,
    ) -> None:
        self.target_fps = float(target_fps)
        self.coordinates = coordinates or WiLoRCoordinateConverter()

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
