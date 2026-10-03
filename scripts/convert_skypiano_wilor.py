"""Convert SKY-Piano markers to a synthetic ragged WiLoR-style cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.preprocessing.skypiano import (
    RAW_SCHEMA_VERSION,
    skypiano_camera_marker_data,
    skypiano_camera_markers,
)
from airpiano.preprocessing.wilor.cache import atomic_json, atomic_npz, utc_now


DEFAULT_MOTION_ROOT = PROJECT_ROOT / "assets" / "sky_piano" / "motion_hand"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "assets" / "sky_piano" / "wilor"
DATASET_NAME = "SKY-Piano"
INPUT_SUFFIX = "_imputed.csv"
CAMERA_SCHEMA_VERSION = "airpiano-skypiano-camera-v1"
TARGET_FPS = 30.0
DEFAULT_CHUNK_FRAMES = 512
IMAGE_SIZE_WH = (1920, 1080)
MODEL_FOCAL_LENGTH = 5000.0
MODEL_IMAGE_SIZE = 256
FIT_QUANTILES = (0.001, 0.999)
SAFE_FRACTION = 0.8


def fit_virtual_camera(
    records: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Fit one shared WiLoR-convention pinhole camera to marker records."""

    points = np.concatenate(
        [records[record_id].reshape(-1, 3) for record_id in sorted(records)]
    ).astype(np.float64, copy=False)
    if not np.isfinite(points).all():
        raise ValueError("SKY-Piano imputed hand markers must all be finite")

    lower = np.quantile(points[:, :2], FIT_QUANTILES[0], axis=0)
    upper = np.quantile(points[:, :2], FIT_QUANTILES[1], axis=0)
    span = upper - lower
    width, height = IMAGE_SIZE_WH
    pixels_per_meter = min(
        SAFE_FRACTION * width / span[0],
        SAFE_FRACTION * height / span[1],
    )
    scaled_focal_length = (
        MODEL_FOCAL_LENGTH / MODEL_IMAGE_SIZE * max(IMAGE_SIZE_WH)
    )
    center = (lower + upper) / 2.0
    translation = np.asarray(
        (-center[0], -center[1], scaled_focal_length / pixels_per_meter),
        dtype=np.float64,
    )
    return {
        "schema_version": CAMERA_SCHEMA_VERSION,
        "model": "pinhole",
        "image_size_wh": list(IMAGE_SIZE_WH),
        "principal_point_xy": [width / 2.0, height / 2.0],
        "model_focal_length": MODEL_FOCAL_LENGTH,
        "model_image_size": MODEL_IMAGE_SIZE,
        "scaled_focal_length": scaled_focal_length,
        "translation_xyz_meters": translation.tolist(),
        "axis_signs": [-1.0, 1.0, -1.0],
        "fit": {
            "records": sorted(records),
            "lower_quantile": FIT_QUANTILES[0],
            "upper_quantile": FIT_QUANTILES[1],
            "safe_fraction": SAFE_FRACTION,
            "xy_lower_meters": lower.tolist(),
            "xy_upper_meters": upper.tolist(),
            "nominal_pixels_per_meter": float(pixels_per_meter),
        },
    }


def project_points(points: np.ndarray, camera: Mapping[str, Any]) -> np.ndarray:
    """Project absolute camera-convention points into the virtual image."""

    translation = np.asarray(
        camera["translation_xyz_meters"], dtype=np.float64
    )
    camera_points = points.astype(np.float64, copy=False) + translation
    focal_length = float(camera["scaled_focal_length"])
    principal_point = np.asarray(
        camera["principal_point_xy"], dtype=np.float64
    )
    return (
        focal_length
        * camera_points[..., :2]
        / camera_points[..., 2:3]
        + principal_point
    )


def build_chunk(
    markers: np.ndarray,
    frame_start: int,
    frame_end: int,
    camera: Mapping[str, Any],
    local_markers: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build one frame-major ragged chunk from absolute marker joints."""

    absolute = markers[frame_start:frame_end]
    local = (
        absolute - absolute[:, :, :1]
        if local_markers is None
        else local_markers[frame_start:frame_end]
    )
    projected = project_points(absolute, camera)
    width, height = camera["image_size_wh"]
    inside = (
        (projected[..., 0] >= 0.0)
        & (projected[..., 0] < width)
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] < height)
    )
    visible = inside.all(axis=2)
    detection_frames, detection_hands = np.nonzero(visible)
    detection_count = len(detection_frames)

    offsets = np.zeros(len(absolute) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(visible.sum(axis=1), dtype=np.int64)
    selected_local = local[detection_frames, detection_hands]
    selected_projected = projected[detection_frames, detection_hands]
    selected_wrists = absolute[detection_frames, detection_hands, 0]
    translation = np.asarray(
        camera["translation_xyz_meters"], dtype=np.float32
    )

    if detection_count:
        boxes = np.concatenate(
            (
                selected_projected.min(axis=1),
                selected_projected.max(axis=1),
            ),
            axis=1,
        ).astype(np.float32)
    else:
        boxes = np.empty((0, 4), dtype=np.float32)

    return {
        "frame_indices": np.arange(frame_start, frame_end, dtype=np.int64),
        "frame_timestamps_seconds": np.arange(
            frame_start, frame_end, dtype=np.float64
        )
        / TARGET_FPS,
        "frame_detection_offsets": offsets,
        "boxes_xyxy": boxes,
        "detector_confidence": np.ones(detection_count, dtype=np.float32),
        "is_right": detection_hands.astype(np.uint8, copy=False),
        "pred_keypoints_3d": selected_local.astype(np.float32, copy=False),
        "pred_cam_t_full": (
            selected_wrists.astype(np.float32, copy=False) + translation
        ).astype(np.float32, copy=False),
        "pred_keypoints_2d_full": selected_projected.astype(
            np.float32, copy=False
        ),
        "image_size_wh": np.tile(
            np.asarray(camera["image_size_wh"], dtype=np.float32),
            (detection_count, 1),
        ),
    }


def _completed_record_matches(
    output: Path,
    source: Path,
    camera: Mapping[str, Any],
) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file() or not (output / "_SUCCESS").is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest["schema_version"] == RAW_SCHEMA_VERSION
        and manifest["status"] == "complete"
        and manifest["motion"]["path"] == str(source.resolve())
        and manifest["camera"] == camera
    )


def convert_motion_csv(
    input_path: str | Path,
    output_dir: str | Path,
    camera: Mapping[str, Any],
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> tuple[Path, int, int, bool]:
    """Convert one imputed marker CSV or reuse its completed raw cache."""

    source = Path(input_path)
    output = Path(output_dir)
    if output.exists():
        if _completed_record_matches(output, source, camera):
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            return (
                output,
                int(manifest["decoded_frame_count"]),
                int(manifest["detection_count"]),
                True,
            )
        raise FileExistsError(
            f"incomplete or incompatible SKY-Piano cache at {output}; "
            "rerun with --overwrite"
        )

    markers, local_markers = skypiano_camera_marker_data(source)
    output.mkdir(parents=True)
    chunks_directory = output / "chunks"
    chunks_directory.mkdir()
    timestamp = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "status": "processing",
        "motion": {
            "path": str(source.resolve()),
            "fps": TARGET_FPS,
            "units": "meters",
            "source_units": "millimeters",
        },
        "camera": dict(camera),
        "chunks": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    atomic_json(output / "manifest.json", manifest)

    inventory: list[dict[str, Any]] = []
    detection_count = 0
    for chunk_index, frame_start in enumerate(
        range(0, len(markers), chunk_frames)
    ):
        frame_end = min(frame_start + chunk_frames, len(markers))
        arrays = build_chunk(
            markers,
            frame_start,
            frame_end,
            camera,
            local_markers,
        )
        chunk_path = chunks_directory / f"{chunk_index:06d}.npz"
        atomic_npz(chunk_path, arrays)
        detections = int(arrays["frame_detection_offsets"][-1])
        detection_count += detections
        inventory.append(
            {
                "file": f"chunks/{chunk_path.name}",
                "frame_start": frame_start,
                "frame_end": frame_end,
                "detections": detections,
            }
        )
        manifest.update(
            {
                "chunks": inventory,
                "decoded_frame_count": frame_end,
                "detection_count": detection_count,
                "updated_at": utc_now(),
            }
        )
        atomic_json(output / "manifest.json", manifest)

    completed_at = utc_now()
    manifest.update(
        {
            "status": "complete",
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    )
    atomic_json(output / "manifest.json", manifest)
    (output / "_SUCCESS").touch()
    return output, len(markers), detection_count, False


def convert_motion_root(
    motion_root: str | Path,
    output_root: str | Path,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert every direct SKY-Piano imputed CSV with one shared camera."""

    source_root = Path(motion_root)
    inputs = sorted(source_root.glob(f"*{INPUT_SUFFIX}"))
    if not inputs:
        raise FileNotFoundError(
            f"no SKY-Piano imputed motion CSVs found under {source_root}"
        )
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    records = {
        path.name[: -len(INPUT_SUFFIX)]: skypiano_camera_markers(path)
        for path in inputs
    }
    camera = fit_virtual_camera(records)
    dataset_root = Path(output_root) / DATASET_NAME
    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)
    camera_path = dataset_root / "_camera.json"
    if camera_path.is_file():
        existing_camera = json.loads(camera_path.read_text(encoding="utf-8"))
        if existing_camera != camera:
            raise RuntimeError(
                f"shared camera mismatch at {camera_path}; rerun with --overwrite"
            )
    elif dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"SKY-Piano output exists without shared camera metadata at "
            f"{dataset_root}; rerun with --overwrite"
        )
    else:
        atomic_json(camera_path, camera)

    converted = 0
    reused = 0
    total_frames = 0
    total_detections = 0
    for input_path in inputs:
        record_id = input_path.name[: -len(INPUT_SUFFIX)]
        output, frames, detections, was_reused = convert_motion_csv(
            input_path,
            dataset_root / record_id,
            camera,
            chunk_frames=chunk_frames,
        )
        converted += int(not was_reused)
        reused += int(was_reused)
        total_frames += frames
        total_detections += detections
        print(f"SKY-Piano ragged: {record_id} -> {output}", flush=True)
    return {
        "records": len(inputs),
        "converted": converted,
        "reused": reused,
        "frames": total_frames,
        "detections": total_detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=DEFAULT_MOTION_ROOT,
        help="directory containing direct *_imputed.csv files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination root; records are written below SKY-Piano/",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="number of motion frames per compressed chunk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the complete SKY-Piano destination dataset",
    )
    args = parser.parse_args()

    summary = convert_motion_root(
        args.motion_root,
        args.output_root,
        chunk_frames=args.chunk_frames,
        overwrite=args.overwrite,
    )
    print(
        "SKY-Piano ragged conversion complete: "
        f"records={summary['records']} converted={summary['converted']} "
        f"reused={summary['reused']} frames={summary['frames']} "
        f"detections={summary['detections']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
