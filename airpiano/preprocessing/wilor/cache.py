"""Resumable, atomically published storage for ragged WiLoR predictions."""

from __future__ import annotations

import json
import math
import os
import select
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from airpiano.preprocessing.wilor.backend import WiLoRBackend


SCHEMA_VERSION = "airpiano-wilor-ragged-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    """Publish JSON only after its complete contents reach the filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish a compressed chunk only after it is completely written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    """Read codec, dimensions, duration, and source FPS through ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,duration,time_base"
        ),
        "-show_entries",
        "format=duration,format_name,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream = payload["streams"][0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(
        stream.get("r_frame_rate")
    )
    if not fps or fps <= 0:
        raise RuntimeError(f"could not determine frame rate for {path}")
    return {
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "fps": fps,
        "time_base": stream.get("time_base"),
        "declared_frame_count": (
            int(stream["nb_frames"])
            if stream.get("nb_frames", "N/A") != "N/A"
            else None
        ),
        "duration_seconds": float(
            stream.get("duration")
            or payload.get("format", {}).get("duration")
            or 0
        ),
        "container": payload.get("format", {}).get("format_name"),
        "size_bytes": int(
            payload.get("format", {}).get("size") or path.stat().st_size
        ),
    }


class OpenCVVideoReader:
    """Sequential OpenCV decoder retaining source timestamps where available."""

    def __init__(self, path: Path, fps: float) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        self.fps = fps
        self.frames_read = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if ok:
            self.frames_read += 1
        return ok, frame

    def grab(self) -> bool:
        ok = self.capture.grab()
        if ok:
            self.frames_read += 1
        return ok

    def timestamp_seconds(self) -> float:
        position_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms >= 0:
            return position_ms / 1000.0
        return (self.frames_read - 1) / self.fps

    def release(self) -> None:
        self.capture.release()


class FFmpegVideoReader:
    """Decode frames through system FFmpeg when OpenCV lacks a usable codec."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3
        self.frames_read = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self.process.stdout is None:
            raise RuntimeError("failed to open FFmpeg output pipe")

    def _read_frame_bytes(self) -> bytes | None:
        blocks = bytearray()
        descriptor = self.process.stdout.fileno()
        while len(blocks) < self.frame_bytes:
            readable, _, _ = select.select([descriptor], [], [], 60.0)
            if not readable:
                return_code = self.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"FFmpeg video decoder exited with status {return_code}"
                    )
                raise RuntimeError(
                    "FFmpeg video decoder produced no data for 60 seconds"
                )
            block = os.read(
                descriptor,
                min(1024 * 1024, self.frame_bytes - len(blocks)),
            )
            if not block:
                break
            blocks.extend(block)
        if not blocks:
            return None
        if len(blocks) != self.frame_bytes:
            raise RuntimeError(
                f"FFmpeg returned a partial frame "
                f"({len(blocks)}/{self.frame_bytes} bytes)"
            )
        self.frames_read += 1
        return bytes(blocks)

    def _check_eof(self) -> None:
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg video decoder exited with status {return_code}"
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        data = self._read_frame_bytes()
        if data is None:
            self._check_eof()
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return True, frame.copy()

    def grab(self) -> bool:
        data = self._read_frame_bytes()
        if data is None:
            self._check_eof()
            return False
        return True

    def timestamp_seconds(self) -> float:
        return (self.frames_read - 1) / self.fps

    def release(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def open_video_reader(
    path: Path, metadata: dict[str, Any]
) -> OpenCVVideoReader | FFmpegVideoReader:
    """Use OpenCV normally and FFmpeg for AV1 or unavailable OpenCV codecs."""

    if metadata.get("codec_name") == "av1":
        return FFmpegVideoReader(
            path, metadata["width"], metadata["height"], metadata["fps"]
        )
    try:
        return OpenCVVideoReader(path, metadata["fps"])
    except RuntimeError:
        return FFmpegVideoReader(
            path, metadata["width"], metadata["height"], metadata["fps"]
        )


def require_schema(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Reject every raw-cache schema except AirPiano's current contract."""

    actual = manifest["schema_version"]
    if actual != SCHEMA_VERSION:
        raise ValueError(
            f"raw WiLoR cache schema mismatch at {manifest_path}: "
            f"expected {SCHEMA_VERSION!r}, found {actual!r}"
        )


def read_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require_schema(manifest, path)
    return manifest


def completed_output_is_valid(output_dir: str | Path) -> bool:
    """Return completion state while rejecting incompatible cache schemas."""

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_manifest(output)
    if not (output / "_SUCCESS").is_file():
        return False

    decoded = int(manifest.get("decoded_frame_count", 0))
    chunks = manifest.get("chunks", [])
    video = manifest.get("video", {})
    expected = round(
        float(video.get("duration_seconds", 0)) * float(video.get("fps", 0))
    )
    tolerance = max(2, math.ceil(expected * 0.002))
    inventory_end = int(chunks[-1]["frame_end"]) if chunks else 0
    return (
        manifest.get("status") == "complete"
        and decoded > 0
        and inventory_end == decoded
        and (expected <= 0 or decoded >= expected - tolerance)
    )


def existing_chunk_info(path: Path) -> tuple[int, int, int, int]:
    """Return frame count, index bounds, and detections for one saved chunk."""

    with np.load(path, allow_pickle=False) as payload:
        indices = payload["frame_indices"]
        count = int(len(indices))
        first = int(indices[0]) if count else -1
        last = int(indices[-1]) if count else -1
        detections = int(payload["frame_detection_offsets"][-1])
    return count, first, last, detections


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    backend: WiLoRBackend,
    chunk_frames: int = 512,
    overwrite: bool = False,
) -> Path:
    """Infer a video, resuming every completely published source-frame chunk."""

    video = Path(video_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if overwrite and output.exists():
        shutil.rmtree(output)
    elif completed_output_is_valid(output):
        return output

    chunks_directory = output / "chunks"
    chunks_directory.mkdir(parents=True, exist_ok=True)
    success_path = output / "_SUCCESS"
    success_path.unlink(missing_ok=True)
    manifest_path = output / "manifest.json"
    old_manifest = read_manifest(output) if manifest_path.exists() else None

    video_metadata = probe_video(video)
    timestamp = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "processing",
        "video_path": str(video),
        "video": video_metadata,
        "model": backend.model_metadata(),
        "inference": backend.inference_metadata(chunk_frames),
        "vertex_reconstruction": (
            "Run MANO with saved rotation matrices and betas; reflect vertex x "
            "for is_right=0."
        ),
        "chunks": [],
        "created_at": (
            old_manifest.get("created_at", timestamp)
            if old_manifest is not None
            else timestamp
        ),
        "updated_at": timestamp,
    }
    atomic_json(manifest_path, manifest)

    reader = open_video_reader(video, video_metadata)
    frame_index = 0
    last_timestamp = -1.0
    detection_count = 0
    chunk_inventory: list[dict[str, Any]] = []
    try:
        while True:
            # Inventory length advances after a partial final chunk; integer
            # division by chunk size would incorrectly revisit that filename.
            chunk_index = len(chunk_inventory)
            chunk_path = chunks_directory / f"{chunk_index:06d}.npz"
            if chunk_path.exists() and not overwrite:
                count, first, last, saved_detections = existing_chunk_info(chunk_path)
                if count <= 0 or count > chunk_frames:
                    raise RuntimeError(f"invalid existing chunk {chunk_path}")
                if first != frame_index or last != frame_index + count - 1:
                    raise RuntimeError(f"non-contiguous existing chunk {chunk_path}")
                for _ in range(count):
                    if not reader.grab():
                        raise RuntimeError(
                            f"video ended while resuming past {chunk_path}"
                        )
                chunk_inventory.append(
                    {
                        "file": f"chunks/{chunk_path.name}",
                        "frame_start": first,
                        "frame_end": last + 1,
                        "detections": saved_detections,
                    }
                )
                detection_count += saved_detections
                frame_index += count
                last_timestamp = (frame_index - 1) / video_metadata["fps"]
                continue

            indices: list[int] = []
            timestamps: list[float] = []
            frames: list[np.ndarray] = []
            for _ in range(chunk_frames):
                ok, frame = reader.read()
                if not ok:
                    break
                indices.append(frame_index)
                fallback = frame_index / video_metadata["fps"]
                candidate = reader.timestamp_seconds()
                source_timestamp = (
                    candidate
                    if candidate >= 0
                    and (frame_index == 0 or candidate > last_timestamp)
                    else fallback
                )
                timestamps.append(source_timestamp)
                last_timestamp = source_timestamp
                frames.append(frame)
                frame_index += 1
            if not frames:
                break

            arrays = backend.build_chunk(indices, timestamps, frames)
            atomic_npz(chunk_path, arrays)
            detections = int(arrays["frame_detection_offsets"][-1])
            detection_count += detections
            chunk_inventory.append(
                {
                    "file": f"chunks/{chunk_path.name}",
                    "frame_start": indices[0],
                    "frame_end": indices[-1] + 1,
                    "detections": detections,
                }
            )
            manifest.update(
                {
                    "chunks": chunk_inventory,
                    "decoded_frame_count": frame_index,
                    "detection_count": detection_count,
                    "updated_at": utc_now(),
                }
            )
            atomic_json(manifest_path, manifest)
            print(
                f"[WiLoR] {video}: {frame_index} frames, "
                f"{detection_count} hands",
                flush=True,
            )
    finally:
        reader.release()

    expected_frames = round(
        video_metadata["duration_seconds"] * video_metadata["fps"]
    )
    tolerance = max(2, math.ceil(expected_frames * 0.002))
    if expected_frames > 0 and frame_index < expected_frames - tolerance:
        raise RuntimeError(
            f"video decoding ended early: decoded {frame_index} frames, "
            f"expected approximately {expected_frames} (tolerance {tolerance})"
        )

    completed_at = utc_now()
    manifest.update(
        {
            "status": "complete",
            "chunks": chunk_inventory,
            "decoded_frame_count": frame_index,
            "detection_count": detection_count,
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    )
    atomic_json(manifest_path, manifest)
    success_path.touch()
    return output
