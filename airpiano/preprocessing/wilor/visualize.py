"""Render ragged WiLoR detections over matching source-video frames."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from airpiano.utils import resolve_path

from .cache import open_video_reader, read_manifest
from .selection import select_best_hand_indices


HAND_NAMES = ("LEFT", "RIGHT")
HAND_COLORS = ((225, 90, 190), (55, 205, 255))  # BGR
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
DEPTH_COLORS = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
)[:, 0]


@dataclass(frozen=True)
class RaggedFrame:
    """All detections and selected-hand indices for one cached video frame."""

    frame_index: int
    timestamp_seconds: float
    boxes_xyxy: np.ndarray
    detector_confidence: np.ndarray
    is_right: np.ndarray
    keypoints_2d_full: np.ndarray
    absolute_z: np.ndarray
    selected_indices: np.ndarray


def frame_bounds(
    chunks: Sequence[dict],
    fps: float,
    start_seconds: float,
    duration_seconds: float | None,
) -> tuple[int, int]:
    """Select cached frames in a half-open wall-clock interval."""

    available_start = int(chunks[0]["frame_start"])
    available_end = int(chunks[-1]["frame_end"])
    requested_start = math.ceil(start_seconds * fps - 1e-9)
    start = max(available_start, requested_start)
    if duration_seconds is None:
        end = available_end
    else:
        end = min(
            available_end,
            math.ceil((start_seconds + duration_seconds) * fps - 1e-9),
        )
    if start_seconds < 0 or (duration_seconds is not None and duration_seconds <= 0):
        raise ValueError("start_seconds must be nonnegative and duration positive")
    if end <= start:
        raise ValueError("requested interval contains no cached frames")
    return start, end


def iter_ragged_frames(
    raw_cache_dir: Path,
    chunks: Sequence[dict],
    start_frame: int,
    end_frame: int,
) -> Iterator[RaggedFrame]:
    """Stream a manifest snapshot without materializing a full performance."""

    for chunk in chunks:
        chunk_start = int(chunk["frame_start"])
        chunk_end = int(chunk["frame_end"])
        if chunk_end <= start_frame or chunk_start >= end_frame:
            continue
        with np.load(raw_cache_dir / chunk["file"], allow_pickle=False) as payload:
            offsets = np.asarray(payload["frame_detection_offsets"])
            confidence = np.asarray(payload["detector_confidence"])
            handedness = np.asarray(payload["is_right"])
            selected = select_best_hand_indices(offsets, confidence, handedness)
            frame_indices = np.asarray(payload["frame_indices"])
            timestamps = np.asarray(payload["frame_timestamps_seconds"])
            boxes = np.asarray(payload["boxes_xyxy"])
            keypoints = np.asarray(payload["pred_keypoints_2d_full"])
            local_z = np.asarray(payload["pred_keypoints_3d"])[..., 2]
            camera_z = np.asarray(payload["pred_cam_t_full"])[:, 2]
            absolute_z = local_z + camera_z[:, None]

            for local_frame, frame_index_value in enumerate(frame_indices):
                frame_index = int(frame_index_value)
                if frame_index < start_frame or frame_index >= end_frame:
                    continue
                detection_start = int(offsets[local_frame])
                detection_end = int(offsets[local_frame + 1])
                frame_selected = selected[local_frame]
                frame_selected = np.where(
                    frame_selected >= 0,
                    frame_selected - detection_start,
                    -1,
                ).astype(np.int64, copy=False)
                yield RaggedFrame(
                    frame_index=frame_index,
                    timestamp_seconds=float(timestamps[local_frame]),
                    boxes_xyxy=boxes[detection_start:detection_end],
                    detector_confidence=confidence[detection_start:detection_end],
                    is_right=handedness[detection_start:detection_end],
                    keypoints_2d_full=keypoints[detection_start:detection_end],
                    absolute_z=absolute_z[detection_start:detection_end],
                    selected_indices=frame_selected,
                )


def selected_depth_range(frames: Iterator[RaggedFrame]) -> tuple[float, float] | None:
    """Return exact depth limits over only the two selected hand tracks."""

    minimum = math.inf
    maximum = -math.inf
    for frame in frames:
        for detection in frame.selected_indices:
            if detection < 0:
                continue
            values = frame.absolute_z[int(detection)]
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    return None if minimum == math.inf else (minimum, maximum)


def depth_color(value: float, limits: tuple[float, float]) -> tuple[int, int, int]:
    """Map smaller/nearer Z to blue and larger/farther Z to red."""

    minimum, maximum = limits
    fraction = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    index = int(round(255 * min(1.0, max(0.0, fraction))))
    return tuple(int(channel) for channel in DEPTH_COLORS[index])


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.5,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (5, 5, 5), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 1, cv2.LINE_AA,
    )


def draw_overlay(
    source_frame: np.ndarray,
    ragged: RaggedFrame,
    fps: float,
    limits: tuple[float, float] | None,
) -> np.ndarray:
    """Draw candidates plus depth-colored selected skeletons on one BGR frame."""

    image = source_frame.copy()
    height, width = image.shape[:2]
    line_width = max(1, round(min(width, height) / 600))
    selected = {int(index) for index in ragged.selected_indices if index >= 0}

    for index, (box, score, handedness) in enumerate(zip(
        ragged.boxes_xyxy,
        ragged.detector_confidence,
        ragged.is_right,
    )):
        slot = int(handedness)
        color = HAND_COLORS[slot]
        x0, y0, x1, y1 = np.rint(box).astype(int)
        thickness = line_width + (2 if index in selected else 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        _put_text(
            image,
            f"{HAND_NAMES[slot][0]} {float(score):.3f}",
            (x0, max(16, y0 - 5)),
            color=color,
            scale=0.4,
        )

    if limits is not None:
        for slot, detection in enumerate(ragged.selected_indices):
            if detection < 0:
                continue
            detection = int(detection)
            points = np.rint(ragged.keypoints_2d_full[detection]).astype(np.int32)
            z_values = ragged.absolute_z[detection]
            for first, second in HAND_CONNECTIONS:
                color = depth_color(
                    float((z_values[first] + z_values[second]) * 0.5), limits
                )
                cv2.line(
                    image, tuple(points[first]), tuple(points[second]),
                    color, line_width + 1, cv2.LINE_AA,
                )
            for joint, z_value in zip(points, z_values):
                cv2.circle(
                    image, tuple(joint), line_width + 3,
                    depth_color(float(z_value), limits), -1, cv2.LINE_AA,
                )
            cv2.circle(
                image, tuple(points[0]), line_width + 5,
                HAND_COLORS[slot], line_width + 1, cv2.LINE_AA,
            )

    panel = image.copy()
    panel_height = min(height, 68)
    cv2.rectangle(panel, (0, 0), (width - 1, panel_height), (10, 12, 18), -1)
    cv2.addWeighted(panel, 0.72, image, 0.28, 0.0, image)
    present = [name for name, index in zip(HAND_NAMES, ragged.selected_indices) if index >= 0]
    _put_text(
        image,
        f"raw WiLoR  frame={ragged.frame_index}  "
        f"t={ragged.frame_index / fps:.3f}s  cached={ragged.timestamp_seconds:.3f}s",
        (12, 25),
        scale=0.52,
    )
    _put_text(
        image,
        f"selected={'+'.join(present) if present else 'NONE'}  "
        f"candidates={len(ragged.boxes_xyxy)}",
        (12, 51),
        scale=0.48,
    )

    if limits is not None and width >= 280 and height >= 100:
        minimum, maximum = limits
        bar_width = min(240, width - 40)
        bar_height = 12
        x0 = width - bar_width - 14
        y0 = height - 29
        gradient = cv2.resize(DEPTH_COLORS.reshape(1, 256, 3), (bar_width, bar_height))
        image[y0:y0 + bar_height, x0:x0 + bar_width] = gradient
        _put_text(image, f"near {minimum:.4f}", (x0, y0 - 4), scale=0.35)
        far = f"far {maximum:.4f}"
        text_width = cv2.getTextSize(
            far, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )[0][0]
        _put_text(image, far, (x0 + bar_width - text_width, y0 - 4), scale=0.35)
    return image


def _ffmpeg_command(
    video_path: Path,
    temporary_output: Path,
    width: int,
    height: int,
    fps: float,
    start_frame: int,
    frame_count: int,
) -> list[str]:
    start = start_frame / fps
    duration = frame_count / fps
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.12g}", "-i", "pipe:0",
        "-ss", f"{start:.12g}", "-i", str(video_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-t", f"{duration:.12g}", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        str(temporary_output),
    ]


def render_ragged_video(
    video_path: str | Path,
    raw_cache_dir: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Overlay one complete or currently published ragged-cache interval."""

    video = resolve_path(video_path)
    raw_cache = resolve_path(raw_cache_dir)
    output = resolve_path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)

    manifest = read_manifest(raw_cache)
    chunks = tuple(dict(chunk) for chunk in manifest["chunks"])
    if not chunks:
        raise ValueError("raw WiLoR manifest has no published chunks")
    metadata = dict(manifest["video"])
    fps = float(metadata["fps"])
    start_frame, end_frame = frame_bounds(
        chunks, fps, start_seconds, duration_seconds
    )
    frame_count = end_frame - start_frame
    limits = selected_depth_range(
        iter_ragged_frames(raw_cache, chunks, start_frame, end_frame)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tmp.mp4", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()

    reader = open_video_reader(video, metadata)
    process: subprocess.Popen[bytes] | None = None
    try:
        for _ in range(start_frame):
            if not reader.grab():
                raise EOFError(f"source video ended before frame {start_frame}")

        command = _ffmpeg_command(
            video,
            temporary,
            int(metadata["width"]),
            int(metadata["height"]),
            fps,
            start_frame,
            frame_count,
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("FFmpeg did not open its raw-video input")

        frames = iter_ragged_frames(raw_cache, chunks, start_frame, end_frame)
        rendered_count = 0
        for ragged in frames:
            expected_frame = start_frame + rendered_count
            if ragged.frame_index != expected_frame:
                raise ValueError(
                    f"ragged frame sequence expected {expected_frame}, "
                    f"found {ragged.frame_index}"
                )
            ok, source_frame = reader.read()
            if not ok or source_frame is None:
                raise EOFError(f"source video ended before frame {expected_frame}")
            expected_shape = (int(metadata["height"]), int(metadata["width"]), 3)
            if source_frame.shape != expected_shape:
                raise ValueError(
                    f"source frame shape {source_frame.shape} != {expected_shape}"
                )
            process.stdin.write(draw_overlay(source_frame, ragged, fps, limits).tobytes())
            rendered_count += 1
        if rendered_count != frame_count:
            raise ValueError(
                f"ragged cache yielded {rendered_count} frames, expected {frame_count}"
            )

        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(
                return_code, command, stderr=stderr
            )
        os.replace(temporary, output)
        return output
    finally:
        reader.release()
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = render_ragged_video(
        args.video,
        args.raw_cache,
        args.output,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
