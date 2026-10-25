"""Transparent front and fixed-wrist side views of one canonical hand segment."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from airpiano.preprocessing.canonical import LowestWristHandSelector
from airpiano.preprocessing.wilor.extraction import read_manifest


BONES = tuple(
    (a, b)
    for first in (1, 5, 9, 13, 17)
    for a, b in zip((0, first, first + 1, first + 2), range(first, first + 4))
)
LINE_WIDTH = 6
DEPTH_LIMIT_MM = 50.0


def depth_colors(depth_mm: np.ndarray | float) -> np.ndarray:
    """Map wrist-relative camera Z to saturated BGR, on a fixed +/-50 mm scale."""
    values = np.clip(np.asarray(depth_mm) / DEPTH_LIMIT_MM, -1, 1)
    palette = np.array([
        [20, 20, 255], [0, 140, 255], [0, 245, 255],
        [255, 240, 0], [255, 60, 0],
    ])
    return np.rint(np.stack([
        np.interp(values, [-1, -0.5, 0, 0.5, 1], palette[:, channel])
        for channel in range(3)
    ], axis=-1)).astype(np.uint8)


class _Canvas:
    """Draw matching premultiplied color and coverage, then emit straight BGRA."""

    def __init__(self, width: int, height: int):
        self.color = np.zeros((height, width, 3), dtype=np.uint8)
        self.alpha = np.zeros((height, width), dtype=np.uint8)

    def line(self, start, end, color, thickness):
        start, end = tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))
        cv2.line(self.color, start, end, color, thickness, cv2.LINE_AA)
        cv2.line(self.alpha, start, end, 255, thickness, cv2.LINE_AA)

    def circle(self, point, radius, color):
        point = tuple(np.rint(point).astype(int))
        cv2.circle(self.color, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(self.alpha, point, radius, 255, -1, cv2.LINE_AA)

    def bgra(self):
        # OpenCV interpolates both color and coverage against transparent black.
        # Undo that premultiplication so compositors do not darken AA edges twice.
        alpha = self.alpha.astype(np.float32)
        color = self.color.astype(np.float32)
        straight = color * (255.0 / np.maximum(alpha, 1))[..., None]
        # Extend edge color under transparent pixels for 4:2:0 chroma subsampling.
        blurred_alpha = cv2.blur(alpha, (5, 5))
        edge_color = cv2.blur(color, (5, 5)) * (
            255.0 / np.maximum(blurred_alpha, 1)
        )[..., None]
        straight[alpha == 0] = edge_color[alpha == 0]
        bgra = np.empty((*alpha.shape, 4), dtype=np.uint8)
        bgra[..., :3] = np.rint(np.clip(straight, 0, 255)).astype(np.uint8)
        bgra[..., 3] = self.alpha
        return bgra


def _draw_hand(canvas: _Canvas, xy: np.ndarray, depth_mm: np.ndarray):
    colors = depth_colors(depth_mm)
    for start, end in BONES:
        canvas.line(xy[start], xy[end], (15, 15, 15), LINE_WIDTH + 4)
        for step in range(8):
            t0, t1 = step / 8, (step + 1) / 8
            p0 = xy[start] * (1 - t0) + xy[end] * t0
            p1 = xy[start] * (1 - t1) + xy[end] * t1
            middle = (t0 + t1) / 2
            z = depth_mm[start] * (1 - middle) + depth_mm[end] * middle
            canvas.line(p0, p1, tuple(int(c) for c in depth_colors(z)), LINE_WIDTH)
    for joint, point in enumerate(xy):
        radius = LINE_WIDTH + (joint in (0, 4, 8, 12, 16, 20))
        canvas.circle(point, radius + 1, (15, 15, 15))
        canvas.circle(point, radius, tuple(int(c) for c in colors[joint]))


def _draw_wrist_cross(canvas: _Canvas, anchor: np.ndarray, arm: int):
    """A compact neutral four-headed cross, visible on dark and light backgrounds."""
    for thickness, color in ((5, (15, 15, 15)), (2, (245, 245, 245))):
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vector = np.asarray(direction, dtype=float)
            tip = anchor + vector * arm
            perpendicular = np.array([-vector[1], vector[0]])
            canvas.line(anchor, tip, color, thickness)
            for sign in (-1, 1):
                canvas.line(tip, tip - vector * 6 + perpendicular * sign * 4,
                            color, thickness)


def _read_geometry(raw_dir, canonical_path, frame_start, frame_stop, fps, width, height):
    manifest = read_manifest(raw_dir)
    if manifest["status"] != "complete" or not (raw_dir / "_SUCCESS").is_file():
        raise ValueError("A completed raw WiLoR cache is required")
    metadata = manifest["video"]
    if not math.isclose(float(metadata["fps"]), fps, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Raw WiLoR and output FPS must match; resampling is not implicit")
    if (metadata["width"], metadata["height"]) != (width, height):
        raise ValueError("Front-view dimensions must match original cached video pixels")
    with np.load(canonical_path, allow_pickle=False) as payload:
        if str(payload["schema_version"].item()) != "airpiano-wilor-canonical-v1":
            raise ValueError("Unsupported canonical hand schema")
        if not math.isclose(float(payload["fps"]), fps, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Canonical and output FPS must match; resampling is not implicit")
        config = json.loads(str(payload["canonicalization_config"].item()))
        if config["selector"] != "lowest_wrist" or config["postprocessor"] != "wrist_delta":
            raise ValueError("Expected lowest-wrist canonical selection with wrist-delta roots")
        pose = payload["pose"]
        if not 0 <= frame_start < frame_stop <= len(pose):
            raise ValueError("Requested segment lies outside the canonical hand sequence")
        xyz = pose[frame_start:frame_stop].copy()
        presence = payload["presence"][frame_start:frame_stop].copy()
    if frame_stop > manifest["decoded_frame_count"]:
        raise ValueError("Requested segment lies outside the raw WiLoR sequence")
    # Canonical roots contain displacement, not position; the side wrist is fixed.
    xyz[:, :, 0] = 0
    count = frame_stop - frame_start
    front_xy = np.full((count, 2, 21, 2), np.nan, dtype=np.float32)
    front_depth = np.full((count, 2, 21), np.nan, dtype=np.float32)
    seen = np.zeros(count, dtype=bool)
    fields = ("frame_indices", "frame_detection_offsets", "detector_confidence",
              "is_right", "pred_keypoints_2d_full", "pred_keypoints_3d")
    for chunk in manifest["chunks"]:
        if chunk["frame_end"] <= frame_start or chunk["frame_start"] >= frame_stop:
            continue
        with np.load(raw_dir / chunk["file"], allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in fields}
        selected = LowestWristHandSelector()(arrays)
        for local, source_index in enumerate(arrays["frame_indices"]):
            if not frame_start <= source_index < frame_stop:
                continue
            index = int(source_index) - frame_start
            if seen[index]:
                raise ValueError(f"Duplicate raw frame {source_index}")
            seen[index] = True
            if not np.array_equal(selected[local] >= 0, presence[index]):
                raise ValueError(f"Raw/canonical presence mismatch at frame {source_index}")
            for hand in np.flatnonzero(presence[index]):
                detection = selected[local, hand]
                front_xy[index, hand] = arrays["pred_keypoints_2d_full"][detection]
                raw_xyz = arrays["pred_keypoints_3d"][detection]
                relative = raw_xyz - raw_xyz[:1]
                if not np.allclose(relative, xyz[index, hand], atol=1e-6, rtol=1e-5):
                    raise ValueError(f"Raw/canonical geometry mismatch at frame {source_index}")
                front_depth[index, hand] = relative[:, 2] * 1000
    if not seen.all():
        raise ValueError("Raw cache does not cover every requested canonical frame")
    if not np.isfinite(front_xy[presence]).all() or not np.isfinite(xyz[presence]).all():
        raise ValueError("Nonfinite coordinates in selected hand geometry")
    projected = np.stack((-xyz[..., 1], xyz[..., 2]), axis=-1)
    extent = (np.max(np.abs(projected[presence]), axis=(0, 1))
              if presence.any() else np.zeros(2))
    scale = float(np.min(np.array([width * 0.22, height * 0.42]) / np.maximum(extent, 0.01)))
    anchors = np.array([[width * 0.25, height * 0.5],
                        [width * 0.75, height * 0.5]], dtype=np.float32)
    side_xy = (projected * scale + anchors[None, :, None, :]).astype(np.float32)
    side_xy[~presence] = np.nan
    depth = xyz[..., 2] * 1000
    depth[~presence] = np.nan
    return front_xy, side_xy, presence, anchors, front_depth, depth, scale


def _encode_video(path, frames, fps, width, height):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgra", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-an",
        # Extract packed BGRA alpha before color conversion: some swscale builds
        # round alpha values >=128 upward when converting BGRA directly to YUVA.
        "-filter_complex", "[0:v]split[color][coverage];[color]format=yuv420p[yuv];"
        "[coverage]alphaextract[alpha];[yuv][alpha]alphamerge[out]",
        "-map", "[out]", "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p", "-lossless", "1", "-deadline", "good",
        "-cpu-used", "4", "-row-mt", "1", "-threads", "4", "-auto-alt-ref", "0",
        str(path),
    ]
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, frame in enumerate(frames):
                digest.update(frame[..., 3].tobytes())
                process.stdin.write(frame.tobytes())
                if (index + 1) % 100 == 0:
                    print(f"[hands] {path.name}: {index + 1} frames", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(f"VP9 encoding failed: {errors.read().decode(errors='replace')}")
        except BrokenPipeError as error:
            process.wait()
            errors.seek(0)
            raise RuntimeError(f"VP9 encoder closed its input: {errors.read().decode(errors='replace')}") from error
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
    return digest.hexdigest()


def _verify_alpha(path, width, height, expected_frames, expected_digest):
    # FFmpeg's native VP9 decoder may discard WebM alpha; explicitly use libvpx.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", "libvpx-vp9",
        "-i", str(path), "-vf", "alphaextract", "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1",
    ]
    digest = hashlib.sha256()
    count = transparent = opaque = antialiased = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while data := process.stdout.read(width * height):
                if len(data) != width * height:
                    raise RuntimeError("Truncated alpha frame from VP9 decoder")
                digest.update(data)
                values = np.frombuffer(data, dtype=np.uint8)
                transparent += int(np.count_nonzero(values == 0))
                opaque += int(np.count_nonzero(values == 255))
                antialiased += int(np.count_nonzero((values > 0) & (values < 255)))
                count += 1
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(f"VP9 alpha decoding failed: {errors.read().decode(errors='replace')}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    if count != expected_frames or digest.hexdigest() != expected_digest:
        raise RuntimeError("Encoded video does not preserve every source alpha pixel/frame")
    if not transparent:
        raise RuntimeError("Encoded hand video has no transparent background")
    return {"decoder": "libvpx-vp9", "frames": count, "alpha_sha256": digest.hexdigest(),
            "source_alpha_exact": True, "transparent_pixels": transparent,
            "opaque_pixels": opaque, "antialiased_pixels": antialiased}


def render_hand_videos(*, raw_dir: Path, canonical_path: Path, frame_start: int,
                       frame_stop: int, fps: float, width: int, height: int,
                       output_dir: Path) -> dict:
    """Write transparent VP9 videos and pixel geometry for one unaugmented slice.

    Hand axis order is anatomical left, right. ``hand_geometry.npz`` contains
    ``front_xy``/``side_xy`` [T,2,21,2] pixels (NaN for absent hands), ``presence``
    [T,2], fixed ``anchors`` [2,2] pixels, and wrist-relative ``front_depth_mm``/
    ``depth_mm`` [T,2,21]. Side geometry is screen X=-camera Y, screen Y=camera Z.
    Canonical XYZ are read directly, before the model's scale-by-ten transform.
    """
    if not math.isfinite(fps) or fps <= 0 or width < 64 or height < 64:
        raise ValueError("Positive FPS and dimensions at least 64x64 are required")
    raw_dir, canonical_path, output_dir = Path(raw_dir), Path(canonical_path), Path(output_dir)
    names = ("hands_front.webm", "hands_side.webm", "hand_geometry.npz")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    front, side, presence, anchors, front_depth, depth, scale = _read_geometry(
        raw_dir, canonical_path, frame_start, frame_stop, fps, width, height
    )
    count = frame_stop - frame_start
    arm = max(12, round(min(width, height) * 0.035))

    def frames(view, z, show_axes):
        for index in range(count):
            canvas = _Canvas(width, height)
            for hand in np.flatnonzero(presence[index]):
                _draw_hand(canvas, view[index, hand], z[index, hand])
            if show_axes:
                for anchor in anchors:
                    _draw_wrist_cross(canvas, anchor, arm)
            yield canvas.bgra()

    output_dir.mkdir(parents=True, exist_ok=True)
    validation = {}
    with tempfile.TemporaryDirectory(prefix=".segment-hands-", dir=output_dir) as temporary:
        directory = Path(temporary)
        for name, view, z, axes in ((names[0], front, front_depth, False),
                                    (names[1], side, depth, True)):
            digest = _encode_video(directory / name, frames(view, z, axes), fps, width, height)
            validation[name] = _verify_alpha(directory / name, width, height, count, digest)
        np.savez_compressed(
            directory / names[2], front_xy=front, side_xy=side, presence=presence,
            anchors=anchors, front_depth_mm=front_depth, depth_mm=depth,
            frame_indices=np.arange(frame_start, frame_stop, dtype=np.int64),
            fps=np.asarray(fps), side_scale_pixels_per_meter=np.asarray(scale),
        )
        for name in names:
            (directory / name).replace(output_dir / name)
    return {
        "front_video": names[0], "side_video": names[1], "geometry": names[2],
        "frames": count, "frame_start": frame_start, "frame_stop_exclusive": frame_stop,
        "fps": fps, "width": width, "height": height, "codec": "vp9", "alpha": True,
        "line_width_pixels": LINE_WIDTH, "depth_range_mm": [-DEPTH_LIMIT_MM, DEPTH_LIMIT_MM],
        "depth_colors": {"near": "red", "wrist": "yellow", "far": "blue"},
        "hand_order": ["left", "right"], "presence_counts": presence.sum(axis=0).tolist(),
        "side_projection": "screen_x=-camera_y; screen_y=camera_z; wrist=0",
        "side_scale_pixels_per_meter": scale, "side_anchors_pixels": anchors.tolist(),
        "side_wrist_cross": {"arms": 4, "arm_pixels": arm, "all_frames": True},
        "geometry_keys": {
            "front_xy": "[T,2,21,2] original video pixels; absent hands are NaN",
            "side_xy": "[T,2,21,2] side-view pixels; absent hands are NaN",
            "presence": "[T,2] canonical anatomical left/right presence",
            "anchors": "[2,2] fixed left/right side wrist pixels",
            "front_depth_mm": "[T,2,21] selected raw joint Z minus wrist Z, millimeters",
            "depth_mm": "[T,2,21] unscaled canonical side joint Z, root zero, millimeters",
            "frame_indices": "[T] original canonical frame indices",
            "fps": "scalar frames per second",
            "side_scale_pixels_per_meter": "scalar common side-view scale for both hands/all frames",
        },
        "alpha_validation": validation,
    }
