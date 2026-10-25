#!/usr/bin/env python3
"""Overlay raw WiLoR skeletons on a recording, retaining source audio.

Example (from the repository root, in the project Python environment)::

    python scripts/inspect_wilor.py AIR/jinxuan_tabletop_aligned/beethoven_8_3

Defaults match configs/default.yaml. Override --video-root and --wilor-root
for another dataset, and --output for a specific MP4 destination. Requires
NumPy, OpenCV, FFmpeg and ffprobe; no model weights or GPU are needed.

All raw detections are shown by default, including duplicates. Colors encode
estimated camera-axis depth relative to each hand's wrist, on a fixed scale:
red = closer, yellow = wrist depth, blue = farther. Depth is not key height.
The default video shows only bold skeletons. Use --annotations for labels
and a depth legend, or --line-width to adjust bone thickness in pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.preprocessing.canonical import LowestWristHandSelector
from airpiano.preprocessing.wilor.extraction import (
    atomic_json,
    completed_output_is_valid,
    open_video_reader,
    probe_video,
    read_manifest,
)


BONES = tuple(
    (a, b)
    for first in (1, 5, 9, 13, 17)
    for a, b in zip((0, first, first + 1, first + 2), range(first, first + 4))
)
CACHE_FIELDS = (
    "frame_indices", "frame_timestamps_seconds", "frame_detection_offsets",
    "pred_keypoints_2d_full", "pred_keypoints_3d", "is_right",
    "detector_confidence", "boxes_xyxy", "image_size_wh",
)


def cache_frames(directory, manifest):
    """Stream validated chunks; offsets are local to each chunk."""
    expected = 0
    width, height = manifest["video"]["width"], manifest["video"]["height"]
    previous_time = -math.inf
    for chunk in manifest["chunks"]:
        with np.load(directory / chunk["file"], allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in CACHE_FIELDS}
        indices = arrays["frame_indices"]
        offsets = arrays["frame_detection_offsets"]
        count = len(arrays["is_right"])
        if not np.array_equal(indices, np.arange(expected, expected + len(indices))):
            raise ValueError(f"Non-contiguous frame indices in {chunk['file']}")
        if (chunk["frame_start"] != expected
                or chunk["frame_end"] != expected + len(indices)
                or chunk["detections"] != count):
            raise ValueError(f"Chunk inventory mismatch in {chunk['file']}")
        if (offsets.shape != (len(indices) + 1,) or offsets[0] != 0
                or offsets[-1] != count or np.any(np.diff(offsets) < 0)):
            raise ValueError(f"Invalid detection offsets in {chunk['file']}")
        shapes = {
            "pred_keypoints_2d_full": (count, 21, 2),
            "pred_keypoints_3d": (count, 21, 3),
            "is_right": (count,), "detector_confidence": (count,),
            "boxes_xyxy": (count, 4), "image_size_wh": (count, 2),
            "frame_timestamps_seconds": (len(indices),),
        }
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"Unexpected {key} shape in {chunk['file']}")
        if not np.all(arrays["image_size_wh"] == (width, height)):
            raise ValueError("Cached joints do not match the video dimensions")
        selected = LowestWristHandSelector()(arrays)
        for local_index, index in enumerate(indices):
            timestamp = float(arrays["frame_timestamps_seconds"][local_index])
            if not math.isfinite(timestamp) or timestamp <= previous_time:
                raise ValueError(f"Invalid timestamp at cache frame {index}")
            previous_time = timestamp
            yield int(index), timestamp, arrays, local_index, selected[local_index]
        expected += len(indices)
    if expected != manifest["decoded_frame_count"]:
        raise ValueError("Cache frame count disagrees with its manifest")


def depth_colors(depth_mm, limit_mm):
    """Saturated red/orange/yellow/cyan/blue depth map, in OpenCV BGR order."""
    values = np.clip(np.asarray(depth_mm, dtype=float) / limit_mm, -1, 1)
    stops = [-1, -0.5, 0, 0.5, 1]
    palette = np.array([
        [20, 20, 255], [0, 140, 255], [0, 245, 255],
        [255, 240, 0], [255, 60, 0],
    ])
    return np.rint(np.stack([
        np.interp(values, stops, palette[:, channel]) for channel in range(3)
    ], axis=-1)).astype(np.uint8)


def label(frame, text, position, scale=0.65, color=(245, 245, 245)):
    # Some OpenCV versions change text spacing with stroke thickness, so use
    # a backing rectangle instead of drawing the text twice for an outline.
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    x, y = position
    cv2.rectangle(frame, (x - 3, y - height - 3),
                  (x + width + 3, y + baseline + 3), (25, 25, 25), -1)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_hand(frame, arrays, detection, limit_mm, boxes=False, *,
              annotations=False, line_width=6):
    # Stored full-image joints and left-hand x coordinates are already corrected.
    xy = arrays["pred_keypoints_2d_full"][detection]
    xyz = arrays["pred_keypoints_3d"][detection]
    z_mm = (xyz[:, 2] - xyz[0, 2]) * 1000
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(z_mm)
    # Guard OpenCV's integer conversion for pathological model predictions.
    valid &= (np.abs(xy) < 1e7).all(axis=1)
    points = np.rint(np.where(valid[:, None], xy, 0)).astype(np.int32)
    colors = depth_colors(np.where(valid, z_mm, 0), limit_mm)
    thickness = line_width
    for start, end in BONES:
        if not (valid[start] and valid[end]):
            continue
        cv2.line(frame, tuple(points[start]), tuple(points[end]),
                 (15, 15, 15), thickness + 4, cv2.LINE_AA)
        # Interpolate color along each bone so its endpoints match joint colors.
        for step in range(8):
            t0, t1 = step / 8, (step + 1) / 8
            p0 = np.rint(points[start] * (1 - t0) + points[end] * t0).astype(int)
            p1 = np.rint(points[start] * (1 - t1) + points[end] * t1).astype(int)
            z = z_mm[start] * (1 - (t0 + t1) / 2) + z_mm[end] * (t0 + t1) / 2
            color = tuple(int(c) for c in depth_colors(z, limit_mm))
            cv2.line(frame, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)
    for joint in np.flatnonzero(valid):
        point = tuple(points[joint])
        radius = thickness + (1 if joint in (0, 4, 8, 12, 16, 20) else 0)
        cv2.circle(frame, point, radius + 1, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, tuple(int(c) for c in colors[joint]), -1, cv2.LINE_AA)
    if boxes:
        box = arrays["boxes_xyxy"][detection]
        if np.isfinite(box).all() and (np.abs(box) < 1e7).all():
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    if annotations and valid[0]:
        side = "R" if arrays["is_right"][detection] else "L"
        confidence = arrays["detector_confidence"][detection]
        x, y = points[0]
        position = (int(np.clip(x - 35, 5, max(5, frame.shape[1] - 150))),
                    int(np.clip(y + 28, 25, frame.shape[0] - 10)))
        label(frame, f"{side} {confidence:.2f}", position)


def draw_hud(frame, record_id, index, timestamp, shown, raw, limit_mm):
    """Keep the legend at the top, clear of the keyboard and hands."""
    width = frame.shape[1]
    # Render at a fixed design size, then fit to smaller source videos.
    panel = np.full((155, 1100, 3), (25, 25, 25), dtype=np.uint8)
    label(panel, f"WiLoR | {record_id}", (18, 30), 0.67)
    label(panel, f"Frame {index:05d}   Time {timestamp:6.2f}s   Hands {shown}/{raw} raw",
          (18, 59), 0.6)
    label(panel, "Estimated Z relative to each wrist (mm)", (18, 87), 0.55)
    gradient = depth_colors(np.linspace(-limit_mm, limit_mm, 460), limit_mm)
    panel[99:115, 18:478] = gradient[None, :, :]
    label(panel, f"-{limit_mm:g} closer", (18, 141), 0.5)
    label(panel, "0", (243, 141), 0.5)
    label(panel, f"+{limit_mm:g} farther", (365, 141), 0.5)
    label(panel, "L/R = anatomical hand; label = detector confidence", (510, 111), 0.52)
    scale = min(1.0, (width - 20) / panel.shape[1], (frame.shape[0] - 20) / 155)
    if scale <= 0:
        return
    if scale < 1:
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = panel.shape[:2]
    frame[10:10 + h, 10:10 + w] = cv2.addWeighted(
        frame[10:10 + h, 10:10 + w], 0.12, panel, 0.88, 0
    )


def render(args):
    record = Path(args.record_id)
    if record.is_absolute() or ".." in record.parts or record.suffix:
        raise ValueError("record_id must be a relative recording path without an extension")
    video = (args.video_root / record).with_suffix(".mp4").resolve()
    directory = (args.wilor_root / record).resolve()
    output = (args.output or PROJECT_ROOT / "assets/inspect" /
              ("_".join(record.parts) + "_wilor.mp4")).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if output == video:
        raise ValueError("Output must differ from the source video")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Use --overwrite to replace {output}")
    if not completed_output_is_valid(directory):
        raise ValueError(f"A completed raw WiLoR cache is required: {directory}")
    manifest = read_manifest(directory)
    metadata = manifest["video"]
    source = probe_video(video)
    for key in ("width", "height"):
        if source[key] != metadata[key]:
            raise ValueError(f"Source video {key} does not match the cache")
    if not math.isclose(source["fps"], metadata.get("source_fps", metadata["fps"]), rel_tol=1e-5):
        raise ValueError("Source video FPS does not match the cache")
    if abs(source["duration_seconds"] - metadata["duration_seconds"]) > 1 / source["fps"]:
        raise ValueError("Source duration does not match the cache")
    fps = float(metadata["fps"])
    total = min(args.max_frames or manifest["decoded_frame_count"], manifest["decoded_frame_count"])
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_id": args.record_id, "video": str(video), "cache": str(directory),
        "output": str(output), "fps": fps, "frames": 0,
        "depth": {"mode": "camera_z_relative_to_wrist", "units": "mm",
                  "range": [-args.depth_range_mm, args.depth_range_mm],
                  "near_color": "red", "wrist_color": "yellow", "far_color": "blue"},
        "appearance": {"annotations": args.annotations, "line_width": args.line_width,
                       "boxes": args.boxes},
        "selection": args.selection, "min_confidence": args.min_confidence,
        "raw_detections": 0, "shown_detections": 0,
        "frames_without_detections": 0, "frames_with_extra_detections": 0,
        "frames_without_left_detection": 0, "frames_without_right_detection": 0,
    }
    with tempfile.TemporaryDirectory(prefix=".wilor-inspect-", dir=output.parent) as temporary:
        temporary_output = Path(temporary) / "overlay.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{metadata['width']}x{metadata['height']}", "-r", str(fps),
            "-i", "pipe:0", "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
            "-threads", "4", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            # Only shorten source audio for preview renders.
            *(["-t", f"{total / fps:.9f}"]
              if total < manifest["decoded_frame_count"] else []),
            "-movflags", "+faststart", str(temporary_output),
        ]
        reader = open_video_reader(video, metadata)
        encoder = None
        try:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            for index, timestamp, arrays, local_index, selected in cache_frames(directory, manifest):
                if index >= total:
                    break
                ok, frame = reader.read()
                if not ok:
                    raise RuntimeError(f"Video decoding ended early at frame {index}")
                start, end = arrays["frame_detection_offsets"][local_index:local_index + 2]
                candidates = range(start, end) if args.selection == "all" else selected[selected >= 0]
                detections = [int(i) for i in candidates
                              if arrays["detector_confidence"][i] >= args.min_confidence]
                for detection in detections:
                    draw_hand(frame, arrays, detection, args.depth_range_mm, args.boxes,
                              annotations=args.annotations, line_width=args.line_width)
                raw = int(end - start)
                if args.annotations:
                    draw_hud(frame, args.record_id, index, timestamp, len(detections), raw,
                             args.depth_range_mm)
                encoder.stdin.write(frame.tobytes())
                summary["frames"] += 1
                summary["raw_detections"] += raw
                summary["shown_detections"] += len(detections)
                summary["frames_without_detections"] += int(raw == 0)
                summary["frames_with_extra_detections"] += int(raw > 2)
                sides = arrays["is_right"][start:end]
                summary["frames_without_left_detection"] += int(not np.any(sides == 0))
                summary["frames_without_right_detection"] += int(not np.any(sides == 1))
                if (index + 1) % 150 == 0 or index + 1 == total:
                    print(f"[render] {index + 1}/{total} frames", flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg could not encode the overlay video")
            if summary["frames"] != total:
                raise RuntimeError(f"Rendered {summary['frames']} frames, expected {total}")
        finally:
            reader.release()
            if encoder is not None and encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
        encoded = probe_video(temporary_output)
        if encoded["declared_frame_count"] != total:
            raise RuntimeError("Encoded frame count does not match the requested video")
        temporary_output.replace(output)
    summary["duration_seconds"] = total / fps
    atomic_json(output.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_id", help="Dataset-relative recording path without .mp4")
    parser.add_argument("--video-root", type=Path, default=Path("/storage/jinxuan/data/AirPianoFull/Video"))
    parser.add_argument("--wilor-root", type=Path, default=Path("/storage2/jinxuan/data/AirPianoFull/wilor"))
    parser.add_argument("--output", type=Path, help="MP4 path; defaults to assets/inspect/<record>_wilor.mp4")
    parser.add_argument("--depth-range-mm", type=float, default=60, help="Fixed symmetric wrist-relative depth range (default: +/-60 mm)")
    parser.add_argument("--selection", choices=("all", "lowest-wrist"), default="all", help="All detections or the pair chosen by canonical preprocessing")
    parser.add_argument("--min-confidence", type=float, default=0, help="Optional detector confidence filter")
    parser.add_argument("--annotations", action="store_true", help="Add text labels and a depth legend; default is skeleton only")
    parser.add_argument("--line-width", type=int, default=6, help="Skeleton line thickness in pixels (default: 6)")
    parser.add_argument("--boxes", action="store_true", help="Also draw detector bounding boxes")
    parser.add_argument("--max-frames", type=int, default=0, help="Render a short preview; 0 renders the full recording")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (0-51; lower is higher quality)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.depth_range_mm) or args.depth_range_mm <= 0:
        parser.error("--depth-range-mm must be finite and positive")
    if args.line_width < 1:
        parser.error("--line-width must be positive")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    if args.max_frames < 0 or not 0 <= args.crf <= 51:
        parser.error("--max-frames must be nonnegative and --crf must be between 0 and 51")
    render(args)


if __name__ == "__main__":
    main()
