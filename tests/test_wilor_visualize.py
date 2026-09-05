"""Tests for the native-frame ragged WiLoR video visualizer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from airpiano.preprocessing.wilor.cache import SCHEMA_VERSION
from airpiano.preprocessing.wilor.selection import select_best_hand_indices
from airpiano.preprocessing.wilor import visualize as visualize_module
from airpiano.preprocessing.wilor.visualize import (
    depth_color,
    draw_overlay,
    frame_bounds,
    iter_ragged_frames,
    render_ragged_video,
    selected_depth_range,
)


def _chunk_arrays(
    frame_start: int,
    detections: list[list[tuple[int, float, float, float]]],
) -> dict[str, np.ndarray]:
    flat = [detection for frame in detections for detection in frame]
    offsets = np.zeros(len(detections) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(frame) for frame in detections])
    count = len(flat)
    boxes = np.zeros((count, 4), dtype=np.float32)
    confidence = np.empty(count, dtype=np.float32)
    handedness = np.empty(count, dtype=np.uint8)
    points = np.empty((count, 21, 2), dtype=np.float32)
    joints = np.zeros((count, 21, 3), dtype=np.float32)
    camera = np.zeros((count, 3), dtype=np.float32)
    joint_index = np.arange(21, dtype=np.float32)
    for index, (is_right, score, local_z, camera_z) in enumerate(flat):
        handedness[index] = is_right
        confidence[index] = score
        boxes[index] = [4 + index, 5 + index, 42 + index, 43 + index]
        points[index, :, 0] = 10 + index + joint_index
        points[index, :, 1] = 12 + joint_index
        joints[index, :, 2] = local_z + joint_index * 0.01
        camera[index, 2] = camera_z
    return {
        "frame_indices": np.arange(
            frame_start, frame_start + len(detections), dtype=np.int64
        ),
        "frame_timestamps_seconds": np.arange(
            frame_start, frame_start + len(detections), dtype=np.float64
        ) / 10.0,
        "frame_detection_offsets": offsets,
        "boxes_xyxy": boxes,
        "detector_confidence": confidence,
        "is_right": handedness,
        "pred_keypoints_3d": joints,
        "pred_cam_t_full": camera,
        "pred_keypoints_2d_full": points,
    }


def _write_cache(root: Path, *, schema: str = SCHEMA_VERSION) -> Path:
    raw = root / "raw"
    chunks = raw / "chunks"
    chunks.mkdir(parents=True)
    groups = (
        [
            [(0, 0.2, -100.0, 0.0), (0, 0.9, 0.1, 1.0), (1, 0.8, 0.2, 2.0)],
            [],
        ],
        [
            [(0, 0.7, 0.3, 3.0)],
            [(1, 0.6, 0.4, 4.0)],
        ],
    )
    inventory = []
    for chunk_index, detections in enumerate(groups):
        frame_start = chunk_index * 2
        arrays = _chunk_arrays(frame_start, detections)
        path = chunks / f"{chunk_index:06d}.npz"
        np.savez_compressed(path, **arrays)
        inventory.append(
            {
                "file": f"chunks/{path.name}",
                "frame_start": frame_start,
                "frame_end": frame_start + 2,
                "detections": int(arrays["frame_detection_offsets"][-1]),
            }
        )
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "status": "processing",
                "video": {
                    "codec_name": "h264",
                    "width": 64,
                    "height": 48,
                    "fps": 10.0,
                },
                "chunks": inventory,
            }
        ),
        encoding="utf-8",
    )
    return raw


def _write_source_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=10:d=0.4",
            "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.4",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )


class RaggedWiLoRVisualizeTest(unittest.TestCase):
    def test_selection_keeps_best_hands_and_stable_ties(self) -> None:
        selected = select_best_hand_indices(
            np.asarray([0, 4, 4], dtype=np.int64),
            np.asarray([0.6, 0.9, 0.9, 0.7], dtype=np.float32),
            np.asarray([1, 0, 0, 1], dtype=np.uint8),
        )
        np.testing.assert_array_equal(selected, [[1, 3], [-1, -1]])

    def test_frames_depth_range_overlay_and_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = _write_cache(Path(directory))
            manifest = json.loads((raw / "manifest.json").read_text())
            chunks = tuple(manifest["chunks"])
            frames = list(iter_ragged_frames(raw, chunks, 0, 4))

        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2, 3])
        np.testing.assert_array_equal(frames[0].selected_indices, [1, 2])
        np.testing.assert_array_equal(frames[1].selected_indices, [-1, -1])
        np.testing.assert_allclose(
            selected_depth_range(iter(frames)), (1.1, 4.6), atol=1e-6
        )
        self.assertNotEqual(depth_color(1.1, (1.1, 4.6)), depth_color(4.6, (1.1, 4.6)))

        source = np.full((48, 64, 3), 127, dtype=np.uint8)
        rendered = draw_overlay(source, frames[0], 10.0, (1.1, 4.6))
        no_hands = draw_overlay(source, frames[1], 10.0, (1.1, 4.6))
        self.assertEqual(rendered.shape, source.shape)
        self.assertFalse(np.array_equal(rendered, source))
        self.assertFalse(np.array_equal(no_hands, source))

    def test_frame_bounds_are_half_open_and_default_to_published_cache(self) -> None:
        chunks = (
            {"frame_start": 0, "frame_end": 2},
            {"frame_start": 2, "frame_end": 4},
        )
        self.assertEqual(frame_bounds(chunks, 10.0, 0.0, None), (0, 4))
        self.assertEqual(frame_bounds(chunks, 10.0, 0.1, 0.2), (1, 3))
        with self.assertRaises(ValueError):
            frame_bounds(chunks, 10.0, 1.0, None)

    def test_render_accepts_processing_cache_and_preserves_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            output = root / "rendered.mp4"
            raw = _write_cache(root)
            _write_source_video(video)

            result = render_ragged_video(video, raw, output)
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_streams",
                    "-of", "json", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]

            self.assertEqual(result, output.resolve())
            video_stream = next(
                stream for stream in streams if stream["codec_type"] == "video"
            )
            self.assertEqual((video_stream["width"], video_stream["height"]), (64, 48))
            self.assertEqual(int(video_stream["nb_frames"]), 4)
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))
            self.assertEqual(list(root.glob(".*.tmp.mp4")), [])

            with self.assertRaises(FileExistsError):
                render_ragged_video(video, raw, output)
            self.assertEqual(render_ragged_video(video, raw, output, overwrite=True), output.resolve())

    def test_legacy_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_cache(root, schema="pianomotion-wilor-v1")
            with self.assertRaisesRegex(
                ValueError, "airpiano-wilor-ragged-v1.*pianomotion-wilor-v1"
            ):
                render_ragged_video(root / "unused.mp4", raw, root / "output.mp4")

    def test_encoder_failure_leaves_no_output_or_temporary_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            output = root / "rendered.mp4"
            raw = _write_cache(root)
            _write_source_video(video)
            failing_command = [
                sys.executable,
                "-c",
                "import sys; sys.stdin.buffer.read(); raise SystemExit(7)",
            ]

            with (
                patch.object(
                    visualize_module,
                    "_ffmpeg_command",
                    return_value=failing_command,
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                render_ragged_video(video, raw, output)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".*.tmp.mp4")), [])


if __name__ == "__main__":
    unittest.main()
