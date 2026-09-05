from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from airpiano.preprocessing import pipeline
from airpiano.preprocessing.canonical import (
    CANONICAL_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    WiLoRCoordinateConverter,
    WiLoRToCanonical,
    raw_cache_has_markers,
)


def _chunk_arrays(frame_start: int, frame_end: int, detections: list[list[tuple]]) -> dict:
    frame_groups = detections[frame_start:frame_end]
    flat = [item for group in frame_groups for item in group]
    offsets = np.zeros(len(frame_groups) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(group) for group in frame_groups])
    count = len(flat)
    confidence = np.asarray([item[1] for item in flat], dtype=np.float32)
    handedness = np.asarray([item[0] for item in flat], dtype=np.uint8)
    keypoints_3d = np.zeros((count, 21, 3), dtype=np.float32)
    keypoints_2d = np.zeros((count, 21, 2), dtype=np.float32)
    image_size = np.tile(np.asarray([100.0, 200.0], dtype=np.float32), (count, 1))
    for index, (_, _, x, y, z_base) in enumerate(flat):
        joint = np.arange(21, dtype=np.float32)
        keypoints_2d[index, :, 0] = x
        keypoints_2d[index, :, 1] = y
        keypoints_3d[index, :, 0] = x * 0.01 + joint * 0.01
        keypoints_3d[index, :, 1] = y * 0.01 + joint * 0.02
        keypoints_3d[index, :, 2] = z_base + joint * 0.01
    return {
        "frame_indices": np.arange(frame_start, frame_end, dtype=np.int64),
        "frame_timestamps_seconds": np.arange(
            frame_start, frame_end, dtype=np.float64
        )
        / 60.0,
        "frame_detection_offsets": offsets,
        "detector_confidence": confidence,
        "is_right": handedness,
        "pred_keypoints_3d": keypoints_3d,
        "pred_keypoints_2d_full": keypoints_2d,
        "image_size_wh": image_size,
    }


def _write_raw_cache(path: Path, schema_version: str = RAW_SCHEMA_VERSION) -> None:
    path.mkdir(parents=True)
    (path / "chunks").mkdir()
    # Entries are (is_right, confidence, full_x, full_y, local_z_base).
    # Frame zero has an exact left-hand confidence tie; the first must win.
    detections = [
        [(0, 0.9, 20, 40, 1), (0, 0.9, 90, 180, 2), (1, 0.8, 60, 80, 3)],
        [(0, 0.7, 40, 60, 2), (1, 0.6, 80, 100, 4)],
        [(0, 0.8, 30, 50, 3), (1, 0.8, 50, 70, 5)],
        [(0, 0.8, 70, 90, 4)],
    ]
    chunks = []
    total_detections = 0
    for index, (start, end) in enumerate(((0, 2), (2, 4))):
        arrays = _chunk_arrays(start, end, detections)
        chunk_path = path / "chunks" / f"{index:06d}.npz"
        np.savez_compressed(chunk_path, **arrays)
        count = int(arrays["frame_detection_offsets"][-1])
        total_detections += count
        chunks.append(
            {
                "file": f"chunks/{chunk_path.name}",
                "frame_start": start,
                "frame_end": end,
                "detections": count,
            }
        )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "status": "complete",
                "decoded_frame_count": 4,
                "detection_count": total_detections,
                "video": {"fps": 60.0},
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )
    (path / "_SUCCESS").touch()


class PreprocessingTest(unittest.TestCase):
    def test_pipeline_reads_unique_records_and_constructs_only_selected_tasks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text(
                "# comment\nsubject/one\n\nsubject/two with space\n",
                encoding="utf-8",
            )
            second.write_text(
                "subject/one\nsubject/session.v1/take.03\n",
                encoding="utf-8",
            )
            self.assertEqual(
                pipeline.read_record_stems([first, second]),
                [
                    "subject/one",
                    "subject/two with space",
                    "subject/session.v1/take.03",
                ],
            )

            empty = root / "empty.txt"
            empty.write_text("", encoding="utf-8")
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "task": "video_to_ragged",
                        "split_files": [str(empty)],
                        "overwrite": False,
                        "video_to_ragged": {"_target_": "example.Video"},
                        "ragged_to_canonical": {"_target_": "example.Canonical"},
                    },
                }
            )
            constructed: list[str] = []

            def construct(node):
                constructed.append(str(node._target_))
                return object()

            expected = {
                "video_to_ragged": ["example.Video"],
                "ragged_to_canonical": ["example.Canonical"],
                "all": ["example.Video", "example.Canonical"],
            }
            with (
                patch.object(pipeline, "instantiate", side_effect=construct),
                redirect_stdout(io.StringIO()),
            ):
                for task, targets in expected.items():
                    config.preprocess.task = task
                    constructed.clear()
                    pipeline.preprocess(config)
                    self.assertEqual(constructed, targets)

    def test_pipeline_appends_extensions_to_root_relative_stems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_id = "performer one/session.v1/take.03"
            split = root / "split.txt"
            split.write_text(f"{record_id}\n", encoding="utf-8")
            video = root / "video" / f"{record_id}.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            calls: list[tuple[Path, Path, bool]] = []

            class VideoToRagged:
                def run(self, video_path, output_dir, overwrite=False):
                    calls.append((Path(video_path), Path(output_dir), overwrite))
                    return Path(output_dir)

            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "task": "video_to_ragged",
                        "split_files": [str(split)],
                        "overwrite": True,
                        "video_to_ragged": {"_target_": "example.Video"},
                    },
                }
            )

            with (
                patch.object(pipeline, "instantiate", return_value=VideoToRagged()),
                redirect_stdout(io.StringIO()),
            ):
                pipeline.preprocess(config)

            self.assertEqual(
                calls,
                [(video.resolve(), root / "raw" / record_id, True)],
            )

    def test_wilor_to_canonical_selects_stable_hands_and_resamples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw" / "subject" / "video"
            _write_raw_cache(raw)
            output = root / "canonical" / "subject" / "video.npz"

            self.assertEqual(WiLoRToCanonical().run(raw, output), output.resolve())
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files), {"pose", "presence", "fps", "schema_version"}
                )
                pose = payload["pose"]
                presence = payload["presence"]
                self.assertEqual(pose.dtype, np.float32)
                self.assertEqual(pose.shape, (2, 2, 21, 3))
                self.assertEqual(presence.dtype, np.bool_)
                self.assertEqual(presence.tolist(), [[True, True], [True, False]])
                self.assertEqual(float(payload["fps"]), 30.0)
                self.assertEqual(
                    str(payload["schema_version"]), CANONICAL_SCHEMA_VERSION
                )

            # Target frame 0 averages native frames 0 and 1. Stable ties choose
            # x=.2, not the second tied x=.9, on native frame zero.
            np.testing.assert_allclose(
                pose[0, 0, 0], [0.3, 0.25, 0.0], atol=1e-6
            )
            np.testing.assert_allclose(
                pose[0, 1, 0], [0.7, 0.45, 0.0], atol=1e-6
            )
            np.testing.assert_allclose(pose[0, 0, 1, 2], 0.01, atol=1e-6)
            self.assertTrue(np.all(pose[1, 1] == 0.0))

            # Existing canonical output is trusted before raw input is consulted.
            original = output.read_bytes()
            (raw / "manifest.json").write_text("not json", encoding="utf-8")
            self.assertEqual(WiLoRToCanonical().run(raw, output).read_bytes(), original)

    def test_wilor_to_canonical_supports_wrist_relative_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "wrist_xyz.npz"
            _write_raw_cache(raw)

            converter = WiLoRCoordinateConverter(
                xy_source="wrist_relative_3d",
                z_source="wrist_relative_3d",
            )
            WiLoRToCanonical(coordinates=converter).run(raw, output)

            with np.load(output, allow_pickle=False) as payload:
                pose = payload["pose"]
                presence = payload["presence"]

            np.testing.assert_array_equal(presence, [[True, True], [True, False]])
            np.testing.assert_allclose(pose[:, :, 0], 0.0, atol=1e-7)
            np.testing.assert_allclose(
                pose[0, 0, 1], [0.01, 0.02, 0.01], atol=1e-6
            )
            np.testing.assert_allclose(
                pose[0, 1, 1], [0.01, 0.02, 0.01], atol=1e-6
            )
            self.assertTrue(np.all(pose[1, 1] == 0.0))

    def test_raw_cache_marker_check_and_native_manifest_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            self.assertFalse(raw_cache_has_markers(raw))
            raw.mkdir()
            (raw / "manifest.json").write_text("not json", encoding="utf-8")
            self.assertFalse(raw_cache_has_markers(raw))
            (raw / "_SUCCESS").touch()
            self.assertTrue(raw_cache_has_markers(raw))

            with self.assertRaises(json.JSONDecodeError):
                WiLoRToCanonical().run(raw, root / "output.npz")

    def test_canonicalizer_rejects_legacy_raw_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            _write_raw_cache(raw, schema_version="pianomotion-wilor-v1")

            with self.assertRaisesRegex(
                ValueError,
                "airpiano-wilor-ragged-v1.*pianomotion-wilor-v1",
            ):
                WiLoRToCanonical().run(raw, root / "output.npz")

    def test_pipeline_propagates_a_completed_malformed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.txt"
            split.write_text("subject/video\n", encoding="utf-8")
            raw = root / "raw" / "subject" / "video"
            raw.mkdir(parents=True)
            (raw / "manifest.json").write_text("not json", encoding="utf-8")
            (raw / "_SUCCESS").touch()
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "task": "ragged_to_canonical",
                        "split_files": [str(split)],
                        "overwrite": False,
                        "ragged_to_canonical": {
                            "_target_": (
                                "airpiano.preprocessing.canonical."
                                "WiLoRToCanonical"
                            )
                        },
                    },
                }
            )

            with self.assertRaises(json.JSONDecodeError):
                pipeline.preprocess(config)

    def test_pipeline_reuses_canonical_and_skips_missing_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.txt"
            split.write_text(
                "subject/cached\nsubject/incomplete\nsubject/valid\n",
                encoding="utf-8",
            )
            raw_root = root / "raw"
            hand_root = root / "canonical"
            incomplete = raw_root / "subject" / "incomplete"
            valid = raw_root / "subject" / "valid"
            incomplete.mkdir(parents=True)
            valid.mkdir(parents=True)
            (valid / "manifest.json").write_text("{}", encoding="utf-8")
            (valid / "_SUCCESS").touch()
            cached = hand_root / "subject" / "cached.npz"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"trusted canonical fixture")
            converted: list[str] = []

            class RaggedToCanonical:
                def run(self, raw_record_dir, output_path, overwrite=False):
                    converted.append(Path(raw_record_dir).name)
                    output = Path(output_path)
                    if output.exists() and not overwrite:
                        return output
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"canonical fixture")
                    return output

            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "videos"),
                        "raw_wilor_root": str(raw_root),
                        "hand_root": str(hand_root),
                    },
                    "preprocess": {
                        "task": "ragged_to_canonical",
                        "split_files": [str(split)],
                        "overwrite": False,
                        "ragged_to_canonical": {
                            "_target_": (
                                "airpiano.preprocessing.canonical."
                                "WiLoRToCanonical"
                            )
                        },
                    },
                }
            )
            printed = io.StringIO()
            with (
                patch.object(
                    pipeline,
                    "instantiate",
                    return_value=RaggedToCanonical(),
                ),
                redirect_stdout(printed),
            ):
                pipeline.preprocess(config)

            self.assertEqual(converted, ["cached", "valid"])
            self.assertEqual(cached.read_bytes(), b"trusted canonical fixture")
            self.assertTrue((hand_root / "subject" / "valid.npz").is_file())
            report = printed.getvalue()
            self.assertIn("subject/incomplete", report)
            self.assertIn(
                "preprocess complete: requested=3 completed_or_reused=2 missing=1",
                report,
            )


if __name__ == "__main__":
    unittest.main()
