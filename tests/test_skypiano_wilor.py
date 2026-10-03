from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airpiano.preprocessing.canonical import (
    WiLoRCoordinateConverter,
    WiLoRToCanonical,
    WristDeltaPostprocessor,
)
from airpiano.preprocessing.skypiano import (
    HAND_SIDES,
    JOINT_MARKERS,
    skypiano_camera_markers,
)
from scripts.convert_skypiano_motion import read_skypiano_pose
from scripts.convert_skypiano_wilor import (
    CAMERA_SCHEMA_VERSION,
    DATASET_NAME,
    RAW_SCHEMA_VERSION,
    build_chunk,
    convert_motion_root,
)


def _write_motion_csv(path: Path, record_offset: float) -> None:
    header = [
        f"{side}{marker}_{axis}"
        for side in HAND_SIDES
        for marker in JOINT_MARKERS
        for axis in "xyz"
    ]
    rows = []
    for frame in range(3):
        row = []
        for hand, _side in enumerate(HAND_SIDES):
            for joint, _marker in enumerate(JOINT_MARKERS):
                row.extend(
                    (
                        (-300.0 if hand == 0 else 300.0)
                        + joint * 2.0
                        + frame * 10.0
                        + record_offset,
                        -100.0 + joint * 5.0 + frame * 4.0 + hand * 20.0,
                        20.0 + joint * 2.0 + frame,
                    )
                )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class SkyPianoWiLoRTest(unittest.TestCase):
    def test_build_chunk_omits_a_hand_when_any_joint_is_outside(self) -> None:
        markers = np.zeros((1, 2, 21, 3), dtype=np.float32)
        markers[0, 1, -1, 0] = 10.0
        camera = {
            "image_size_wh": [100, 100],
            "principal_point_xy": [50.0, 50.0],
            "scaled_focal_length": 100.0,
            "translation_xyz_meters": [0.0, 0.0, 10.0],
        }

        arrays = build_chunk(markers, 0, 1, camera)

        np.testing.assert_array_equal(
            arrays["frame_detection_offsets"], [0, 1]
        )
        np.testing.assert_array_equal(arrays["is_right"], [0])
        np.testing.assert_allclose(
            arrays["pred_keypoints_2d_full"], 50.0
        )
        np.testing.assert_allclose(arrays["pred_cam_t_full"], [[0, 0, 10]])

    def test_conversion_reuse_and_all_canonical_coordinate_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            motion_root = root / "motion"
            output_root = root / "wilor"
            motion_root.mkdir()
            first = motion_root / "first_imputed.csv"
            second = motion_root / "second_imputed.csv"
            _write_motion_csv(first, 0.0)
            _write_motion_csv(second, 25.0)

            summary = convert_motion_root(
                motion_root,
                output_root,
                chunk_frames=2,
            )
            self.assertEqual(
                summary,
                {
                    "records": 2,
                    "converted": 2,
                    "reused": 0,
                    "frames": 6,
                    "detections": 12,
                },
            )

            dataset_root = output_root / DATASET_NAME
            camera = json.loads(
                (dataset_root / "_camera.json").read_text(encoding="utf-8")
            )
            self.assertEqual(camera["schema_version"], CAMERA_SCHEMA_VERSION)
            self.assertEqual(camera["scaled_focal_length"], 37500.0)
            raw = dataset_root / "first"
            manifest = json.loads(
                (raw / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], RAW_SCHEMA_VERSION)
            self.assertEqual(manifest["motion"]["fps"], 30.0)
            self.assertEqual(len(manifest["chunks"]), 2)
            self.assertEqual(manifest["detection_count"], 6)

            with np.load(raw / "chunks/000000.npz", allow_pickle=False) as chunk:
                self.assertEqual(
                    set(chunk.files),
                    {
                        "frame_indices",
                        "frame_timestamps_seconds",
                        "frame_detection_offsets",
                        "boxes_xyxy",
                        "detector_confidence",
                        "is_right",
                        "pred_keypoints_3d",
                        "pred_cam_t_full",
                        "pred_keypoints_2d_full",
                        "image_size_wh",
                    },
                )
                self.assertEqual(chunk["pred_keypoints_3d"].dtype, np.float32)
                self.assertEqual(chunk["frame_indices"].dtype, np.int64)
                self.assertTrue(
                    np.all((chunk["pred_keypoints_2d_full"] >= 0.0))
                )
                self.assertTrue(
                    np.all(
                        chunk["pred_keypoints_2d_full"]
                        < chunk["image_size_wh"][:, None]
                    )
                )

            expected_3d, expected_presence = read_skypiano_pose(first)
            xyz_output = root / "xyz.npz"
            xyz_converter = WiLoRCoordinateConverter(
                xy_source="wrist_relative_3d",
                z_source="wrist_relative_3d",
            )
            WiLoRToCanonical(coordinates=xyz_converter).run(raw, xyz_output)
            with np.load(xyz_output, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["presence"], expected_presence
                )
                np.testing.assert_allclose(payload["pose"], expected_3d)

            image_output = root / "image.npz"
            WiLoRToCanonical().run(raw, image_output)
            with np.load(image_output, allow_pickle=False) as payload:
                pose_2p5d = payload["pose"]
                self.assertTrue(np.all(pose_2p5d[..., :2] >= 0.0))
                self.assertTrue(np.all(pose_2p5d[..., :2] < 1.0))
                np.testing.assert_allclose(
                    pose_2p5d[..., 2], expected_3d[..., 2]
                )

            delta_output = root / "delta.npz"
            delta_converter = WiLoRCoordinateConverter(
                xy_source="wrist_relative_3d",
                z_source="wrist_relative_3d",
                wrist_source="camera_3d",
            )
            WiLoRToCanonical(
                coordinates=delta_converter,
                postprocessor=WristDeltaPostprocessor(),
            ).run(raw, delta_output)
            absolute = skypiano_camera_markers(first)
            translation = np.asarray(
                camera["translation_xyz_meters"], dtype=np.float32
            )
            camera_wrists = absolute[:, :, 0] + translation
            expected_delta = np.zeros_like(camera_wrists)
            expected_delta[1:] = camera_wrists[1:] - camera_wrists[:-1]
            with np.load(delta_output, allow_pickle=False) as payload:
                np.testing.assert_allclose(
                    payload["pose"][:, :, 0], expected_delta, atol=1e-6
                )

            reused = convert_motion_root(motion_root, output_root, chunk_frames=2)
            self.assertEqual(reused["converted"], 0)
            self.assertEqual(reused["reused"], 2)

            camera["translation_xyz_meters"][0] += 1.0
            (dataset_root / "_camera.json").write_text(
                json.dumps(camera), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "camera mismatch"):
                convert_motion_root(motion_root, output_root, chunk_frames=2)

            sentinel = dataset_root / "sentinel"
            sentinel.touch()
            replaced = convert_motion_root(
                motion_root,
                output_root,
                chunk_frames=2,
                overwrite=True,
            )
            self.assertEqual(replaced["converted"], 2)
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
