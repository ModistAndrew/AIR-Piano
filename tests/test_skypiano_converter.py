"""Tests for SKY-Piano marker-to-canonical conversion."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airpiano.data.hands import CanonicalNpzReader
from airpiano.preprocessing.canonical import CANONICAL_SCHEMA_VERSION
from scripts.convert_skypiano_motion import (
    INPUT_SUFFIX,
    convert_motion_root,
)


HAND_SIDES = ("L", "R")
EXPECTED_JOINT_MARKERS = (
    "Wrist",
    "TC",
    "TM",
    "TD",
    "TF",
    "IM",
    "IP",
    "ID",
    "IF",
    "MM",
    "MP",
    "MD",
    "MF",
    "RM",
    "RP",
    "RD",
    "RF",
    "LM",
    "LP",
    "LD",
    "LF",
)


class SkyPianoConverterTest(unittest.TestCase):
    @staticmethod
    def _write_motion(path: Path, *, scale: float = 1.0) -> None:
        fields = [
            f"{side}{marker}_{axis}"
            for side in HAND_SIDES
            for marker in EXPECTED_JOINT_MARKERS
            for axis in "xyz"
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame in range(2):
                row: dict[str, str | float] = {}
                for hand, side in enumerate(HAND_SIDES):
                    wrist = np.asarray(
                        (100.0, 200.0, 300.0), dtype=np.float32
                    ) * (hand + 1)
                    for joint, marker in enumerate(EXPECTED_JOINT_MARKERS):
                        delta = (
                            np.asarray(
                                (joint * 10.0, joint * 20.0, joint * 30.0),
                                dtype=np.float32,
                            )
                            * (hand + 1)
                            * scale
                        )
                        value = wrist + delta
                        for axis, axis_name in enumerate("xyz"):
                            row[f"{side}{marker}_{axis_name}"] = float(value[axis])
                if frame == 1:
                    row["LIF_z"] = ""
                writer.writerow(row)

    def test_maps_marker_order_axes_units_presence_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion_root = root / "motion"
            output_root = root / "canonical"
            input_path = motion_root / f"take{INPUT_SUFFIX}"
            self._write_motion(input_path)

            summary = convert_motion_root(motion_root, output_root)

            self.assertEqual(
                summary, {"records": 1, "converted": 1, "reused": 0}
            )
            output = output_root / "take.npz"
            with np.load(output, allow_pickle=False) as payload:
                pose = payload["pose"]
                presence = payload["presence"]
                self.assertEqual(pose.dtype, np.float32)
                self.assertEqual(presence.dtype, np.bool_)
                self.assertEqual(payload["fps"].dtype, np.float64)
                self.assertEqual(float(payload["fps"]), 30.0)
                self.assertEqual(
                    str(payload["schema_version"]), CANONICAL_SCHEMA_VERSION
                )

            self.assertEqual(pose.shape, (2, 2, 21, 3))
            np.testing.assert_array_equal(
                presence,
                np.asarray(((True, True), (False, True)), dtype=np.bool_),
            )
            np.testing.assert_array_equal(pose[:, :, 0], 0.0)
            expected_left = np.asarray((-0.01, 0.02, -0.03), dtype=np.float32)
            expected_right = expected_left * np.float32(2.0)
            np.testing.assert_allclose(pose[0, 0, 1], expected_left)
            np.testing.assert_allclose(pose[0, 1, 1], expected_right)
            np.testing.assert_array_equal(pose[1, 0], 0.0)
            sequence = CanonicalNpzReader().read(output)
            self.assertEqual(tuple(sequence.pose.shape), (2, 2, 21, 3))
            self.assertEqual(sequence.fps, 30.0)

    def test_reuses_outputs_unless_overwrite_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion_root = root / "motion"
            output_root = root / "canonical"
            input_path = motion_root / f"take{INPUT_SUFFIX}"
            self._write_motion(input_path)
            convert_motion_root(motion_root, output_root)
            output = output_root / "take.npz"
            original = output.read_bytes()

            self._write_motion(input_path, scale=2.0)
            self.assertEqual(
                convert_motion_root(motion_root, output_root),
                {"records": 1, "converted": 0, "reused": 1},
            )
            self.assertEqual(output.read_bytes(), original)

            self.assertEqual(
                convert_motion_root(motion_root, output_root, overwrite=True),
                {"records": 1, "converted": 1, "reused": 0},
            )
            self.assertNotEqual(output.read_bytes(), original)

    def test_rejects_an_empty_motion_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "no SKY-Piano"):
                convert_motion_root(root / "motion", root / "canonical")


if __name__ == "__main__":
    unittest.main()
