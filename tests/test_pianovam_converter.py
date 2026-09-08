"""Tests for generating AirPiano splits from PianoVAM metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_pianovam_splits import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    convert_splits,
)


class PianoVamSplitConverterTest(unittest.TestCase):
    @staticmethod
    def _write_dataset(root: Path, records: list[dict]) -> None:
        root.mkdir()
        (root / "Video").mkdir()
        (root / "MIDI").mkdir()
        (root / "metadata.json").write_text(
            json.dumps({str(index): record for index, record in enumerate(records)}),
            encoding="utf-8",
        )
        for record in records:
            if str(record["split"]).startswith("special("):
                continue
            record_id = record["record_time"]
            (root / "Video" / f"{record_id}.mp4").touch()
            (root / "MIDI" / f"{record_id}.mid").touch()

    def test_official_splits_are_sorted_and_ext_train_is_merged(self) -> None:
        records = [
            {"record_time": "train-b", "split": "train"},
            {"record_time": "test-a", "split": "test"},
            {"record_time": "train-a", "split": "ext-train"},
            {"record_time": "valid-a", "split": "valid"},
            {"record_time": "blurry", "split": "special(blurry)"},
            {"record_time": "four-hands", "split": "special(4hands)"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "splits"
            self._write_dataset(data_root, records)

            counts = convert_splits(data_root, output_root)

            self.assertEqual(counts, {"train": 2, "valid": 1, "test": 1})
            self.assertEqual(
                (output_root / "train.txt").read_text(encoding="utf-8"),
                "train-a\ntrain-b\n",
            )
            self.assertEqual(
                (output_root / "valid.txt").read_text(encoding="utf-8"),
                "valid-a\n",
            )
            self.assertEqual(
                (output_root / "test.txt").read_text(encoding="utf-8"),
                "test-a\n",
            )

    def test_duplicate_record_ids_are_rejected(self) -> None:
        records = [
            {"record_time": "same", "split": "train"},
            {"record_time": "same", "split": "valid"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            self._write_dataset(data_root, records)

            with self.assertRaisesRegex(ValueError, "duplicate record_time 'same'"):
                convert_splits(data_root, root / "splits")

    def test_expected_corpus_counts_without_repository_assets(self) -> None:
        records = [
            *(
                {"record_time": f"train-{index:03d}", "split": "train"}
                for index in range(53)
            ),
            *(
                {"record_time": f"ext-{index:03d}", "split": "ext-train"}
                for index in range(28)
            ),
            *(
                {"record_time": f"valid-{index:03d}", "split": "valid"}
                for index in range(9)
            ),
            *(
                {"record_time": f"test-{index:03d}", "split": "test"}
                for index in range(9)
            ),
            *(
                {
                    "record_time": f"blurry-{index:03d}",
                    "split": "special(blurry)",
                }
                for index in range(7)
            ),
            {"record_time": "four-hands", "split": "special(4hands)"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            self._write_dataset(data_root, records)

            counts = convert_splits(data_root, root / "splits")

            self.assertEqual(counts, {"train": 81, "valid": 9, "test": 9})

    def test_unexpected_split_is_rejected(self) -> None:
        records = [{"record_time": "record", "split": "development"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            self._write_dataset(data_root, records)

            with self.assertRaisesRegex(ValueError, "unexpected split 'development'"):
                convert_splits(data_root, root / "splits")

    def test_missing_selected_modality_is_rejected_before_writing(self) -> None:
        records = [{"record_time": "record", "split": "train"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "splits"
            self._write_dataset(data_root, records)
            (data_root / "MIDI" / "record.mid").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "record.mid"):
                convert_splits(data_root, output_root)
            self.assertFalse(output_root.exists())

    def test_defaults_target_the_local_pianovam_install_and_ignored_assets(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(DEFAULT_DATA_ROOT, Path("/storage/jinxuan/data/PianoVAM_v1"))
        self.assertEqual(
            DEFAULT_OUTPUT_ROOT,
            project_root / "assets" / "splits" / "pianovam",
        )


if __name__ == "__main__":
    unittest.main()
