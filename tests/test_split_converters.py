"""Tests for dataset-prefixed split conversion utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_legacy_splits import convert_splits as convert_pianomotion
from scripts.convert_pianovam_splits import convert_splits as convert_pianovam


class SplitConverterTest(unittest.TestCase):
    def test_pianomotion_records_include_dataset_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            output = root / "splits"
            source.mkdir()
            for name in ("train", "valid", "test"):
                (source / f"{name}.txt").write_text(
                    "subject video\n", encoding="utf-8"
                )

            counts = convert_pianomotion(source, output)

            self.assertEqual(counts, {"train": 1, "valid": 1, "test": 1})
            for name in ("train", "valid"):
                self.assertEqual(
                    (output / f"{name}.txt").read_text(encoding="utf-8"),
                    "PianoMotion10M/subject/video\n",
                )
            self.assertEqual(
                (output / "test_full.txt").read_text(encoding="utf-8"),
                "PianoMotion10M/subject/video\n",
            )
            self.assertFalse((output / "test.txt").exists())

    def test_pianovam_records_include_dataset_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "AirPianoFull"
            output = Path(directory) / "splits"
            record_id = "2024-01-02_03-04-05"
            metadata = root / "Metadata" / "PianoVAM_v1"
            video = root / "Video" / "PianoVAM_v1"
            midi = root / "MIDI" / "PianoVAM_v1"
            metadata.mkdir(parents=True)
            video.mkdir(parents=True)
            midi.mkdir(parents=True)
            (metadata / "metadata.json").write_text(
                json.dumps(
                    {
                        "record": {
                            "record_time": record_id,
                            "split": "train",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (video / f"{record_id}.mp4").touch()
            (midi / f"{record_id}.mid").touch()

            counts = convert_pianovam(root, output)

            self.assertEqual(counts, {"train": 1, "valid": 0, "test": 0})
            self.assertEqual(
                (output / "train.txt").read_text(encoding="utf-8"),
                f"PianoVAM_v1/{record_id}\n",
            )


if __name__ == "__main__":
    unittest.main()
