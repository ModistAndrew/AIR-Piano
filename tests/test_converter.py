"""Tests for the one-time legacy split conversion utility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.convert_legacy_splits import (
    DEFAULT_OUTPUT_ROOT,
    convert_split,
    convert_splits,
)


class LegacySplitConverterTest(unittest.TestCase):
    def test_conversion_preserves_order_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.txt"
            destination = root / "generalized.txt"
            source.write_text(
                "# subject video\n"
                "person-a take.01\n"
                "\n"
                "person-b take-02\n"
                "person-a take.01\n",
                encoding="utf-8",
            )

            count = convert_split(source, destination)

            self.assertEqual(count, 2)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "person-a/take.01\nperson-b/take-02\n",
            )

    def test_all_splits_have_expected_counts_without_repository_assets(self) -> None:
        expected = {"train": 882, "valid": 101, "test": 983}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "legacy"
            output_root = root / "generalized"
            source_root.mkdir()
            for name, count in expected.items():
                rows = [f"{name}-subject-{index} take.{index}" for index in range(count)]
                (source_root / f"{name}.txt").write_text(
                    "# generated fixture\n\n"
                    + "\n".join(rows)
                    + f"\n{rows[0]}\n",
                    encoding="utf-8",
                )

            counts = convert_splits(source_root, output_root)

            self.assertEqual(counts, expected)
            for name, count in expected.items():
                records = (output_root / f"{name}.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(len(records), count)
                self.assertEqual(len(set(records)), count)
                self.assertTrue(all("/" in record for record in records))

    def test_default_output_is_the_ignored_asset_tree(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            DEFAULT_OUTPUT_ROOT,
            project_root / "assets" / "splits" / "pianomotion10m",
        )


if __name__ == "__main__":
    unittest.main()
