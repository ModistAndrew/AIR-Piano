"""Tests for repository-wide filesystem helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from airpiano.utils import PROJECT_ROOT, resolve_path


class PathResolutionTest(unittest.TestCase):
    def test_relative_paths_are_airpiano_root_relative(self) -> None:
        self.assertEqual(
            resolve_path("assets/example.npz"),
            (PROJECT_ROOT / "assets" / "example.npz").resolve(),
        )

    def test_absolute_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.npz"
            self.assertEqual(resolve_path(path), path.resolve())


if __name__ == "__main__":
    unittest.main()
