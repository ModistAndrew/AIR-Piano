"""Tests for the combined PianoMotion/PianoVAM symlink view."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.create_full_data_view import create_data_view


class FullDataViewTest(unittest.TestCase):
    @staticmethod
    def _write_splits(root: Path, values: dict[str, list[str]]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name in ("train", "valid", "test"):
            (root / f"{name}.txt").write_text(
                "".join(f"{record}\n" for record in values[name]),
                encoding="utf-8",
            )

    def _fixture(self, root: Path) -> dict[str, Path]:
        full_splits = root / "full"
        pianomotion_splits = root / "pianomotion"
        pianovam_splits = root / "pianovam"
        self._write_splits(
            full_splits,
            {"train": ["artist/song", "take-a"], "valid": [], "test": []},
        )
        self._write_splits(
            pianomotion_splits,
            {"train": ["artist/song"], "valid": [], "test": []},
        )
        self._write_splits(
            pianovam_splits,
            {"train": ["take-a"], "valid": [], "test": []},
        )
        roots = {
            "full_split_root": full_splits,
            "pianomotion_split_root": pianomotion_splits,
            "pianovam_split_root": pianovam_splits,
            "view_root": root / "view",
            "pianomotion_midi_root": root / "pm-midi",
            "pianomotion_hand_root": root / "pm-hand",
            "pianovam_midi_root": root / "pv-midi",
            "pianovam_hand_root": root / "pv-hand",
        }
        for path in roots.values():
            if path.name.endswith(("midi", "hand")):
                path.mkdir(parents=True)
        (roots["pianomotion_midi_root"] / "artist").mkdir()
        (roots["pianomotion_midi_root"] / "artist/song.mid").touch()
        (roots["pianovam_midi_root"] / "take-a.mid").touch()
        (roots["pianomotion_hand_root"] / "artist").mkdir()
        (roots["pianomotion_hand_root"] / "artist/song.npz").touch()
        return roots

    def test_creates_nested_absolute_links_and_allows_missing_hands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = self._fixture(Path(directory))

            summary = create_data_view(**roots)

            self.assertEqual(
                summary,
                {
                    "records": 2,
                    "pianomotion": 1,
                    "pianovam": 1,
                    "midi_links": 2,
                    "hand_links": 2,
                    "missing_hand_sources": 1,
                },
            )
            nested = roots["view_root"] / "midi/artist/song.mid"
            missing = roots["view_root"] / "airpiano_wilor_2p5d_30fps/take-a.npz"
            self.assertTrue(nested.is_symlink())
            self.assertTrue(nested.readlink().is_absolute())
            self.assertEqual(
                nested.resolve(),
                (roots["pianomotion_midi_root"] / "artist/song.mid").resolve(),
            )
            self.assertTrue(missing.is_symlink())
            self.assertFalse(missing.exists())

            self.assertEqual(create_data_view(**roots), summary)

    def test_duplicate_and_ambiguous_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._fixture(root)
            self._write_splits(
                roots["full_split_root"],
                {"train": ["take-a"], "valid": ["take-a"], "test": []},
            )
            with self.assertRaisesRegex(ValueError, "duplicate record 'take-a'"):
                create_data_view(**roots)

            self._write_splits(
                roots["full_split_root"],
                {"train": ["take-a"], "valid": [], "test": []},
            )
            self._write_splits(
                roots["pianomotion_split_root"],
                {"train": ["take-a"], "valid": [], "test": []},
            )
            with self.assertRaisesRegex(ValueError, "both source datasets"):
                create_data_view(**roots)

    def test_missing_midi_and_destination_conflicts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._fixture(root)
            (roots["pianovam_midi_root"] / "take-a.mid").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "take-a.mid"):
                create_data_view(**roots)
            self.assertFalse(roots["view_root"].exists())

            (roots["pianovam_midi_root"] / "take-a.mid").touch()
            conflict = roots["view_root"] / "midi/take-a.mid"
            conflict.parent.mkdir(parents=True)
            conflict.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                create_data_view(**roots)
            self.assertEqual(conflict.read_text(encoding="utf-8"), "keep")

            conflict.unlink()
            wrong_target = root / "wrong.mid"
            wrong_target.touch()
            conflict.symlink_to(wrong_target)
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                create_data_view(**roots)
            self.assertEqual(conflict.resolve(), wrong_target.resolve())


if __name__ == "__main__":
    unittest.main()
