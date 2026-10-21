from __future__ import annotations

import contextlib
import io
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mido
from parangonar.dp.dtw import FlexDynamicTimeWarping

from airpiano.data.midi import MidoNoteReader
from scripts import check_midi_similarity as similarity


def make_chords(seed: int, count: int = 72) -> list[tuple[int, ...]]:
    randomizer = random.Random(seed)
    root = 48 + seed % 5
    chords = []
    steps = (-5, -3, -2, 1, 2, 4, 5, 7)
    qualities = ((0, 4, 7), (0, 3, 7), (0, 4, 7, 11), (0, 3, 7, 10))
    for index in range(count):
        root = min(68, max(40, root + randomizer.choice(steps)))
        quality = qualities[(index + randomizer.randrange(len(qualities))) % len(qualities)]
        chords.append(tuple(root + interval for interval in quality))
    return chords


def make_onsets(count: int) -> list[int]:
    gaps = (180, 240, 300, 210, 360, 270, 150)
    onsets = []
    tick = 0
    for index in range(count):
        onsets.append(tick)
        tick += gaps[index % len(gaps)]
    return onsets


def write_midi(
    path: Path,
    chords: list[tuple[int, ...]],
    onsets: list[int],
    *,
    tempo: int = 500_000,
) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[int, int, mido.Message]] = []
    for onset, pitches in zip(onsets, chords):
        for pitch in pitches:
            events.append(
                (
                    onset,
                    1,
                    mido.Message("note_on", note=pitch, velocity=80, channel=0),
                )
            )
            events.append(
                (
                    onset + 60,
                    0,
                    mido.Message("note_off", note=pitch, velocity=0, channel=0),
                )
            )
    events.sort(key=lambda event: (event[0], event[1], event[2].note))
    previous_tick = 0
    for tick, _, message in events:
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    midi.save(path)


class MidiSimilarityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        cls.chords = make_chords(seed=7)
        cls.onsets = make_onsets(len(cls.chords))
        cls.base_path = cls.root / "base.mid"
        write_midi(cls.base_path, cls.chords, cls.onsets)
        cls.reader = MidoNoteReader()
        cls.base = similarity.read_profile(cls.base_path, "base", cls.reader)
        cls.aligner = FlexDynamicTimeWarping(buffer=1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def compare(self, path: Path, record_id: str = "transformed"):
        profile = similarity.read_profile(path, record_id, self.reader)
        screen = similarity.screen_pair(profile, self.base)
        alignment = similarity.best_alignment(
            profile,
            self.base,
            screen.shifts,
            self.aligner,
        )
        classification = similarity.classify(
            screen,
            alignment,
            likely_threshold=75.0,
            possible_threshold=60.0,
        )
        return screen, alignment, classification

    def test_expected_transformations_are_likely(self) -> None:
        exact_path = self.root / "exact.mid"
        shutil.copyfile(self.base_path, exact_path)

        tempo_path = self.root / "tempo.mid"
        write_midi(tempo_path, self.chords, self.onsets, tempo=750_000)

        transposed_path = self.root / "transposed.mid"
        write_midi(
            transposed_path,
            [tuple(pitch + 5 for pitch in chord) for chord in self.chords],
            self.onsets,
        )

        excerpt_path = self.root / "excerpt.mid"
        excerpt_slice = slice(12, 60)
        excerpt_onsets = [
            tick - self.onsets[12] for tick in self.onsets[excerpt_slice]
        ]
        write_midi(
            excerpt_path,
            self.chords[excerpt_slice],
            excerpt_onsets,
        )

        keep = [index for index in range(len(self.chords)) if index % 10 != 0]
        deletion_path = self.root / "deletion.mid"
        write_midi(
            deletion_path,
            [self.chords[index] for index in keep],
            [self.onsets[index] for index in keep],
        )

        cases = {
            "exact": exact_path,
            "tempo": tempo_path,
            "transposition": transposed_path,
            "excerpt": excerpt_path,
            "note deletion": deletion_path,
        }
        for name, path in cases.items():
            with self.subTest(name=name):
                screen, alignment, classification = self.compare(path, name)
                self.assertEqual(classification, "likely")
                self.assertGreaterEqual(alignment.score, 75.0)
                self.assertGreaterEqual(alignment.query_coverage, 0.50)
                self.assertGreaterEqual(alignment.matched_events, 40)
                if name == "exact":
                    self.assertTrue(screen.exact_file)
                if name == "transposition":
                    self.assertEqual(alignment.transposition, 5)

    def test_unrelated_material_is_below_possible_threshold(self) -> None:
        unrelated_path = self.root / "unrelated.mid"
        unrelated_chords = make_chords(seed=812)
        unrelated_onsets = make_onsets(len(unrelated_chords))
        write_midi(unrelated_path, unrelated_chords, unrelated_onsets)
        _, alignment, classification = self.compare(unrelated_path, "unrelated")
        self.assertEqual(classification, "none")
        self.assertLess(alignment.score, 60.0)

    def test_split_paths_and_missing_file_policy(self) -> None:
        midi_root = self.root / "midi_root"
        (midi_root / "collection").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.base_path, midi_root / "collection/song.mid")
        split_path = self.root / "custom.txt"
        split_path.write_text(
            "collection/song\ncollection/missing\n",
            encoding="utf-8",
        )

        self.assertEqual(
            similarity.resolve_split_path(split_path),
            split_path.resolve(),
        )
        with self.assertRaises(similarity.SplitError):
            similarity.resolve_split_path(self.root / "missing.txt")

        args = similarity.parse_args([str(split_path), str(split_path)])
        self.assertEqual(args.left_split, split_path)
        self.assertEqual(args.midi_root, similarity.DEFAULT_MIDI_ROOT)

        entries = similarity.read_split_entries(split_path, midi_root)
        self.assertEqual(len(entries), 2)
        with self.assertRaises(similarity.SplitError):
            similarity.load_split(
                str(split_path),
                midi_root,
                self.reader,
                skip_missing=False,
            )

        loaded = similarity.load_split(
            str(split_path),
            midi_root,
            self.reader,
            skip_missing=True,
        )
        self.assertEqual(loaded.requested_count, 2)
        self.assertEqual(len(loaded.profiles), 1)
        self.assertEqual(len(loaded.skipped), 1)

        empty_path = self.root / "empty.txt"
        empty_path.write_text("# comment only\n", encoding="utf-8")
        with self.assertRaises(similarity.SplitError):
            similarity.read_split_entries(empty_path, midi_root)

    def test_cli_report_uses_explicit_split_paths(self) -> None:
        midi_root = self.root / "cli_root"
        midi_root.mkdir(exist_ok=True)
        shutil.copyfile(self.base_path, midi_root / "left.mid")
        shutil.copyfile(self.base_path, midi_root / "right.mid")
        left_split = self.root / "left.txt"
        right_split = self.root / "right.txt"
        left_split.write_text("left\n", encoding="utf-8")
        right_split.write_text("right\n", encoding="utf-8")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = similarity.main(
                [
                    str(left_split),
                    str(right_split),
                    "--midi-root",
                    str(midi_root),
                    "--screen-candidates",
                    "1",
                    "--show-unmatched",
                    "0",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("likely pairs:             1", stdout.getvalue())
        self.assertIn("byte-identical", stdout.getvalue())


    def test_direct_script_entry_point(self) -> None:
        midi_root = self.root / "subprocess_root"
        midi_root.mkdir(exist_ok=True)
        shutil.copyfile(self.base_path, midi_root / "one.mid")
        shutil.copyfile(self.base_path, midi_root / "two.mid")
        left_split = self.root / "subprocess_left.txt"
        right_split = self.root / "subprocess_right.txt"
        left_split.write_text("one\n", encoding="utf-8")
        right_split.write_text("two\n", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(similarity.PROJECT_ROOT / "scripts/check_midi_similarity.py"),
                str(left_split),
                str(right_split),
                "--midi-root",
                str(midi_root),
                "--screen-candidates",
                "1",
                "--show-unmatched",
                "0",
            ],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("likely pairs:             1", completed.stdout)

if __name__ == "__main__":
    unittest.main()
