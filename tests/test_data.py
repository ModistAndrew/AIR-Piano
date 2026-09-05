"""Contract tests for the modular, in-memory alignment data pipeline."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import mido
import numpy as np
import torch

from airpiano.data import (
    CANONICAL_HAND_SCHEMA,
    AlignmentDataset,
    CanonicalNpzReader,
    CenteredXYZ10Transform,
    Chord,
    ChordOnsetEvents,
    EagerSongReader,
    FileLoader,
    FilePair,
    FutureChordSlicer,
    HandLoader,
    HandSequence,
    InitializerChordMerger,
    MidiSequence,
    MidoNoteReader,
    Note,
    Song,
    RelativeStemPairer,
    TargetGenerator,
    TriangularSoftener,
    collate_alignment_batch,
)


def _note(
    note_id: int,
    onset: float,
    offset: float,
    pitch: int,
    velocity: int = 80,
) -> Note:
    return Note(note_id, onset, offset, pitch, velocity)


def _song(
    chords: tuple[Chord, ...],
    *,
    frame_count: int,
    fps: float,
    presence: torch.Tensor | None = None,
) -> Song:
    hands = torch.arange(frame_count * 2 * 21 * 3, dtype=torch.float32).reshape(
        frame_count, 2, 21, 3
    )
    if presence is None:
        presence = torch.ones((frame_count, 2), dtype=torch.bool)
    return Song(
        record_id="subject/video",
        notes=tuple(note for chord in chords for note in chord.notes),
        chords=chords,
        hands=hands,
        presence=presence,
        fps=fps,
    )


class PairingTest(unittest.TestCase):
    def test_pairer_returns_file_intersection_and_reports_every_missing_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            midi_root = root / "midi"
            hand_root = root / "hands"
            stem_root = Path("performer one") / "session.v1"
            (midi_root / stem_root).mkdir(parents=True)
            (hand_root / stem_root).mkdir(parents=True)
            (midi_root / stem_root / "complete.take.mid").touch()
            (hand_root / stem_root / "complete.take.npz").touch()
            (midi_root / stem_root / "no_hand.mid").touch()
            (hand_root / stem_root / "no_midi.npz").touch()
            split = root / "split.txt"
            split.write_text(
                "# root-relative record stems\n"
                "\n"
                "performer one/session.v1/complete.take\n"
                "performer one/session.v1/no_hand\n"
                "performer one/session.v1/no_midi\n"
                "performer one/session.v1/neither\n"
                "performer one/session.v1/complete.take\n",
                encoding="utf-8",
            )

            printed = io.StringIO()
            with redirect_stdout(printed):
                pairs = RelativeStemPairer().pair(split, midi_root, hand_root)

            self.assertEqual(
                [pair.record_id for pair in pairs],
                ["performer one/session.v1/complete.take"],
            )
            self.assertEqual(
                pairs[0].midi_path,
                midi_root / "performer one/session.v1/complete.take.mid",
            )
            self.assertEqual(
                pairs[0].hand_path,
                hand_root / "performer one/session.v1/complete.take.npz",
            )
            report = printed.getvalue()
            self.assertIn("skip performer one/session.v1/no_hand", report)
            self.assertIn("missing hand cache", report)
            self.assertIn("skip performer one/session.v1/no_midi", report)
            self.assertIn("missing MIDI", report)
            self.assertIn("skip performer one/session.v1/neither", report)
            neither_line = next(line for line in report.splitlines() if "neither" in line)
            self.assertIn("missing MIDI", neither_line)
            self.assertIn("missing hand cache", neither_line)
            self.assertIn(
                "pairing summary: requested=4 paired=1 missing=3", report
            )

    def test_file_loader_exposes_reader_and_eagerly_composes_modalities(self) -> None:
        note = _note(0, 0.0, 0.2, 60)
        chord = Chord(0, (note,))
        midi = MidiSequence((note,), (chord,))
        hand = HandSequence(
            torch.zeros((2, 2, 21, 3), dtype=torch.float32),
            torch.tensor([[True, False], [False, True]]),
            30.0,
        )
        pair = FilePair("s/v", Path("unused.mid"), Path("unused.npz"))

        class Pairer:
            def pair(self, split_file, midi_root, hand_root):
                return [pair]

        class MidiLoader:
            def load(self, path):
                self.path = path
                return midi

        class HandsLoader:
            def load(self, path):
                self.path = path
                return hand

        reader = EagerSongReader()
        loader = FileLoader(Pairer(), reader)
        songs = loader.load(
            split_file="split.txt",
            midi_root="midi",
            hand_root="hands",
            midi_loader=MidiLoader(),
            hand_loader=HandsLoader(),
        )

        self.assertIs(loader.reader, reader)
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0].record_id, "s/v")
        self.assertIs(songs[0].hands, hand.pose)
        self.assertIs(songs[0].presence, hand.presence)


class MidiTest(unittest.TestCase):
    def test_reader_uses_merged_track_tempo_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tempo.mid"
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            tempo = mido.MidiTrack()
            tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            tempo.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=480))
            notes = mido.MidiTrack()
            notes.append(mido.Message("note_on", note=60, velocity=70, time=0))
            notes.append(mido.Message("note_off", note=60, velocity=0, time=480))
            notes.append(mido.Message("note_on", note=62, velocity=71, time=0))
            notes.append(mido.Message("note_off", note=62, velocity=0, time=480))
            midi.tracks.extend((tempo, notes))
            midi.save(path)

            parsed = MidoNoteReader().read(path)

        self.assertEqual([note.id for note in parsed], [0, 1])
        self.assertEqual([note.pitch for note in parsed], [60, 62])
        self.assertAlmostEqual(parsed[0].onset, 0.0)
        self.assertAlmostEqual(parsed[0].offset, 0.5)
        self.assertAlmostEqual(parsed[1].onset, 0.5)
        self.assertAlmostEqual(parsed[1].offset, 1.5)

    def test_reader_pairs_repeated_channel_pitch_notes_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifo.mid"
            midi = mido.MidiFile(type=0, ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=10, time=0))
            track.append(mido.Message("note_on", note=60, velocity=20, time=240))
            track.append(mido.Message("note_off", note=60, velocity=0, time=240))
            track.append(mido.Message("note_on", note=60, velocity=0, time=240))
            midi.tracks.append(track)
            midi.save(path)

            parsed = MidoNoteReader().read(path)

        self.assertEqual(len(parsed), 2)
        self.assertEqual([note.velocity for note in parsed], [10, 20])
        self.assertAlmostEqual(parsed[0].onset, 0.0)
        self.assertAlmostEqual(parsed[0].offset, 0.5)
        self.assertAlmostEqual(parsed[1].onset, 0.25)
        self.assertAlmostEqual(parsed[1].offset, 0.75)

    def test_chord_merger_is_inclusive_and_initializer_anchored(self) -> None:
        notes = tuple(
            _note(index, onset, onset + 0.2, 60 + index)
            for index, onset in enumerate((0.0, 0.05, 0.09, 0.139, 0.141))
        )

        chords = InitializerChordMerger(tolerance_seconds=0.05).merge(notes)

        self.assertEqual([chord.id for chord in chords], [0, 1, 2])
        self.assertEqual(
            [[note.id for note in chord.notes] for chord in chords],
            [[0, 1], [2, 3], [4]],
        )
        self.assertIs(chords[0].notes[0], notes[0])


class HandCacheTest(unittest.TestCase):
    def test_canonical_reader_and_centered_xyz10_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.npz"
            pose = np.empty((2, 2, 21, 3), dtype=np.float32)
            pose[0, 0] = (0.75, 0.25, 0.2)
            pose[0, 1] = (9.0, 9.0, 9.0)
            pose[1, 0] = (8.0, 8.0, 8.0)
            pose[1, 1] = (0.0, 1.0, -0.3)
            presence = np.asarray([[True, False], [False, True]], dtype=np.bool_)
            np.savez(
                path,
                pose=pose,
                presence=presence,
                fps=np.asarray(30.0, dtype=np.float64),
                schema_version=np.asarray(CANONICAL_HAND_SCHEMA),
            )

            raw = CanonicalNpzReader().read(path)
            transformed = HandLoader(
                CanonicalNpzReader(), CenteredXYZ10Transform()
            ).load(path)

        self.assertEqual(raw.pose.dtype, torch.float32)
        self.assertEqual(raw.presence.dtype, torch.bool)
        self.assertEqual(raw.fps, 30.0)
        torch.testing.assert_close(
            transformed.pose[0, 0, 0], torch.tensor([0.25, -0.25, 2.0])
        )
        torch.testing.assert_close(
            transformed.pose[1, 1, 0], torch.tensor([-0.5, 0.5, -3.0])
        )
        self.assertEqual(torch.count_nonzero(transformed.pose[0, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(transformed.pose[1, 0]).item(), 0)
        torch.testing.assert_close(transformed.presence, raw.presence)

class SlicingAndTargetTest(unittest.TestCase):
    def test_starts_include_exact_right_aligned_tail(self) -> None:
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=3, chord_count=3
        )
        self.assertEqual(slicer.starts(3), (0,))
        self.assertEqual(slicer.starts(11), (0, 3, 6, 7))

    def test_slice_selects_overlaps_then_future_and_collapses_duplicate_pitch(self) -> None:
        chords = (
            Chord(0, (_note(0, 0.0, 0.2, 50),)),
            Chord(
                1,
                (
                    _note(1, 2.0, 2.4, 60),
                    _note(4, 2.01, 2.3, 0),
                    _note(5, 2.02, 2.3, 127),
                    _note(2, 2.03, 2.5, 60),
                ),
            ),
            Chord(2, (_note(3, 5.0, 5.2, 70),)),
        )
        song = _song(chords, frame_count=30, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=2, chord_count=3
        )

        window = slicer.slice(song, frame_start=19)

        self.assertEqual(window["chord_ids"].tolist(), [1, 2, -1])
        self.assertEqual(window["chord_mask"].tolist(), [True, True, False])
        self.assertEqual(window["chords"][0, 60].item(), 1.0)
        self.assertEqual(window["chords"][0, 0].item(), 1.0)
        self.assertEqual(window["chords"][0, 127].item(), 1.0)
        self.assertEqual(window["chords"][0].sum().item(), 3.0)
        self.assertEqual(len(window["selected_chords"][0].notes), 4)
        torch.testing.assert_close(window["hands"], song.hands[19:23])

        no_overlap = slicer.candidate_positions(chords, frame_start=6)
        self.assertEqual(no_overlap, (1, 2))

    def test_more_overlapping_chords_than_fixed_width_raises(self) -> None:
        chords = tuple(
            Chord(index, (_note(index, 0.01 * index, 1.0, 60 + index),))
            for index in range(3)
        )
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=2, chord_count=2
        )
        with self.assertRaisesRegex(ValueError, "3 overlapping chords"):
            slicer.candidate_positions(chords, frame_start=0)

    def test_dataset_precomputes_chords_masks_padded_targets_and_collates(self) -> None:
        class CountingSlicer(FutureChordSlicer):
            def __init__(self):
                super().__init__(
                    fps=10.0,
                    segment_frames=4,
                    stride_frames=2,
                    chord_count=3,
                )
                self.selection_calls = 0

            def candidate_positions(self, chords, frame_start):
                self.selection_calls += 1
                return super().candidate_positions(chords, frame_start)

        chords = (
            Chord(0, (_note(0, 0.1, 0.2, 60),)),
            Chord(1, (_note(1, 1.0, 1.2, 64),)),
        )
        presence = torch.tensor([[False, False], [True, False]])
        song = _song(chords, frame_count=2, fps=10.0, presence=presence)
        slicer = CountingSlicer()
        generator = TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2))

        dataset = AlignmentDataset([song], slicer, generator)
        self.assertEqual(slicer.selection_calls, 1)
        sample = dataset[0]
        self.assertEqual(slicer.selection_calls, 1)

        self.assertEqual(tuple(sample["hands"].shape), (4, 2, 21, 3))
        self.assertEqual(tuple(sample["chords"].shape), (3, 128))
        self.assertEqual(tuple(sample["target"].shape), (4, 3))
        self.assertEqual(sample["frame_mask"].tolist(), [True, True, False, False])
        self.assertEqual(
            sample["hand_presence"].tolist(),
            [[False, False], [True, False], [False, False], [False, False]],
        )
        torch.testing.assert_close(
            sample["target"][:, 0], torch.tensor([0.5, 1.0, 0.0, 0.0])
        )
        self.assertEqual(torch.count_nonzero(sample["target"][2:]).item(), 0)

        batch = collate_alignment_batch([sample, sample])
        self.assertEqual(tuple(batch["hands"].shape), (2, 4, 2, 21, 3))
        self.assertEqual(tuple(batch["target"].shape), (2, 4, 3))
        self.assertEqual(batch["record_id"], ["subject/video", "subject/video"])
        self.assertEqual(batch["frame_start"].dtype, torch.long)
        self.assertEqual(batch["fps"].dtype, torch.float64)

    def test_dataset_omits_windows_with_only_padded_chords(self) -> None:
        chords = (Chord(0, (_note(0, 0.1, 0.2, 60),)),)
        song = _song(chords, frame_count=11, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=4,
            stride_frames=3,
            chord_count=3,
        )
        generator = TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2))

        dataset = AlignmentDataset([song], slicer, generator)

        self.assertEqual(dataset.samples, ((0, 0, (0,)),))

    def test_target_uses_global_left_edge_frame_times(self) -> None:
        chord = Chord(0, (_note(0, 1.1, 1.2, 60),))
        target = TargetGenerator(
            ChordOnsetEvents(), TriangularSoftener(radius_seconds=0.2)
        ).generate(
            (chord,),
            frame_start=10,
            segment_frames=3,
            fps=10.0,
            chord_count=2,
        )
        torch.testing.assert_close(
            target[:, 0], torch.tensor([0.5, 1.0, 0.5]), atol=1e-6, rtol=0
        )
        self.assertEqual(torch.count_nonzero(target[:, 1]).item(), 0)


if __name__ == "__main__":
    unittest.main()
