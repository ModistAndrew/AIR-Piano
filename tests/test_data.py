"""Contract tests for the modular, in-memory alignment data pipeline."""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
import torch
from omegaconf import OmegaConf

from airpiano.data import (
    CANONICAL_HAND_SCHEMA,
    AlignmentDataset,
    CanonicalNpzReader,
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
    OffsetScaleTransform,
    Song,
    RelativeStemPairer,
    TargetGenerator,
    TriangularSoftener,
    build_dataset,
    collate_alignment_batch,
)
from airpiano.data.midi import quantize_duration_seconds


def _note(
    note_id: int,
    onset: float,
    offset: float,
    pitch: int,
    velocity: int = 80,
) -> Note:
    return Note(note_id, onset, offset, pitch, velocity)


def _chord(
    chord_id: int,
    notes: tuple[Note, ...],
    duration: float | None = None,
    quantized_dur: torch.Tensor | None = None,
) -> Chord:
    if duration is None:
        duration = max(note.offset for note in notes) - notes[0].onset
    if quantized_dur is None:
        quantized_dur = quantize_duration_seconds(
            torch.tensor(duration, dtype=torch.float64)
        )
    return Chord(chord_id, notes, duration, quantized_dur)


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
        chord = _chord(0, (note,))
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
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in chords]),
            torch.tensor([0.09, 0.051, 0.2]),
        )
        expected_quantized = quantize_duration_seconds(
            torch.tensor([0.09, 0.051, 0.2], dtype=torch.float64)
        )
        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in chords]),
            expected_quantized,
        )
        self.assertEqual(chords[0].quantized_dur.dtype, torch.float32)
        self.assertEqual(tuple(chords[0].quantized_dur.shape), (200,))

    def test_chord_merger_uses_next_onset_then_latest_final_release(self) -> None:
        notes = (
            _note(0, 0.0, 4.0, 60),
            _note(1, 1.0, 1.2, 64),
            _note(2, 1.02, 1.7, 67),
        )

        chords = InitializerChordMerger(tolerance_seconds=0.05).merge(notes)

        self.assertEqual(len(chords), 2)
        self.assertAlmostEqual(chords[0].duration, 1.0)
        self.assertAlmostEqual(chords[1].duration, 0.7)
        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in chords]),
            quantize_duration_seconds(torch.tensor([1.0, 0.7])),
        )

    def test_chord_merger_scales_all_duration_encodings_once_per_song(
        self,
    ) -> None:
        notes = (
            _note(0, 0.0, 0.4, 60),
            _note(1, 1.0, 1.2, 64),
            _note(2, 3.0, 3.5, 67),
        )
        merger = InitializerChordMerger(
            tolerance_seconds=0.05,
            duration_scale_range=(0.5, 2.0),
        )

        with patch(
            "airpiano.data.midi.random.uniform", return_value=1.5
        ) as uniform:
            chords = merger.merge(notes)

        uniform.assert_called_once_with(0.5, 2.0)
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in chords]),
            torch.tensor([1.0, 2.0, 0.5]),
        )
        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in chords]),
            quantize_duration_seconds(torch.tensor([1.5, 3.0, 0.75])),
        )

    def test_chord_merger_normalizes_durations_by_song_mean(self) -> None:
        notes = (
            _note(0, 0.0, 0.4, 60),
            _note(1, 1.0, 1.2, 64),
            _note(2, 3.0, 3.5, 67),
        )
        merger = InitializerChordMerger(duration_normalization="song_mean")

        with patch(
            "airpiano.data.midi.quantize_duration_seconds",
            wraps=quantize_duration_seconds,
        ) as quantize:
            chords = merger.merge(notes)

        normalized_durations = torch.stack(
            [call.args[0] for call in quantize.call_args_list]
        )
        self.assertAlmostEqual(normalized_durations.mean().item(), 1.0)
        torch.testing.assert_close(
            normalized_durations,
            torch.tensor(
                [6.0 / 7.0, 12.0 / 7.0, 3.0 / 7.0], dtype=torch.float64
            ),
        )
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in chords]),
            torch.tensor([1.0, 2.0, 0.5]),
        )

    def test_song_mean_duration_encoding_is_invariant_to_uniform_tempo(self) -> None:
        notes = (
            _note(0, 0.0, 0.4, 60),
            _note(1, 1.0, 1.2, 64),
            _note(2, 3.0, 3.5, 67),
        )
        tempo_scale = 2.5
        scaled_notes = tuple(
            _note(
                note.id,
                note.onset * tempo_scale,
                note.offset * tempo_scale,
                note.pitch,
                note.velocity,
            )
            for note in notes
        )
        merger = InitializerChordMerger(duration_normalization="song_mean")

        chords = merger.merge(notes)
        scaled_chords = merger.merge(scaled_notes)

        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in scaled_chords]),
            torch.stack([chord.quantized_dur for chord in chords]),
        )
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in scaled_chords]),
            tempo_scale * torch.tensor([chord.duration for chord in chords]),
        )

    def test_song_mean_duration_normalization_handles_empty_and_single_chord(
        self,
    ) -> None:
        merger = InitializerChordMerger(duration_normalization="song_mean")

        self.assertEqual(merger.merge(()), ())
        chords = merger.merge((_note(0, 2.0, 2.4, 60),))

        self.assertEqual(len(chords), 1)
        self.assertAlmostEqual(chords[0].duration, 0.4)
        torch.testing.assert_close(
            chords[0].quantized_dur,
            quantize_duration_seconds(torch.tensor(1.0)),
        )

    def test_chord_merger_rejects_scale_with_song_mean_normalization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "duration_scale_range cannot be combined with song_mean",
        ):
            InitializerChordMerger(
                duration_scale_range=(0.5, 2.0),
                duration_normalization="song_mean",
            )

    def test_chord_merger_rejects_unknown_duration_normalization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "duration_normalization must be 'none' or 'song_mean'",
        ):
            InitializerChordMerger(duration_normalization="median")


class HandCacheTest(unittest.TestCase):
    def test_canonical_reader_and_configurable_transform(self) -> None:
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
                CanonicalNpzReader(),
                OffsetScaleTransform(
                    offset=[0.5, 0.5, 0.0], scale=[1.0, 1.0, 10.0]
                ),
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

    def test_wrist_relative_xyz_transform_scales_every_axis(self) -> None:
        pose = torch.zeros(1, 2, 21, 3)
        pose[0, 0, 1] = torch.tensor([0.01, -0.02, 0.03])
        pose[0, 1] = 9.0
        presence = torch.tensor([[True, False]])

        transformed = OffsetScaleTransform(
            offset=[0.0, 0.0, 0.0], scale=[10.0, 10.0, 10.0]
        )(pose, presence)

        torch.testing.assert_close(
            transformed[0, 0, 1], torch.tensor([0.1, -0.2, 0.3])
        )
        self.assertEqual(torch.count_nonzero(transformed[0, 1]).item(), 0)

class SlicingAndTargetTest(unittest.TestCase):
    def test_duration_quantization_matches_reference_formula_and_clips(self) -> None:
        durations = torch.tensor([0.0, 0.25, 1.0, 16.0, 32.0])

        encoded = quantize_duration_seconds(durations)

        expected = [
            min(
                199,
                math.floor(200 * math.log(4 * value + 1) / math.log(65)),
            )
            for value in durations.tolist()
        ]
        self.assertEqual(tuple(encoded.shape), (5, 200))
        self.assertEqual(encoded.argmax(dim=-1).tolist(), expected)
        self.assertTrue(bool((encoded.sum(dim=-1) == 1.0).all()))

    def test_starts_include_exact_right_aligned_tail(self) -> None:
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=3, chord_count=3
        )
        self.assertEqual(slicer.starts(3), (0,))
        self.assertEqual(slicer.starts(11), (0, 3, 6, 7))

    def test_slice_selects_overlaps_then_future_and_collapses_duplicate_pitch(self) -> None:
        chords = (
            _chord(0, (_note(0, 0.0, 0.2, 50),)),
            _chord(
                1,
                (
                    _note(1, 2.0, 2.4, 60),
                    _note(4, 2.01, 2.3, 0),
                    _note(5, 2.02, 2.3, 127),
                    _note(2, 2.03, 2.5, 60),
                ),
            ),
            _chord(2, (_note(3, 5.0, 5.2, 70),)),
        )
        song = _song(chords, frame_count=30, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=2, chord_count=3
        )

        positions = slicer.candidate_positions(chords, frame_start=19)
        window = slicer.slice(song, 19, 23, positions)

        self.assertEqual(window["chord_ids"].tolist(), [1, 2])
        self.assertEqual(tuple(window["chords"].shape), (2, 328))
        self.assertEqual(window["chords"][0, 60].item(), 1.0)
        self.assertEqual(window["chords"][0, 0].item(), 1.0)
        self.assertEqual(window["chords"][0, 127].item(), 1.0)
        self.assertEqual(window["chords"][0].sum().item(), 4.0)
        self.assertEqual(len(window["selected_chords"][0].notes), 4)
        torch.testing.assert_close(window["hands"], song.hands[19:23])

        no_overlap = slicer.candidate_positions(chords, frame_start=6)
        self.assertEqual(no_overlap, (1, 2))

    def test_duration_tokens_use_quantized_durations_stored_on_chords(self) -> None:
        middle_duration = quantize_duration_seconds(torch.tensor(1.2))
        final_duration = quantize_duration_seconds(torch.tensor(0.6))
        chords = (
            _chord(0, (_note(0, 0.1, 5.0, 60),), duration=0.5),
            _chord(
                1,
                (_note(1, 0.6, 0.9, 64),),
                duration=99.0,
                quantized_dur=middle_duration,
            ),
            _chord(
                2,
                (
                    _note(2, 1.8, 2.0, 67),
                    _note(3, 1.82, 2.4, 71),
                ),
                duration=99.0,
                quantized_dur=final_duration,
            ),
        )
        song = _song(chords, frame_count=30, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=10,
            stride_frames=5,
            chord_count=3,
        )

        middle = slicer.slice(song, 0, 1, (1,))["chords"]
        final = slicer.slice(song, 0, 1, (2,))["chords"]

        self.assertEqual(tuple(middle.shape), (1, 328))
        self.assertEqual(middle[0, :128].argmax().item(), 64)
        self.assertEqual(middle[0, 128:].sum().item(), 1.0)
        torch.testing.assert_close(middle[0, 128:], middle_duration)
        torch.testing.assert_close(final[0, 128:], final_duration)

        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
        )
        batch = collate_alignment_batch([dataset[0], dataset[-1]])
        self.assertEqual(tuple(batch["chords"].shape), (2, 3, 328))

    def test_more_overlapping_chords_than_fixed_width_raises(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, 0.01 * index, 1.0, 60 + index),))
            for index in range(3)
        )
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=2, chord_count=2
        )
        with self.assertRaisesRegex(ValueError, "3 overlapping chords"):
            slicer.candidate_positions(chords, frame_start=0)

    def test_chord_mode_uses_token_windows_and_onset_frame_bounds(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=3,
            stride_frames=2,
            chord_count=1,
        )

        self.assertEqual(
            slicer.windows(song),
            (
                (0, 11, (0, 1, 2)),
                (10, 21, (2, 3, 4)),
                (14, 40, (3, 4, 5)),
            ),
        )
        window = slicer.slice(song, 10, 21, (2, 3, 4))
        self.assertEqual(tuple(window["hands"].shape), (11, 2, 21, 3))
        self.assertEqual(tuple(window["chords"].shape), (3, 328))
        self.assertEqual(window["chord_ids"].tolist(), [2, 3, 4])
        self.assertNotIn("frame_mask", window)
        self.assertNotIn("chord_mask", window)
        torch.testing.assert_close(window["hands"], song.hands[10:21])

    def test_last_chord_dataset_crops_pre_onset_frames(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.21, 0.55, 1.04))
        )
        song = _song(chords, frame_count=20, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=1,
            crop_before_first_chord=True,
        )

        self.assertEqual(
            slicer.windows(song),
            (
                (3, 7, (0, 1)),
                (6, 20, (1, 2)),
            ),
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(
                ChordOnsetEvents(),
                TriangularSoftener(0.2),
                target_mode="last_chord",
            ),
        )
        first, second = dataset[0], dataset[1]
        self.assertEqual(first["frame_start"], 3)
        self.assertEqual(second["frame_start"], 6)
        for sample in (first, second):
            self.assertTrue(bool((sample["target"].sum(dim=-1) == 1).all()))

    def test_chord_mode_adds_neighbor_onsets_only_to_hand_bounds(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=3,
            stride_frames=2,
            hand_context_chords=1,
        )

        self.assertEqual(
            slicer.windows(song),
            (
                (0, 15, (0, 1, 2)),
                (5, 26, (2, 3, 4)),
                (10, 40, (3, 4, 5)),
            ),
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
        )
        middle = dataset[1]
        self.assertEqual(tuple(middle["hands"].shape), (21, 2, 21, 3))
        self.assertEqual(middle["chord_ids"].tolist(), [2, 3, 4])
        self.assertEqual(tuple(middle["target"].shape), (21, 3))
        torch.testing.assert_close(middle["hands"], song.hands[5:26])

        short_song = _song(chords, frame_count=24, fps=10.0)
        self.assertEqual(
            slicer.windows(short_song)[1],
            (5, 24, (2, 3, 4)),
        )

    def test_chord_mode_end_padding_preserves_starts_and_clamps_stops(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=3,
            stride_frames=2,
            end_padding_seconds=1.0,
        )

        self.assertEqual(
            slicer.windows(song),
            (
                (0, 21, (0, 1, 2)),
                (10, 31, (2, 3, 4)),
                (14, 40, (3, 4, 5)),
            ),
        )
        short_song = _song(chords, frame_count=11, fps=10.0)
        self.assertEqual(slicer.windows(short_song)[0], (0, 11, (0, 1, 2)))

    def test_chord_mode_keeps_real_chords_and_one_frame_windows(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0))
        )
        song = _song(chords, frame_count=20, fps=10.0)
        singleton = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=1,
            stride_frames=1,
            chord_count=99,
        )
        self.assertEqual(singleton.windows(song)[1], (5, 6, (1,)))

        window = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=4,
            stride_frames=2,
            chord_count=1,
        ).slice(song, 0, 20, (0, 1, 2))
        self.assertEqual(tuple(window["chords"].shape), (3, 328))
        self.assertEqual(window["chord_ids"].tolist(), [0, 1, 2])

    def test_dataset_returns_real_tensors_and_collate_adds_masks(self) -> None:
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
            _chord(0, (_note(0, 0.1, 0.2, 60),)),
            _chord(1, (_note(1, 1.0, 1.2, 64),)),
        )
        presence = torch.tensor([[False, False], [True, False]])
        song = _song(chords, frame_count=2, fps=10.0, presence=presence)
        slicer = CountingSlicer()
        generator = TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2))

        dataset = AlignmentDataset([song], slicer, generator)
        self.assertEqual(slicer.selection_calls, 1)
        sample = dataset[0]
        self.assertEqual(slicer.selection_calls, 1)

        self.assertEqual(tuple(sample["hands"].shape), (2, 2, 21, 3))
        self.assertEqual(tuple(sample["chords"].shape), (2, 328))
        self.assertEqual(tuple(sample["target"].shape), (2, 2))
        self.assertNotIn("frame_mask", sample)
        self.assertNotIn("chord_mask", sample)
        self.assertEqual(
            sample["hand_presence"].tolist(),
            [[False, False], [True, False]],
        )
        torch.testing.assert_close(
            sample["target"][:, 0], torch.tensor([0.5, 1.0])
        )
        self.assertEqual(torch.count_nonzero(sample["target"][:, 1]).item(), 0)

        batch = collate_alignment_batch([sample, sample])
        self.assertEqual(tuple(batch["hands"].shape), (2, 2, 2, 21, 3))
        self.assertEqual(tuple(batch["target"].shape), (2, 2, 2))
        self.assertTrue(bool(batch["frame_mask"].all()))
        self.assertTrue(bool(batch["chord_mask"].all()))
        self.assertEqual(batch["record_id"], ["subject/video", "subject/video"])
        self.assertEqual(batch["frame_start"].dtype, torch.long)
        self.assertEqual(batch["fps"].dtype, torch.float64)

    def test_dataset_omits_windows_with_only_padded_chords(self) -> None:
        chords = (_chord(0, (_note(0, 0.1, 0.2, 60),)),)
        song = _song(chords, frame_count=11, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=4,
            stride_frames=3,
            chord_count=3,
        )
        generator = TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2))

        dataset = AlignmentDataset([song], slicer, generator)

        self.assertEqual(dataset.samples, ((0, 0, 4, (0,)),))

    def test_chord_mode_dataset_collates_variable_frame_lengths(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=3,
            stride_frames=2,
            chord_count=1,
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
        )
        first_sample = dataset[0]
        self.assertEqual(tuple(first_sample["target"].shape), (11, 3))
        self.assertEqual(first_sample["target"][-1, 2].item(), 1.0)

        short_song = _song(chords[:2], frame_count=8, fps=10.0)
        short_sample = AlignmentDataset(
            [short_song], slicer, dataset.target_generator
        )[0]
        batch = collate_alignment_batch(
            [first_sample, short_sample, dataset[2]]
        )

        self.assertEqual(tuple(batch["hands"].shape), (3, 26, 2, 21, 3))
        self.assertEqual(tuple(batch["target"].shape), (3, 26, 3))
        self.assertEqual(int(batch["frame_mask"][0].sum()), 11)
        self.assertFalse(bool(batch["frame_mask"][0, 11:].any()))
        self.assertEqual(torch.count_nonzero(batch["target"][0, 11:]).item(), 0)
        self.assertEqual(batch["chord_mask"][1].tolist(), [True, True, False])
        self.assertEqual(batch["chord_ids"][1].tolist(), [0, 1, -1])
        self.assertEqual(torch.count_nonzero(batch["target"][1, :, 2]).item(), 0)
        self.assertTrue(bool(batch["frame_mask"][2].all()))

        empty_song = _song((), frame_count=10, fps=10.0)
        self.assertEqual(
            len(AlignmentDataset([empty_song], slicer, dataset.target_generator)),
            0,
        )

    def test_random_hand_shift_resamples_each_access_and_clamps(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        dataset = AlignmentDataset(
            [song],
            FutureChordSlicer(
                mode="chord",
                fps=10.0,
                segment_frames=3,
                stride_frames=2,
            ),
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_hand_shift_seconds=2.0,
        )

        with patch(
            "airpiano.data.dataset.random.randint",
            side_effect=(2, -3, -5, 7, -20, 20),
        ):
            cropped = dataset[1]
            expanded = dataset[1]
            clamped = dataset[1]

        self.assertEqual(cropped["frame_start"], 12)
        torch.testing.assert_close(cropped["hands"], song.hands[12:18])
        self.assertEqual(tuple(cropped["target"].shape), (6, 3))
        self.assertEqual(cropped["target"][2, 1].item(), 1.0)
        self.assertEqual(expanded["frame_start"], 5)
        torch.testing.assert_close(expanded["hands"], song.hands[5:28])
        self.assertEqual(expanded["target"][5, 0].item(), 1.0)
        self.assertEqual(clamped["frame_start"], 0)
        torch.testing.assert_close(clamped["hands"], song.hands)
        for sample in (cropped, expanded, clamped):
            self.assertEqual(sample["chord_ids"].tolist(), [2, 3, 4])

    def test_random_hand_shift_resamples_crossed_bounds(self) -> None:
        chords = tuple(
            _chord(index, (_note(index, onset, onset + 0.1, 60 + index),))
            for index, onset in enumerate((0.2, 0.5, 1.0, 1.4, 2.0, 2.5))
        )
        song = _song(chords, frame_count=40, fps=10.0)
        dataset = AlignmentDataset(
            [song],
            FutureChordSlicer(
                mode="chord",
                fps=10.0,
                segment_frames=3,
                stride_frames=2,
            ),
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_hand_shift_seconds=2.0,
        )

        with patch(
            "airpiano.data.dataset.random.randint",
            side_effect=(20, -20, -2, 3),
        ) as randint:
            sample = dataset[1]

        self.assertEqual(randint.call_count, 4)
        self.assertEqual(sample["frame_start"], 8)
        torch.testing.assert_close(sample["hands"], song.hands[8:24])

    def test_build_dataset_enables_random_hand_shift_only_for_train(self) -> None:
        chords = (_chord(0, (_note(0, 0.1, 0.2, 60),)),)
        song = _song(chords, frame_count=10, fps=10.0)

        class StaticFileLoader:
            def load(self, **kwargs):
                del kwargs
                return (song,)

        config = OmegaConf.create(
            {
                "data": {
                    "splits": {
                        "train": "train.txt",
                        "valid": "valid.txt",
                        "test": "test.txt",
                    },
                    "midi_root": "midi",
                    "hand_root": "hands",
                },
                "dataset": {"random_hand_shift_seconds": 2.0},
            }
        )
        components = {
            "file_loader": StaticFileLoader(),
            "midi_loader": object(),
            "hand_loader": object(),
            "slicer": FutureChordSlicer(fps=10.0),
            "target_generator": TargetGenerator(
                ChordOnsetEvents(), TriangularSoftener(0.2)
            ),
        }

        train_dataset = build_dataset(config, "train", components=components)
        self.assertEqual(train_dataset.random_hand_shift_seconds, 2.0)
        self.assertEqual(train_dataset.max_random_hand_shift_frames, 20)
        for split in ("valid", "test"):
            dataset = build_dataset(config, split, components=components)
            self.assertEqual(dataset.random_hand_shift_seconds, 0.0)
            self.assertEqual(dataset.max_random_hand_shift_frames, 0)

    def test_target_uses_global_times_and_preserves_outer_soft_tail(self) -> None:
        chords = (
            _chord(0, (_note(0, 1.1, 1.2, 60),)),
            _chord(1, (_note(1, 1.31, 1.4, 64),)),
        )
        target = TargetGenerator(
            ChordOnsetEvents(), TriangularSoftener(radius_seconds=0.2)
        ).generate(
            chords,
            frame_start=10,
            frame_count=3,
            fps=10.0,
        )
        self.assertEqual(tuple(target.shape), (3, 2))
        torch.testing.assert_close(
            target[:, 0], torch.tensor([0.5, 1.0, 0.5]), atol=1e-6, rtol=0
        )
        self.assertGreater(target[-1, 1].item(), 0.0)

    def test_target_delay_moves_onset_peak_later(self) -> None:
        chord = _chord(0, (_note(0, 1.0, 1.1, 60),))
        generator = TargetGenerator(
            ChordOnsetEvents(delay_seconds=1.0),
            TriangularSoftener(radius_seconds=0.2),
        )

        target = generator.generate(
            (chord,),
            frame_start=20,
            frame_count=21,
            fps=20.0,
        )

        self.assertEqual(generator.event_generator.delay_seconds, 1.0)
        self.assertEqual(int(target[:, 0].argmax()), 20)
        self.assertEqual(target[20, 0].item(), 1.0)

    def test_last_chord_target_is_zero_before_onset_then_one_hot(self) -> None:
        chords = (
            _chord(0, (_note(0, 0.2, 0.3, 60),)),
            _chord(1, (_note(1, 0.5, 0.6, 64),)),
        )
        generator = TargetGenerator(
            ChordOnsetEvents(),
            TriangularSoftener(radius_seconds=0.2),
            target_mode="last_chord",
        )

        target = generator.generate(
            chords,
            frame_start=0,
            frame_count=8,
            fps=10.0,
        )

        torch.testing.assert_close(
            target,
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
