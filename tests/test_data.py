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
    EventSequence,
    FileLoader,
    FilePair,
    FutureChordSlicer,
    HandLoader,
    HandSequence,
    InitializerChordMerger,
    MidiSequence,
    MidoNoteReader,
    Note,
    NoteEvent,
    OffsetScaleTransform,
    Song,
    RelativeStemPairer,
    TargetGenerator,
    TriangularSoftener,
    build_dataset,
    collate_alignment_batch,
)
from airpiano.data.midi import quantize_duration_seconds, quantize_duration_ticks


def _note(
    note_id: int,
    onset: float,
    offset: float,
    pitch: int,
    velocity: int = 80,
    onset_tick: int | None = None,
    offset_tick: int | None = None,
) -> Note:
    return Note(
        note_id,
        onset,
        offset,
        pitch,
        velocity,
        onset_tick,
        offset_tick,
    )


def _chord(
    chord_id: int,
    notes: tuple[Note, ...],
    duration: float | None = None,
    quantized_dur: torch.Tensor | None = None,
) -> Chord:
    events = tuple(NoteEvent(note, False) for note in notes)
    time = events[0].time
    if duration is None:
        duration = max(note.offset for note in notes) - time
    if quantized_dur is None:
        quantized_dur = quantize_duration_seconds(
            torch.tensor(duration, dtype=torch.float64)
        )
    return Chord(
        id=chord_id,
        events=events,
        onset_notes=notes,
        offset_notes=(),
        time=time,
        duration=duration,
        quantized_dur=quantized_dur,
    )


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
    notes_by_id = {
        event.note.id: event.note
        for chord in chords
        for event in chord.events
    }
    return Song(
        record_id="subject/video",
        notes=tuple(notes_by_id[index] for index in sorted(notes_by_id)),
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

    def test_pairer_requires_configured_beat_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            midi_root = root / "midi"
            hand_root = root / "hands"
            beat_root = root / "beats"
            for modality_root, suffix in (
                (midi_root, ".mid"),
                (hand_root, ".npz"),
            ):
                modality_root.mkdir()
                (modality_root / f"complete{suffix}").touch()
                (modality_root / f"missing-beats{suffix}").touch()
            beat_root.mkdir()
            (beat_root / "complete.txt").touch()
            split = root / "split.txt"
            split.write_text("complete\nmissing-beats\n", encoding="utf-8")

            printed = io.StringIO()
            with redirect_stdout(printed):
                pairs = RelativeStemPairer().pair(
                    split,
                    midi_root,
                    hand_root,
                    beat_root,
                )

        self.assertEqual([pair.record_id for pair in pairs], ["complete"])
        self.assertEqual(pairs[0].beat_path, beat_root / "complete.txt")
        self.assertIn("missing beats", printed.getvalue())
        self.assertIn("paired=1 missing=1", printed.getvalue())

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
    def test_event_sequence_expands_offsets_and_orders_actions(self) -> None:
        first = _note(0, 0.0, 0.4, 60, onset_tick=0, offset_tick=4)
        second = _note(1, 0.2, 0.3, 64, onset_tick=2, offset_tick=3)

        onset_only = EventSequence.from_notes((first, second))
        with_offsets = EventSequence.from_notes(
            (first, second), include_offsets=True
        )

        self.assertFalse(onset_only.include_offsets)
        self.assertEqual(
            [(event.note.id, event.is_offset) for event in onset_only.events],
            [(0, False), (1, False)],
        )
        self.assertTrue(with_offsets.include_offsets)
        self.assertEqual(
            [
                (event.note.id, event.is_offset, event.pitch, event.time, event.tick)
                for event in with_offsets.events
            ],
            [
                (0, False, 60, 0.0, 0),
                (1, False, 64, 0.2, 2),
                (1, True, 64, 0.3, 3),
                (0, True, 60, 0.4, 4),
            ],
        )

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

    def test_reader_quantizes_to_beat_ticks_and_preserves_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            midi_path = root / "notes.mid"
            beat_path = root / "beats.txt"
            beat_path.write_text(
                "1.0\t4\n2.0\t1\n3.0\t2\n",
                encoding="utf-8",
            )
            midi = mido.MidiFile(type=0, ticks_per_beat=1000)
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
            track.append(mido.Message("note_on", note=60, velocity=70, time=500))
            track.append(mido.Message("note_off", note=60, velocity=0, time=625))
            track.append(mido.Message("note_on", note=62, velocity=71, time=250))
            track.append(mido.Message("note_off", note=62, velocity=0, time=750))
            track.append(mido.Message("note_on", note=64, velocity=72, time=750))
            track.append(mido.Message("note_off", note=64, velocity=0, time=375))
            midi.tracks.append(track)
            midi.save(midi_path)

            parsed = MidoNoteReader(quantization=True, subdivision=4).read(
                midi_path,
                beat_path,
            )

        self.assertEqual(
            [(note.onset_tick, note.offset_tick) for note in parsed],
            [(-2, 1), (2, 5), (8, 9)],
        )
        self.assertEqual([note.onset for note in parsed], [0.5, 1.375, 2.875])
        self.assertEqual([note.offset for note in parsed], [1.125, 2.125, 3.25])

    def test_reader_quantizes_to_filtered_tick_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            beat_path = Path(directory) / "beats.txt"
            beat_path.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")
            reader = MidoNoteReader(
                quantization=True,
                subdivision=12,
                allowed_tick_offsets=(0, 3, 4, 6, 8, 9),
            )
            tick_times, tick_positions = reader._read_ticks(beat_path)

        self.assertEqual(
            tick_positions,
            (0, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 21, 24),
        )

        def quantize(time_seconds: float) -> int:
            return reader._nearest_tick(
                time_seconds,
                tick_times,
                tick_positions,
            )

        self.assertEqual(
            tuple(quantize(1.0 + index / 12) for index in range(12)),
            (0, 0, 3, 3, 4, 6, 6, 8, 8, 9, 9, 12),
        )
        self.assertEqual(quantize(1.125), 3)
        self.assertEqual(quantize(0.75), -3)
        self.assertEqual(quantize(0.875), 0)
        self.assertEqual(quantize(3.125), 27)
        self.assertEqual(quantize(3.25), 27)

        notes = (
            _note(
                0,
                1.25,
                1.75,
                60,
                onset_tick=quantize(1.25),
                offset_tick=quantize(1.75),
            ),
            _note(
                1,
                1.5,
                2.0,
                64,
                onset_tick=quantize(1.5),
                offset_tick=quantize(2.0),
            ),
        )
        chords = InitializerChordMerger().merge(
            EventSequence.from_notes(notes),
            quantization=True,
        )
        self.assertEqual(
            [chord.duration_ticks for chord in chords],
            [3, 6],
        )
        self.assertEqual(
            [chord.quantized_dur.argmax().item() for chord in chords],
            [3, 6],
        )

    def test_reader_builds_exact_sixth_and_eighth_24_tick_grid(self) -> None:
        offsets = (0, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 21)
        with tempfile.TemporaryDirectory() as directory:
            beat_path = Path(directory) / "beats.txt"
            beat_path.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")
            reader = MidoNoteReader(
                quantization=True,
                subdivision=24,
                allowed_tick_offsets=offsets,
            )
            _, tick_positions = reader._read_ticks(beat_path)

        self.assertEqual(
            tick_positions,
            (
                0, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 21,
                24, 27, 28, 30, 32, 33, 36, 39, 40, 42, 44, 45,
                48,
            ),
        )

    def test_reader_clamps_filtered_grid_out_of_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            beat_path = Path(directory) / "beats.txt"
            beat_path.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")
            reader = MidoNoteReader(
                quantization=True,
                subdivision=12,
                allowed_tick_offsets=(0, 3, 4, 6, 8, 9),
                out_of_bounds_mode="clamp",
            )
            tick_times, tick_positions = reader._read_ticks(beat_path)

        def quantize(time_seconds: float) -> int:
            return reader._nearest_tick(
                time_seconds,
                tick_times,
                tick_positions,
            )

        self.assertEqual(quantize(0.75), 0)
        self.assertEqual(quantize(1.125), 3)
        self.assertEqual(quantize(3.125), 24)
        self.assertEqual(quantize(3.25), 24)

    def test_reader_rejects_invalid_filtered_tick_offsets(self) -> None:
        invalid_offsets = (
            ((0, 3.5), "integers"),
            ((0, 3, 3), "unique"),
            ((3, 4, 6), "include 0"),
            ((0, 12), r"\[0, subdivision\)"),
        )
        for offsets, message in invalid_offsets:
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(ValueError, message):
                    MidoNoteReader(
                        subdivision=12,
                        allowed_tick_offsets=offsets,
                    )

    def test_reader_rejects_unknown_out_of_bounds_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "out_of_bounds_mode"):
            MidoNoteReader(out_of_bounds_mode="wrap")

    def test_chord_merger_uses_tick_groups_and_direct_tick_durations(self) -> None:
        notes = (
            _note(0, 0.0, 0.02, 60, onset_tick=0, offset_tick=0),
            _note(1, 0.2, 0.3, 64, onset_tick=0, offset_tick=4),
            _note(2, 0.21, 0.3, 67, onset_tick=3, offset_tick=3),
            _note(3, 0.22, 20.0, 71, onset_tick=3, offset_tick=250),
        )

        chords = InitializerChordMerger(
            tolerance_seconds=0.05,
            time_mode="mean",
        ).merge(
            EventSequence.from_notes(notes),
            quantization=True,
        )

        self.assertEqual(len(chords), 2)
        self.assertEqual(
            [[note.id for note in chord.onset_notes] for chord in chords],
            [[0, 1], [2, 3]],
        )
        self.assertAlmostEqual(chords[0].time, 0.1)
        self.assertAlmostEqual(chords[1].time, 0.215)
        self.assertAlmostEqual(chords[0].duration, 0.115)
        self.assertAlmostEqual(chords[1].duration, 19.785)
        self.assertEqual([chord.duration_ticks for chord in chords], [3, 247])
        self.assertEqual(
            [chord.quantized_dur.argmax().item() for chord in chords],
            [3, 199],
        )
        torch.testing.assert_close(
            chords[0].quantized_dur,
            quantize_duration_ticks(torch.tensor(3)),
        )

        zero = InitializerChordMerger().merge(
            EventSequence.from_notes(
                (_note(0, 1.0, 1.01, 60, onset_tick=7, offset_tick=7),)
            ),
            quantization=True,
        )
        self.assertEqual(zero[0].duration_ticks, 0)
        self.assertEqual(zero[0].quantized_dur.argmax().item(), 0)

    def test_chord_merger_uses_one_based_96_tick_durations(self) -> None:
        notes = (
            _note(0, 0.0, 0.1, 60, onset_tick=0, offset_tick=1),
            _note(1, 0.1, 9.7, 64, onset_tick=1, offset_tick=97),
            _note(2, 9.7, 9.71, 67, onset_tick=97, offset_tick=97),
        )

        chords = InitializerChordMerger(
            duration_tick_bins=96,
            minimum_duration_ticks=1,
        ).merge(
            EventSequence.from_notes(notes),
            quantization=True,
        )

        self.assertEqual(
            [chord.duration_ticks for chord in chords],
            [1, 96, 0],
        )
        self.assertEqual(
            [tuple(chord.quantized_dur.shape) for chord in chords],
            [(96,), (96,), (96,)],
        )
        self.assertEqual(
            [chord.quantized_dur.argmax().item() for chord in chords],
            [0, 95, 0],
        )

    def test_chord_merger_splits_same_tick_notes_in_absolute_time_order(
        self,
    ) -> None:
        notes = (
            _note(0, 0.2, 0.6, 60, onset_tick=0, offset_tick=4),
            _note(1, 0.1, 0.5, 64, onset_tick=0, offset_tick=4),
            _note(2, 0.3, 0.8, 67, onset_tick=3, offset_tick=8),
        )

        chords = InitializerChordMerger(
            time_mode="mean",
            split_notes=True,
        ).merge(
            EventSequence.from_notes(notes),
            quantization=True,
        )

        self.assertEqual(len(chords), 3)
        self.assertEqual(
            [[note.id for note in chord.onset_notes] for chord in chords],
            [[1], [0], [2]],
        )
        torch.testing.assert_close(
            torch.tensor([chord.time for chord in chords]),
            torch.tensor([0.1, 0.2, 0.3]),
        )
        self.assertEqual(
            [chord.duration_ticks for chord in chords],
            [0, 3, 5],
        )
        self.assertEqual(
            [chord.quantized_dur.argmax().item() for chord in chords],
            [0, 3, 5],
        )

        song = _song(chords, frame_count=10, fps=10.0)
        tokens = FutureChordSlicer().slice(song, 0, 1, (0, 1, 2))["chords"]
        self.assertEqual(
            [tokens[index, :128].argmax().item() for index in range(3)],
            [64, 60, 67],
        )
        self.assertEqual(tokens[:, :128].sum(dim=-1).tolist(), [1.0] * 3)

    def test_chord_merger_groups_mixed_quantized_actions(self) -> None:
        notes = (
            _note(0, 0.0, 0.31, 60, onset_tick=0, offset_tick=3),
            _note(1, 0.02, 0.4, 64, onset_tick=0, offset_tick=4),
            _note(2, 0.29, 0.5, 67, onset_tick=3, offset_tick=5),
        )

        chords = InitializerChordMerger(time_mode="mean").merge(
            EventSequence.from_notes(notes, include_offsets=True),
            quantization=True,
        )

        self.assertEqual(len(chords), 4)
        torch.testing.assert_close(
            torch.tensor([chord.time for chord in chords]),
            torch.tensor([0.01, 0.30, 0.40, 0.50]),
        )
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in chords]),
            torch.tensor([0.29, 0.10, 0.10, 0.0]),
        )
        self.assertEqual(
            [chord.duration_ticks for chord in chords],
            [3, 1, 1, 0],
        )
        self.assertEqual(
            [note.id for note in chords[1].onset_notes],
            [2],
        )
        self.assertEqual(
            [note.id for note in chords[1].offset_notes],
            [0],
        )
        self.assertEqual(
            [(event.note.id, event.is_offset) for event in chords[1].events],
            [(2, False), (0, True)],
        )
        self.assertEqual(chords[-1].onset_notes, ())
        self.assertEqual(chords[-1].offset_notes, (notes[2],))
        self.assertEqual(chords[-1].quantized_dur.argmax().item(), 0)

    def test_chord_merger_is_inclusive_and_initializer_anchored(self) -> None:
        notes = tuple(
            _note(index, onset, onset + 0.2, 60 + index)
            for index, onset in enumerate((0.0, 0.05, 0.09, 0.139, 0.141))
        )

        chords = InitializerChordMerger(
            tolerance_seconds=0.05,
            time_mode="mean",
        ).merge(
            EventSequence.from_notes(notes)
        )

        self.assertEqual([chord.id for chord in chords], [0, 1, 2])
        self.assertEqual(
            [[note.id for note in chord.onset_notes] for chord in chords],
            [[0, 1], [2, 3], [4]],
        )
        self.assertIs(chords[0].onset_notes[0], notes[0])
        torch.testing.assert_close(
            torch.tensor([chord.duration for chord in chords]),
            torch.tensor([0.0895, 0.0265, 0.2]),
        )
        expected_quantized = quantize_duration_seconds(
            torch.tensor([0.0895, 0.0265, 0.2], dtype=torch.float64)
        )
        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in chords]),
            expected_quantized,
        )
        self.assertEqual(chords[0].quantized_dur.dtype, torch.float32)
        self.assertEqual(tuple(chords[0].quantized_dur.shape), (200,))

    def test_chord_merger_switches_between_first_and_mean_time(self) -> None:
        notes = (
            _note(0, 0.0, 0.4, 60),
            _note(1, 0.04, 0.5, 64),
            _note(2, 1.0, 1.2, 67),
        )
        events = EventSequence.from_notes(notes)

        first = InitializerChordMerger(time_mode="first").merge(events)
        mean = InitializerChordMerger(time_mode="mean").merge(events)

        self.assertEqual([chord.time for chord in first], [0.0, 1.0])
        self.assertEqual([chord.time for chord in mean], [0.02, 1.0])
        self.assertEqual(first[0].duration, 1.0)
        self.assertEqual(mean[0].duration, 0.98)

    def test_chord_merger_uses_next_onset_then_latest_final_release(self) -> None:
        notes = (
            _note(0, 0.0, 4.0, 60),
            _note(1, 1.0, 1.2, 64),
            _note(2, 1.02, 1.7, 67),
        )

        chords = InitializerChordMerger(
            tolerance_seconds=0.05,
            time_mode="mean",
        ).merge(
            EventSequence.from_notes(notes)
        )

        self.assertEqual(len(chords), 2)
        self.assertAlmostEqual(chords[0].duration, 1.01)
        self.assertAlmostEqual(chords[1].duration, 0.69)
        torch.testing.assert_close(
            torch.stack([chord.quantized_dur for chord in chords]),
            quantize_duration_seconds(torch.tensor([1.01, 0.69])),
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
            chords = merger.merge(EventSequence.from_notes(notes))

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
            chords = merger.merge(EventSequence.from_notes(notes))

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

        chords = merger.merge(EventSequence.from_notes(notes))
        scaled_chords = merger.merge(EventSequence.from_notes(scaled_notes))

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

        self.assertEqual(merger.merge(EventSequence.from_notes(())), ())
        chords = merger.merge(
            EventSequence.from_notes((_note(0, 2.0, 2.4, 60),))
        )

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

    def test_chord_merger_rejects_unknown_time_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_mode"):
            InitializerChordMerger(time_mode="median")


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

    def test_tick_durations_support_one_based_96_bins(self) -> None:
        encoded = quantize_duration_ticks(
            torch.tensor([0, 1, 2, 96, 120]),
            num_bins=96,
            minimum_tick=1,
        )

        self.assertEqual(tuple(encoded.shape), (5, 96))
        self.assertEqual(
            encoded.argmax(dim=-1).tolist(),
            [0, 0, 1, 95, 95],
        )
        self.assertTrue(bool((encoded.sum(dim=-1) == 1.0).all()))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            quantize_duration_ticks(
                torch.tensor([-1]),
                num_bins=96,
                minimum_tick=1,
            )

        legacy = quantize_duration_ticks(torch.tensor([0, 199, 200]))
        self.assertEqual(tuple(legacy.shape), (3, 200))
        self.assertEqual(legacy.argmax(dim=-1).tolist(), [0, 199, 199])

    def test_tick_durations_support_one_based_48_bins(self) -> None:
        encoded = quantize_duration_ticks(
            torch.tensor([0, 1, 48, 49]),
            num_bins=48,
            minimum_tick=1,
        )

        self.assertEqual(tuple(encoded.shape), (4, 48))
        self.assertEqual(
            encoded.argmax(dim=-1).tolist(),
            [0, 0, 47, 47],
        )
        self.assertTrue(bool((encoded.sum(dim=-1) == 1.0).all()))

    def test_starts_include_exact_right_aligned_tail(self) -> None:
        slicer = FutureChordSlicer(
            fps=10.0, segment_frames=4, stride_frames=3, chord_count=3
        )
        self.assertEqual(slicer.starts(3), (0,))
        self.assertEqual(slicer.starts(11), (0, 3, 6, 7))

    def test_zero_hand_fraction_must_be_between_zero_and_one(self) -> None:
        for value in (-0.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "max_zero_hand_fraction"
            ):
                FutureChordSlicer(max_zero_hand_fraction=value)

    def test_frame_mode_filters_only_fully_empty_frames_above_limit(self) -> None:
        chords = (_chord(0, (_note(0, 0.0, 1.0, 60),)),)
        presence = torch.ones((10, 2), dtype=torch.bool)
        presence[0] = torch.tensor([True, False])
        presence[1] = False
        presence[5:7] = False
        song = _song(
            chords,
            frame_count=10,
            fps=10.0,
            presence=presence,
        )
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=5,
            stride_frames=5,
            max_zero_hand_fraction=0.2,
        )

        self.assertEqual(slicer.windows(song), ((0, 5, (0,)),))

    def test_slice_selects_overlaps_then_future_and_collapses_duplicate_pitch(
        self,
    ) -> None:
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
        self.assertEqual(tuple(window["chords"].shape), (2, 456))
        self.assertEqual(window["chords"][0, 60].item(), 1.0)
        self.assertEqual(window["chords"][0, 0].item(), 1.0)
        self.assertEqual(window["chords"][0, 127].item(), 1.0)
        self.assertEqual(window["chords"][0].sum().item(), 4.0)
        self.assertEqual(len(window["selected_chords"][0].onset_notes), 4)
        torch.testing.assert_close(window["hands"], song.hands[19:23])

        no_overlap = slicer.candidate_positions(chords, frame_start=6)
        self.assertEqual(no_overlap, (1, 2))

        sustained = (_chord(0, (_note(6, 0.0, 2.0, 72),)),)
        self.assertEqual(
            slicer.candidate_positions(sustained, frame_start=10),
            (0,),
        )

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

        self.assertEqual(tuple(middle.shape), (1, 456))
        self.assertEqual(middle[0, :128].argmax().item(), 64)
        self.assertEqual(middle[0, 128:256].sum().item(), 0.0)
        self.assertEqual(middle[0, 256:].sum().item(), 1.0)
        torch.testing.assert_close(middle[0, 256:], middle_duration)
        torch.testing.assert_close(final[0, 256:], final_duration)

        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
        )
        batch = collate_alignment_batch([dataset[0], dataset[-1]])
        self.assertEqual(tuple(batch["chords"].shape), (2, 3, 456))

    def test_compact_duration_tokens_slice_and_collate_as_224_features(
        self,
    ) -> None:
        durations = quantize_duration_ticks(
            torch.tensor([1, 96]),
            num_bins=96,
            minimum_tick=1,
        )
        chords = (
            _chord(
                0,
                (_note(0, 0.1, 0.2, 60),),
                quantized_dur=durations[0],
            ),
            _chord(
                1,
                (_note(1, 0.6, 0.7, 64),),
                quantized_dur=durations[1],
            ),
        )
        song = _song(chords, frame_count=12, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=6,
            stride_frames=6,
            chord_count=2,
            chord_duration_bins=96,
            compact_chord_tokens=True,
        )

        tokens = slicer.slice(song, 0, 6, (0, 1))["chords"]
        self.assertEqual(tuple(tokens.shape), (2, 224))
        self.assertEqual(tokens[:, :128].argmax(dim=-1).tolist(), [60, 64])
        torch.testing.assert_close(tokens[:, 128:], durations)

        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
        )
        batch = collate_alignment_batch([dataset[0], dataset[-1]])
        self.assertEqual(tuple(batch["chords"].shape), (2, 2, 224))

    def test_compact_duration48_tokens_have_176_features(self) -> None:
        durations = quantize_duration_ticks(
            torch.tensor([0, 48]),
            num_bins=48,
            minimum_tick=1,
        )
        chords = (
            _chord(
                0,
                (_note(0, 0.1, 0.2, 60),),
                quantized_dur=durations[0],
            ),
            _chord(
                1,
                (_note(1, 0.6, 0.7, 64),),
                quantized_dur=durations[1],
            ),
        )
        song = _song(chords, frame_count=12, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=6,
            stride_frames=6,
            chord_count=2,
            chord_duration_bins=48,
            compact_chord_tokens=True,
        )

        tokens = slicer.slice(song, 0, 6, (0, 1))["chords"]

        self.assertEqual(tuple(tokens.shape), (2, 176))
        self.assertEqual(tokens[:, :128].argmax(dim=-1).tolist(), [60, 64])
        torch.testing.assert_close(tokens[:, 128:], durations)

    def test_offset_events_drive_candidates_targets_and_token_planes(self) -> None:
        notes = (
            _note(0, 0.0, 0.31, 60, onset_tick=0, offset_tick=3),
            _note(1, 0.02, 0.4, 64, onset_tick=0, offset_tick=4),
            _note(2, 0.29, 0.5, 67, onset_tick=3, offset_tick=5),
        )
        chords = InitializerChordMerger(time_mode="mean").merge(
            EventSequence.from_notes(notes, include_offsets=True),
            quantization=True,
        )
        song = _song(chords, frame_count=10, fps=10.0)
        slicer = FutureChordSlicer(
            fps=10.0,
            segment_frames=1,
            stride_frames=1,
            chord_count=4,
        )

        positions = slicer.candidate_positions(chords, frame_start=2)
        sample = slicer.slice(song, 2, 4, (1,))
        target = TargetGenerator(
            ChordOnsetEvents(), TriangularSoftener(0.2)
        ).generate(
            sample["selected_chords"],
            frame_start=2,
            frame_count=2,
            fps=10.0,
        )

        self.assertEqual(positions, (0, 1, 2, 3))
        self.assertEqual(tuple(sample["chords"].shape), (1, 456))
        self.assertEqual(sample["chords"][0, 67].item(), 1.0)
        self.assertEqual(sample["chords"][0, 128 + 60].item(), 1.0)
        self.assertEqual(sample["chords"][0, 256 + 1].item(), 1.0)
        self.assertEqual(target[:, 0].tolist(), [0.5, 1.0])

        cropped = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=1,
            crop_to_chord_bounds=True,
        )
        self.assertEqual(cropped.chord_frame_bounds(song, (0, 1)), (0, 4))
        self.assertEqual(cropped.chord_frame_bounds(song, (1, 2)), (3, 5))
        self.assertEqual(cropped.chord_frame_bounds(song, (2, 3)), (4, 5))

        compact = FutureChordSlicer(
            fps=10.0,
            segment_frames=1,
            stride_frames=1,
            chord_count=4,
            compact_chord_tokens=True,
        )
        with self.assertRaisesRegex(ValueError, "offset events"):
            compact.slice(song, 2, 4, (1,))

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
        self.assertEqual(tuple(window["chords"].shape), (3, 456))
        self.assertEqual(window["chord_ids"].tolist(), [2, 3, 4])
        self.assertNotIn("frame_mask", window)
        self.assertNotIn("chord_mask", window)
        torch.testing.assert_close(window["hands"], song.hands[10:21])

    def test_chord_crop_changes_only_song_boundary_windows(self) -> None:
        chords = (
            _chord(0, (_note(0, 0.21, 0.4, 60),)),
            _chord(
                1,
                (
                    _note(1, 0.55, 0.91, 61),
                    _note(2, 0.57, 1.04, 62),
                ),
            ),
            _chord(2, (_note(3, 1.21, 1.4, 63),)),
            _chord(3, (_note(4, 1.55, 1.93, 64),)),
        )
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=2,
            end_padding_seconds=3.0,
            hand_context_chords=1,
            crop_to_chord_bounds=True,
        )

        song = _song(chords, frame_count=30, fps=10.0)
        self.assertEqual(
            slicer.windows(song),
            (
                (2, 30, (0, 1)),
                (5, 20, (2, 3)),
            ),
        )

        short_song = _song(chords, frame_count=18, fps=10.0)
        self.assertEqual(slicer.windows(short_song)[1], (5, 18, (2, 3)))

    def test_zero_hand_filter_uses_cropped_chord_bounds(self) -> None:
        chords = (
            _chord(0, (_note(0, 0.21, 0.4, 60),)),
            _chord(1, (_note(1, 0.55, 1.04, 61),)),
        )
        presence = torch.zeros((20, 2), dtype=torch.bool)
        presence[3:11, 0] = True
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=2,
            crop_to_chord_bounds=True,
            max_zero_hand_fraction=0.2,
        )

        song = _song(chords, frame_count=20, fps=10.0, presence=presence)
        self.assertEqual(slicer.windows(song), ((2, 11, (0, 1)),))

        presence[3:5] = False
        filtered_song = _song(
            chords,
            frame_count=20,
            fps=10.0,
            presence=presence,
        )
        self.assertEqual(slicer.windows(filtered_song), ())

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
        self.assertEqual(tuple(window["chords"].shape), (3, 456))
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
        self.assertEqual(tuple(sample["chords"].shape), (2, 456))
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

    def test_random_pitch_shift_is_global_resampled_and_boundary_safe(
        self,
    ) -> None:
        durations = quantize_duration_ticks(
            torch.tensor([12, 24]),
            num_bins=96,
            minimum_tick=1,
        )
        chords = (
            _chord(
                0,
                (_note(0, 0.1, 0.2, 3),),
                quantized_dur=durations[0],
            ),
            _chord(
                1,
                (_note(1, 0.6, 0.7, 125),),
                quantized_dur=durations[1],
            ),
        )
        song = _song(chords, frame_count=10, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=2,
            chord_duration_bins=96,
            compact_chord_tokens=True,
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_pitch_shift_semitones=6,
        )
        original = slicer.slice(song, 0, 10, (0, 1))["chords"]

        with patch(
            "airpiano.data.dataset.random.randint",
            side_effect=(2, -3, 0),
        ) as randint:
            shifted_up = dataset[0]
            shifted_down = dataset[0]
            unshifted = dataset[0]

        self.assertEqual(
            [item.args for item in randint.call_args_list],
            [(-3, 2), (-3, 2), (-3, 2)],
        )
        self.assertEqual(
            shifted_up["chords"][:, :128].argmax(dim=-1).tolist(),
            [5, 127],
        )
        self.assertEqual(
            shifted_down["chords"][:, :128].argmax(dim=-1).tolist(),
            [0, 122],
        )
        torch.testing.assert_close(unshifted["chords"], original)
        for sample in (shifted_up, shifted_down, unshifted):
            self.assertEqual(
                int(torch.count_nonzero(sample["chords"][:, :128])),
                2,
            )
            torch.testing.assert_close(sample["chords"][:, 128:], durations)
            torch.testing.assert_close(sample["hands"], song.hands)
            self.assertEqual(sample["chord_ids"].tolist(), [0, 1])
            torch.testing.assert_close(sample["target"], unshifted["target"])

    def test_random_pitch_shift_moves_legacy_onset_and_offset_planes(
        self,
    ) -> None:
        notes = (
            _note(0, 0.1, 0.5, 60),
            _note(1, 0.1, 0.6, 64),
        )
        chords = InitializerChordMerger().merge(
            EventSequence.from_notes(notes, include_offsets=True)
        )
        song = _song(chords, frame_count=10, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=len(chords),
            stride_frames=len(chords),
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_pitch_shift_semitones=6,
        )
        original = slicer.slice(
            song, 0, 10, tuple(range(len(chords)))
        )["chords"]

        with patch(
            "airpiano.data.dataset.random.randint",
            return_value=4,
        ) as randint:
            shifted = dataset[0]["chords"]

        randint.assert_called_once_with(-6, 6)
        onset_indices = torch.nonzero(shifted[:, :128], as_tuple=False)
        offset_indices = torch.nonzero(shifted[:, 128:256], as_tuple=False)
        self.assertEqual(onset_indices.tolist(), [[0, 64], [0, 68]])
        self.assertEqual(offset_indices.tolist(), [[1, 64], [2, 68]])
        torch.testing.assert_close(shifted[:, 256:], original[:, 256:])

    def test_random_duration_scale_is_global_resampled_and_rounds_up(
        self,
    ) -> None:
        duration_ticks = torch.tensor([1, 3, 48])
        durations = quantize_duration_ticks(
            duration_ticks,
            num_bins=96,
            minimum_tick=1,
        )
        chords = tuple(
            _chord(
                index,
                (_note(index, 0.1 + 0.3 * index, 0.2 + 0.3 * index, 60 + index),),
                quantized_dur=durations[index],
            )
            for index in range(3)
        )
        song = _song(chords, frame_count=10, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=3,
            stride_frames=3,
            chord_duration_bins=96,
            compact_chord_tokens=True,
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_duration_scale_factors=(0.5, 1.0, 2.0),
            minimum_duration_ticks=1,
        )
        original = slicer.slice(song, 0, 10, (0, 1, 2))["chords"]

        with patch(
            "airpiano.data.dataset.random.choice",
            side_effect=(0.5, 1.0, 2.0),
        ) as choice:
            halved = dataset[0]
            unchanged = dataset[0]
            doubled = dataset[0]

        self.assertEqual(
            [tuple(item.args[0]) for item in choice.call_args_list],
            [(0.5, 1.0, 2.0)] * 3,
        )
        self.assertEqual(
            halved["chords"][:, 128:].argmax(dim=-1).tolist(),
            [0, 1, 23],
        )
        torch.testing.assert_close(unchanged["chords"], original)
        self.assertEqual(
            doubled["chords"][:, 128:].argmax(dim=-1).tolist(),
            [1, 5, 95],
        )
        for sample in (halved, unchanged, doubled):
            torch.testing.assert_close(
                sample["chords"][:, :128],
                original[:, :128],
            )
            self.assertTrue(
                bool((sample["chords"][:, 128:].sum(dim=-1) == 1.0).all())
            )
            torch.testing.assert_close(sample["hands"], song.hands)
            self.assertEqual(sample["chord_ids"].tolist(), [0, 1, 2])
            torch.testing.assert_close(sample["target"], unchanged["target"])
        for chord, duration in zip(chords, durations):
            torch.testing.assert_close(chord.quantized_dur, duration)

    def test_random_duration_scale_excludes_out_of_bounds_factors(
        self,
    ) -> None:
        durations = quantize_duration_ticks(
            torch.tensor([49, 96]),
            num_bins=96,
            minimum_tick=1,
        )
        chords = (
            _chord(
                0,
                (_note(0, 0.1, 0.2, 60),),
                quantized_dur=durations[0],
            ),
            _chord(
                1,
                (_note(1, 0.6, 0.7, 64),),
                quantized_dur=durations[1],
            ),
        )
        song = _song(chords, frame_count=10, fps=10.0)
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=2,
            stride_frames=2,
            chord_duration_bins=96,
            compact_chord_tokens=True,
        )
        dataset = AlignmentDataset(
            [song],
            slicer,
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.2)),
            random_duration_scale_factors=(0.5, 1.0, 2.0),
            minimum_duration_ticks=1,
        )

        with patch(
            "airpiano.data.dataset.random.choice",
            return_value=0.5,
        ) as choice:
            sample = dataset[0]

        self.assertEqual(tuple(choice.call_args.args[0]), (0.5, 1.0))
        self.assertEqual(
            sample["chords"][:, 128:].argmax(dim=-1).tolist(),
            [24, 47],
        )
        self.assertTrue(
            bool((sample["chords"][:, 128:].sum(dim=-1) == 1.0).all())
        )

    def test_build_dataset_enables_random_augmentations_only_for_train(
        self,
    ) -> None:
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
                "dataset": {
                    "random_hand_shift_seconds": 2.0,
                    "random_pitch_shift_semitones": 6,
                    "random_duration_scale_factors": [0.5, 1.0, 2.0],
                    "midi_loader": {
                        "chord_merger": {
                            "minimum_duration_ticks": 1,
                        }
                    },
                },
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
        self.assertEqual(train_dataset.random_pitch_shift_semitones, 6)
        self.assertEqual(
            train_dataset.random_duration_scale_factors,
            (0.5, 1.0, 2.0),
        )
        self.assertEqual(train_dataset.minimum_duration_ticks, 1)
        for split in ("valid", "test"):
            dataset = build_dataset(config, split, components=components)
            self.assertEqual(dataset.random_hand_shift_seconds, 0.0)
            self.assertEqual(dataset.max_random_hand_shift_frames, 0)
            self.assertEqual(dataset.random_pitch_shift_semitones, 0)
            self.assertEqual(
                dataset.random_duration_scale_factors,
                (),
            )
            self.assertEqual(dataset.minimum_duration_ticks, 1)

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

    def test_velocity_target_scales_merged_and_single_note_columns(self) -> None:
        chords = (
            _chord(
                0,
                (
                    _note(0, 1.0, 1.2, 60, velocity=32),
                    _note(1, 1.0, 1.2, 64, velocity=96),
                ),
            ),
            _chord(1, (_note(2, 1.5, 1.7, 67, velocity=32),)),
            _chord(2, (_note(3, 2.0, 2.2, 71, velocity=96),)),
        )
        target = TargetGenerator(
            ChordOnsetEvents(),
            TriangularSoftener(radius_seconds=0.2),
            velocity_target=True,
        ).generate(
            chords,
            frame_start=9,
            frame_count=13,
            fps=10.0,
        )

        torch.testing.assert_close(
            target.amax(dim=0),
            torch.tensor([0.5, 0.25, 0.75]),
        )
        torch.testing.assert_close(
            target[[0, 5, 10], [0, 1, 2]],
            torch.tensor([0.25, 0.125, 0.375]),
            atol=1e-6,
            rtol=0,
        )

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

    def test_negative_target_delay_moves_peak_three_frames_earlier(self) -> None:
        chord = _chord(0, (_note(0, 1.0, 1.1, 60),))
        generator = TargetGenerator(
            ChordOnsetEvents(delay_seconds=-0.1),
            TriangularSoftener(radius_seconds=0.2),
        )

        target = generator.generate(
            (chord,),
            frame_start=0,
            frame_count=31,
            fps=30.0,
        )
        peak_frame = int(target[:, 0].argmax())

        self.assertEqual(generator.event_generator.delay_seconds, -0.1)
        self.assertEqual(peak_frame, 27)
        self.assertEqual(30 - peak_frame, 3)
        self.assertAlmostEqual(target[peak_frame, 0].item(), 1.0)

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
