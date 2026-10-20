"""Focused tests for the full-score online DP and MIDI release clocks."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mido
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import airpiano.full_song_real_time as real_time_stage

from airpiano.data import (
    Chord,
    ChordOnsetEvents,
    FutureChordSlicer,
    Note,
    NoteEvent,
    Song,
    TargetGenerator,
    TriangularSoftener,
)
from airpiano.data.midi import quantize_duration_seconds
from airpiano.full_song_real_time import (
    Event,
    RealTimeController,
    infer_full_song_real_time,
)
from airpiano.render.artifact import load_artifact


def _chords(count: int = 60, *, repeated_pitches: bool = False) -> tuple[Chord, ...]:
    chords = []
    for index in range(count):
        onset = index / 10.0
        note = Note(
            index,
            onset,
            onset + 0.1,
            60 if repeated_pitches else 24 + index,
            80,
        )
        chords.append(
            Chord(
                id=index,
                events=(NoteEvent(note, False),),
                onset_notes=(note,),
                offset_notes=(),
                time=onset,
                duration=0.1,
                quantized_dur=quantize_duration_seconds(torch.tensor(0.1)),
                score_subdivision=index * 24,
            )
        )
    return tuple(chords)


def _song(chord_count: int = 60, frame_count: int = 4) -> Song:
    chords = _chords(chord_count)
    hands = torch.arange(frame_count, dtype=torch.float32)[:, None, None, None]
    hands = hands.expand(-1, 2, 21, 3).clone()
    presence = torch.ones((frame_count, 2), dtype=torch.bool)
    presence[::2, 1] = False
    return Song(
        record_id="subject/performance",
        notes=tuple(note for chord in chords for note in chord.onset_notes),
        chords=chords,
        hands=hands,
        presence=presence,
        fps=30.0,
    )


class _ScheduledModel(nn.Module):
    causal_frame_attention = True
    include_chord_duration = False

    def __init__(self, events: dict[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.events = events
        self.calls: list[tuple[list[int], list[int], torch.Tensor]] = []

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        frame_ids = hands[0, :, 0, 0, 0].to(torch.long).tolist()
        chord_ids = (chords[0, :, :128].argmax(dim=-1) - 23).tolist()
        self.calls.append((frame_ids, chord_ids, hand_presence[0].cpu().clone()))
        assert bool(frame_mask.all())
        assert bool(chord_mask.all())
        logits = torch.full(
            (1, len(frame_ids), len(chord_ids)), -2.0, device=hands.device
        )
        # Earlier history rows deliberately disagree with the current row.
        logits[:, :-1, :] = 2.0
        for chord_id in self.events.get(frame_ids[-1], ()):
            logits[0, -1, chord_ids.index(chord_id)] = 2.0
        return logits


class RealTimeControllerTest(unittest.TestCase):
    def test_initial_state_uses_occurrences_even_for_identical_pitches(self) -> None:
        controller = RealTimeController(_chords(repeated_pitches=True))

        self.assertEqual([chord.id for chord in controller.chords[1:]], list(range(1, 61)))
        self.assertEqual(controller.chords[1].pitches, controller.chords[2].pitches)
        self.assertTrue(all(not chord.triggered and not chord.released for chord in controller.chords[1:]))
        self.assertTrue(all(chord.anchor_time is None for chord in controller.chords[1:]))
        np.testing.assert_array_equal(controller.prev, np.zeros(61))
        np.testing.assert_array_equal(controller.cur, np.zeros(61))
        self.assertEqual(controller.best_score, 0)
        self.assertEqual(controller.pending_releases, {})
        self.assertEqual(controller.chord_window_start, 1)
        self.assertEqual(controller.video_history_start, 0)
        self.assertEqual(controller.release_records, [])

    def test_threshold_latching_equality_retrigger_and_unobserved_flags(self) -> None:
        controller = RealTimeController(_chords())
        controller.chords[60].triggered = True
        scores = np.full(50, 0.1)
        scores[1] = 0.8
        scores[2] = 0.5

        events = controller.collect_events(scores, window_start=1, frame_index=30)
        self.assertEqual([event.id for event in events], [2])
        self.assertAlmostEqual(events[0].time, 1.1)
        self.assertAlmostEqual(events[0].score, 0.8)
        self.assertFalse(controller.chords[3].triggered)
        self.assertTrue(controller.chords[60].triggered)
        self.assertEqual(controller.collect_events(scores, window_start=1, frame_index=31), [])

        scores[1] = 0.5
        self.assertEqual(controller.collect_events(scores, window_start=1, frame_index=32), [])
        self.assertTrue(controller.chords[2].triggered)
        scores[1] = 0.49
        controller.collect_events(scores, window_start=1, frame_index=33)
        self.assertFalse(controller.chords[2].triggered)
        scores[1] = 0.51
        self.assertEqual(
            [event.id for event in controller.collect_events(scores, window_start=1, frame_index=34)],
            [2],
        )

        controller.collect_events(np.zeros(50), window_start=11, frame_index=35)
        self.assertTrue(controller.chords[2].triggered)
        self.assertFalse(controller.chords[60].triggered)

    def test_dp_prefix_skip_ignored_events_and_strict_historical_best(self) -> None:
        controller = RealTimeController(_chords())
        self.assertTrue(controller.process_event(Event(10, 0.1, 0.8), frame_index=0))
        np.testing.assert_array_equal(controller.prev[:12], [0] * 10 + [1, 0])
        self.assertEqual(controller.best_score, 1)
        self.assertEqual(controller.prev[-1], 0)
        self.assertFalse(controller.process_event(Event(10, 0.2, 0.9), frame_index=3))
        self.assertEqual(controller.chords[10].anchor_time, 0.1)
        self.assertEqual(controller.chord_window_start, 2)

        self.assertTrue(controller.process_event(Event(11, 0.3, 0.7), frame_index=6))
        np.testing.assert_array_equal(controller.prev[9:14], [0, 1, 2, 1, 0])
        self.assertEqual(controller.best_score, 2)
        self.assertFalse(controller.process_event(Event(13, 0.4, 0.9), frame_index=9))
        self.assertIsNone(controller.chords[13].anchor_time)
        self.assertEqual(controller.best_score, 2)
        self.assertEqual(controller.prev[13], 2)
        self.assertTrue(controller.process_event(Event(12, 0.5, 0.8), frame_index=12))
        self.assertEqual(controller.best_score, 3)
        self.assertEqual(controller.chords[12].anchor_time, 0.5)
        self.assertEqual(controller.prev[-1], 0)

    def test_collected_events_retain_absolute_ids_after_large_window_moves(self) -> None:
        controller = RealTimeController(_chords(90))
        scores = np.zeros(50)
        scores[39:41] = 0.9
        events = controller.collect_events(scores, window_start=1, frame_index=0)
        self.assertEqual([event.id for event in events], [40, 41])

        self.assertTrue(controller.process_event(events[0], frame_index=0))
        self.assertEqual(controller.chord_window_start, 15)
        self.assertTrue(controller.process_event(events[1], frame_index=0))
        self.assertEqual(controller.chord_window_start, 16)
        self.assertEqual(controller.chords[40].anchor_time, 0.1)
        self.assertEqual(controller.chords[41].anchor_time, 0.1)
        self.assertEqual(controller.best_score, 2)
        self.assertEqual(len(controller.prev), 91)

    def test_window_is_clamped_at_both_score_ends(self) -> None:
        controller = RealTimeController(_chords())
        controller.process_event(Event(60, 0.1, 0.8), frame_index=0)
        self.assertEqual(controller.chord_window_start, 11)
        controller.process_event(Event(60, 0.2, 0.8), frame_index=3)
        self.assertEqual(controller.chord_window_start, 11)

        controller = RealTimeController(_chords())
        controller.process_event(Event(1, 0.1, 0.8), frame_index=0)
        self.assertEqual(controller.chord_window_start, 1)

    def test_anchor_estimates_require_two_anchors_except_at_exact_anchor(self) -> None:
        controller = RealTimeController(_chords())
        self.assertTrue(math.isinf(controller.estimate_time(0)))
        controller.chords[10].anchor_time = 3.0
        self.assertEqual(controller.estimate_time(9 * 24), 3.0)
        self.assertTrue(math.isinf(controller.estimate_time(8 * 24)))
        self.assertTrue(math.isinf(controller.estimate_time(10 * 24)))

        controller.chords[30].anchor_time = 7.0
        self.assertEqual(controller.estimate_time(29 * 24), 7.0)
        self.assertAlmostEqual(controller.estimate_time(19 * 24), 5.0)
        self.assertAlmostEqual(controller.estimate_time(0), 1.2)
        self.assertAlmostEqual(controller.estimate_time(40 * 24), 9.2)

    def test_equal_or_decreasing_anchor_times_are_not_corrected(self) -> None:
        controller = RealTimeController(_chords())
        controller.chords[10].anchor_time = 7.0
        controller.chords[30].anchor_time = 7.0
        self.assertEqual(controller.estimate_time(0), 7.0)
        self.assertEqual(controller.estimate_time(40 * 24), 7.0)

        controller.chords[30].anchor_time = 3.0
        self.assertEqual(controller.estimate_time(19 * 24), 5.0)
        self.assertAlmostEqual(controller.estimate_time(0), 8.8)
        self.assertAlmostEqual(controller.estimate_time(40 * 24), 0.8)

    def test_successful_single_anchor_releases_at_first_available_result_time(self) -> None:
        controller = RealTimeController(_chords())
        scores = np.zeros(50)
        scores[9] = 0.8
        event, = controller.collect_events(scores, window_start=1, frame_index=30)
        self.assertTrue(controller.process_event(event, frame_index=30))
        self.assertEqual(controller.pending_releases, {10: 1.1})
        self.assertEqual(controller.release_due(frame_index=30), [])
        self.assertEqual(controller.release_due(frame_index=31), [])

        released = controller.release_due(frame_index=32)
        self.assertEqual([record["chord_id"] for record in released], [10])
        self.assertAlmostEqual(released[0]["predicted_onset"], 1.1)
        self.assertAlmostEqual(released[0]["frame_time"], 32 / 30)
        self.assertAlmostEqual(released[0]["release_time"], 32 / 30 + 0.05)
        self.assertEqual(released[0]["duration"], 1.0)
        self.assertFalse(controller.chords[1].released)
        self.assertTrue(controller.chords[10].released)
        self.assertEqual(controller.pending_releases, {})
        self.assertEqual(controller.release_due(frame_index=100), [])

    def test_failed_trigger_is_scheduled_without_interpolating_missing_chords(self) -> None:
        controller = RealTimeController(_chords())
        scores = np.zeros(50)
        scores[[9, 11]] = 0.8
        events = controller.collect_events(scores, window_start=1, frame_index=0)
        self.assertEqual(
            [controller.process_event(event, frame_index=0) for event in events],
            [True, False],
        )
        self.assertTrue(controller.chords[12].triggered)
        self.assertIsNone(controller.chords[12].anchor_time)
        self.assertEqual(controller.pending_releases, {10: 0.1, 12: 0.1})
        released = controller.release_due(frame_index=30)
        self.assertEqual([record["chord_id"] for record in released], [10, 12])
        self.assertFalse(controller.chords[11].released)
        self.assertTrue(controller.chords[12].released)

    def test_repeated_success_preserves_pending_target_and_releases_only_once(self) -> None:
        controller = RealTimeController(_chords())
        self.assertTrue(controller.process_event(Event(10, 0.1, 0.8), frame_index=0))
        self.assertEqual(controller.release_due(frame_index=1), [])
        self.assertEqual(
            [
                controller.process_event(Event(chord_id, 2 / 30 + 0.1, 0.8), 2)
                for chord_id in (8, 9, 10)
            ],
            [False, True, True],
        )
        self.assertEqual(controller.pending_releases[10], 0.1)
        self.assertAlmostEqual(controller.chords[10].anchor_time, 2 / 30 + 0.1)
        released = controller.release_due(frame_index=2)
        self.assertEqual([record["chord_id"] for record in released], [10])
        self.assertEqual(released[0]["predicted_onset"], 0.1)
        frozen = released[0].copy()

        self.assertFalse(controller.process_event(Event(8, 0.2, 0.8), 3))
        self.assertEqual(controller.pending_releases[8], 2 / 30 + 0.1)
        self.assertFalse(controller.process_event(Event(10, 0.2, 0.8), 3))
        self.assertNotIn(10, controller.pending_releases)

        successes = [
            controller.process_event(Event(chord_id, 4 / 30 + 0.1, 0.8), 4)
            for chord_id in range(1, 11)
        ]
        self.assertTrue(successes[-1])
        self.assertTrue(successes[7])
        self.assertAlmostEqual(controller.chords[8].anchor_time, 4 / 30 + 0.1)
        self.assertEqual(controller.pending_releases[8], 2 / 30 + 0.1)
        self.assertNotIn(10, controller.pending_releases)
        controller.release_due(frame_index=10)
        self.assertEqual(
            [record for record in controller.release_records if record["chord_id"] == 10],
            [frozen],
        )
        rejected_first = [
            record for record in controller.release_records if record["chord_id"] == 8
        ]
        self.assertEqual(len(rejected_first), 1)
        self.assertEqual(rejected_first[0]["predicted_onset"], 2 / 30 + 0.1)
        self.assertFalse(
            controller.process_event(Event(8, 11 / 30 + 0.1, 0.8), 11)
        )
        self.assertNotIn(8, controller.pending_releases)
        self.assertEqual(controller.release_due(frame_index=20), [])

    def test_nonpositive_segments_still_select_video_history(self) -> None:
        for first_time, expected_history in ((1.0, 30), (2.0, 45)):
            with self.subTest(first_time=first_time):
                controller = RealTimeController(_chords(100))
                controller.chords[1].anchor_time = first_time
                controller.process_event(Event(51, 1.0, 0.8), frame_index=60)
                self.assertEqual(controller.chord_window_start, 26)
                self.assertEqual(controller.video_history_start, expected_history)
                self.assertAlmostEqual(
                    controller.estimate_time(25 * 24), expected_history / 30
                )

    def test_history_resets_for_infinity_and_can_move_backward(self) -> None:
        controller = RealTimeController(_chords(100))
        controller.video_history_start = 250
        controller.process_event(Event(51, 6.0, 0.8), frame_index=300)
        self.assertEqual(controller.video_history_start, 0)

        controller.chords[1].anchor_time = 1.0
        controller.process_event(Event(52, 6.1, 0.8), frame_index=301)
        self.assertEqual(controller.video_history_start, 108)

        # Revisions to stored anchors must also revise the retained history.
        controller.chords[1].anchor_time = 0.0
        controller.process_event(Event(53, 6.2, 0.8), frame_index=302)
        self.assertEqual(controller.video_history_start, 97)

    def test_history_clamps_to_received_frames_and_to_zero(self) -> None:
        controller = RealTimeController(_chords(100))
        controller.chords[1].anchor_time = 10.0
        controller.chords[100].anchor_time = 11.0
        controller.process_event(Event(51, 10.5, 0.8), frame_index=3)
        self.assertEqual(controller.video_history_start, 3)

        controller.chords[1].anchor_time = -30.0
        controller.process_event(Event(52, 10.6, 0.8), frame_index=4)
        self.assertEqual(controller.video_history_start, 0)

    def test_stored_anchors_alone_do_not_schedule_notes(self) -> None:
        controller = RealTimeController(_chords())
        for chord_id, time in ((1, 0.1), (3, 0.3), (5, 0.2), (7, 0.4)):
            controller.chords[chord_id].anchor_time = time
        self.assertEqual(controller.pending_releases, {})
        self.assertEqual(controller.release_due(frame_index=30), [])
        self.assertFalse(any(chord.released for chord in controller.chords[1:]))

    def test_release_batch_uses_frozen_event_times_without_querying_anchors(self) -> None:
        controller = RealTimeController(_chords())
        controller.process_event(Event(10, 0.1, 0.8), frame_index=0)
        controller.process_event(Event(11, 0.2, 0.8), frame_index=3)
        controller.chords[1].anchor_time = 100.0
        controller.chords[2].anchor_time = -100.0
        controller.chords[10].anchor_time = 50.0
        previous_dp = controller.prev.copy()
        with patch.object(controller, "estimate_time", side_effect=AssertionError):
            released = controller.release_due(frame_index=8)
        self.assertEqual([record["chord_id"] for record in released], [10, 11])
        for index, record in enumerate(released):
            self.assertEqual(record["frame_index"], 8)
            self.assertAlmostEqual(record["frame_time"], 8 / 30)
            self.assertAlmostEqual(record["predicted_onset"], (index + 1) / 10)
            self.assertAlmostEqual(record["release_time"], 8 / 30 + 0.05)
            self.assertEqual(record["duration"], 1.0)
        self.assertEqual(controller.pending_releases, {})
        self.assertFalse(controller.chords[12].released)
        np.testing.assert_array_equal(controller.prev, previous_dp)
        self.assertEqual(controller.best_score, 2)


class RealTimeInferenceTest(unittest.TestCase):
    def _infer(self, model, song, output):
        return infer_full_song_real_time(
            model,
            song,
            FutureChordSlicer(mode="chord", fps=30.0, segment_frames=12),
            TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.1)),
            torch.device("cpu"),
            output,
            bf16=False,
        )

    def test_streaming_uses_saved_window_current_row_and_received_history(self) -> None:
        song = _song(chord_count=90)
        model = _ScheduledModel({0: (40, 41), 1: (65,)})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "song.npz"
            controller = self._infer(model, song, output)

            self.assertEqual(len(model.calls), 4)
            self.assertEqual([call[0] for call in model.calls], [[0], [0, 1], [1, 2], [1, 2, 3]])
            self.assertEqual([call[1][0] for call in model.calls], [1, 16, 17, 17])
            for frames, chord_ids, presence in model.calls:
                self.assertEqual(len(chord_ids), 50)
                torch.testing.assert_close(presence, song.presence[frames])
            self.assertEqual(controller.best_score, 2)
            self.assertEqual(
                [record["chord_id"] for record in controller.release_records], [40, 41, 65]
            )
            self.assertTrue(controller.chords[65].released)
            self.assertIsNone(controller.chords[65].anchor_time)

            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(artifact["probability"].shape, (4, 90))
                self.assertEqual(artifact["chord_ids"].tolist(), list(range(1, 91)))
                self.assertEqual(artifact["frame_chord_window_start"].tolist(), [1, 16, 17, 17])
                self.assertEqual(artifact["frame_video_history_start"].tolist(), [0, 0, 1, 1])
                self.assertEqual(artifact["event_chord_id"].tolist(), [40, 41, 65])
                np.testing.assert_allclose(artifact["probability"][0, 39:41], torch.sigmoid(torch.tensor(2.0)).item())
                self.assertGreater(float(artifact["probability"][0].sum()), 1.0)
                self.assertTrue(bool((artifact["probability"][0, 50:] == 0).all()))
                self.assertTrue(bool((artifact["probability"][1, :15] == 0).all()))
                self.assertLess(float(artifact["probability"][2, 39]), 0.5)
                self.assertNotIn("next_unreleased", artifact.files)
                np.testing.assert_allclose(artifact["predicted_onset"][39:41], 0.1)
                np.testing.assert_allclose(
                    artifact["release_time"][39:41], 2 / 30 + 0.05
                )
                self.assertEqual(artifact["release_frame_index"][39:41].tolist(), [2, 2])
                self.assertEqual(artifact["event_success"].tolist(), [True, True, False])
                self.assertAlmostEqual(float(artifact["predicted_onset"][64]), 1 / 30 + 0.1)
                self.assertAlmostEqual(float(artifact["release_time"][64]), 0.15)
                self.assertEqual(int(artifact["release_frame_index"][64]), 3)
                self.assertEqual(
                    np.flatnonzero(np.isfinite(artifact["release_time"])).tolist(),
                    [39, 40, 64],
                )

            loaded = load_artifact(output)
            self.assertEqual(loaded.probability.shape, (4, 90))
            self.assertTrue(output.with_suffix(".mid").is_file())

    def test_finite_end_preserves_one_second_notes_without_flushing_future_onsets(self) -> None:
        model = _ScheduledModel({0: (1,), 1: (2,), 3: (3,)})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "song.npz"
            controller = self._infer(model, _song(), output)

            self.assertEqual([record["chord_id"] for record in controller.release_records], [1, 2])
            self.assertEqual([record["frame_index"] for record in controller.release_records], [2, 3])
            self.assertAlmostEqual(controller.chords[1].anchor_time, 0.1)
            self.assertAlmostEqual(controller.chords[2].anchor_time, 1 / 30 + 0.1)
            self.assertFalse(controller.chords[3].released)
            self.assertEqual(controller.pending_releases, {3: 0.2})
            with np.load(output, allow_pickle=False) as artifact:
                np.testing.assert_allclose(artifact["release_time"][:2], [2 / 30 + 0.05, 0.15])
                self.assertTrue(bool(np.isnan(artifact["release_time"][2:]).all()))
                self.assertTrue(bool((artifact["release_frame_index"][2:] == -1).all()))
                self.assertTrue(bool(np.isnan(artifact["predicted_onset"][2:]).all()))

            midi = mido.MidiFile(output.with_suffix(".mid"))
            self.assertEqual(midi.ticks_per_beat, 480)
            tick = 0
            onsets = {}
            offsets = {}
            tempos = []
            for message in mido.merge_tracks(midi.tracks):
                tick += message.time
                if message.type == "set_tempo":
                    tempos.append(message.tempo)
                elif message.type == "note_on" and message.velocity > 0:
                    onsets[message.note] = tick
                    self.assertEqual(message.velocity, 64)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    offsets[message.note] = tick
            self.assertEqual(tempos, [500000])
            self.assertEqual(onsets, {24: 112, 25: 144})
            self.assertEqual(offsets, {24: 1072, 25: 1104})


class RealTimeStageTest(unittest.TestCase):
    def test_stage_reuses_model_and_preprocessing_for_nested_song_outputs(self) -> None:
        first_song = _song()
        songs = [first_song, replace(first_song, record_id="second.take")]
        components = {
            "midi_loader": SimpleNamespace(
                reader=SimpleNamespace(quantization=True),
                include_offsets=False,
                chord_merger=SimpleNamespace(split_notes=False),
            ),
            "slicer": FutureChordSlicer(mode="chord"),
            "target_generator": TargetGenerator(
                ChordOnsetEvents(), TriangularSoftener(0.1)
            ),
        }
        model = _ScheduledModel({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = OmegaConf.create(
                {
                    "device": "cpu",
                    "alignment_loss": "bce",
                    "train": {"bf16": True},
                    "full_song_real_time": {
                        "split_file": str(root / "songs.txt"),
                        "checkpoint": str(root / "last.pt"),
                        "output_dir": str(root / "inference"),
                        "overwrite": False,
                    },
                    "evaluation": {
                        "root": "${full_song_real_time.output_dir}"
                    },
                }
            )
            with (
                patch.object(
                    real_time_stage.data, "build_data_components",
                    return_value=components,
                ),
                patch.object(
                    real_time_stage, "load_model_checkpoint", return_value=model,
                ) as load_model,
                patch.object(
                    real_time_stage, "iter_split_songs",
                    return_value=iter(
                        (index, 2, song, components)
                        for index, song in enumerate(songs, start=1)
                    ),
                ) as load_songs,
                patch.object(
                    real_time_stage, "infer_full_song_real_time",
                    side_effect=[RealTimeController(song.chords) for song in songs],
                ) as infer,
            ):
                real_time_stage.full_song_real_time(config)

            load_model.assert_called_once_with(
                (root / "last.pt").resolve(), torch.device("cpu")
            )
            load_songs.assert_called_once_with(
                config, str(root / "songs.txt"), overwrite=False,
                label="full-song-real-time", components=components,
            )
            output_dir = root / "inference" / "songs"
            self.assertEqual(Path(config.full_song_real_time.output_dir), output_dir)
            self.assertEqual(Path(config.evaluation.root), output_dir)
            self.assertEqual(infer.call_count, 2)
            for call, song in zip(infer.call_args_list, songs, strict=True):
                self.assertIs(call.args[0], model)
                self.assertIs(call.args[1], song)
                self.assertIs(call.args[2], components["slicer"])
                self.assertIs(call.args[3], components["target_generator"])
                self.assertEqual(call.args[4], torch.device("cpu"))
                self.assertEqual(call.args[5], output_dir / f"{song.record_id}.npz")
                self.assertEqual(call.kwargs, {"bf16": True})


if __name__ == "__main__":
    unittest.main()
