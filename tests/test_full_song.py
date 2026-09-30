"""Tests for frame-causal full-song inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import airpiano.full_song as full_song_stage
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
from airpiano.render.artifact import load_artifact


def _chord(
    chord_id: int,
    onset: float,
    pitch: int,
    velocity: int = 80,
    duration: float = 0.1,
    quantized_dur: torch.Tensor | None = None,
) -> Chord:
    note = Note(chord_id, onset, onset + 0.1, pitch, velocity)
    event = NoteEvent(note, False)
    return Chord(
        id=chord_id,
        events=(event,),
        onset_notes=(note,),
        offset_notes=(),
        time=onset,
        duration=duration,
        quantized_dur=(
            quantize_duration_seconds(torch.tensor(duration))
            if quantized_dur is None
            else quantized_dur
        ),
    )


def _song(chord_count: int, frame_count: int = 6) -> Song:
    chords = tuple(
        _chord(index, index / 10.0, 60 + index)
        for index in range(chord_count)
    )
    hands = torch.arange(frame_count, dtype=torch.float32)[:, None, None, None]
    hands = hands.expand(-1, 2, 21, 3).clone()
    return Song(
        record_id="subject/performance.v1",
        notes=tuple(note for chord in chords for note in chord.onset_notes),
        chords=chords,
        hands=hands,
        presence=torch.ones((frame_count, 2), dtype=torch.bool),
        fps=10.0,
    )


def _targets() -> TargetGenerator:
    return TargetGenerator(ChordOnsetEvents(), TriangularSoftener(0.1))


class _StreamingModel(nn.Module):
    causal_frame_attention = True

    def __init__(self, include_chord_duration: bool = False) -> None:
        super().__init__()
        self.include_chord_duration = include_chord_duration
        self.calls: list[tuple[list[int], list[int]]] = []

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        del hand_presence, frame_mask, chord_mask
        frame_ids = hands[0, :, 0, 0, 0].to(torch.long)
        chord_pitches = torch.argmax(chords[0], dim=-1)
        self.calls.append((frame_ids.tolist(), chord_pitches.tolist()))
        logits = torch.full(
            (1, len(frame_ids), len(chord_pitches)),
            -2.0,
            device=hands.device,
        )
        logits[:, :, 0] = 2.0
        if len(chord_pitches) > 1:
            logits[:, :, 1] = frame_ids.to(logits.dtype)
        if len(chord_pitches) > 2:
            logits[:, :, 2] = 2.0
        return logits


class FullSongControllerTest(unittest.TestCase):
    def test_any_trailing_peak_shifts_and_commits_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "song.npz"
            model = _StreamingModel(include_chord_duration=True)
            loss, cell_count, shifts = full_song_stage.infer_full_song(
                model,
                _song(6),
                FutureChordSlicer(
                    mode="chord",
                    fps=10.0,
                    segment_frames=4,
                    stride_frames=2,
                ),
                _targets(),
                torch.device("cpu"),
                output,
                bf16=False,
                alignment_threshold=0.6,
                advance_after_chords=2,
            )

            self.assertEqual(shifts, 2)
            self.assertEqual(cell_count, 36)
            self.assertGreaterEqual(loss, 0.0)
            self.assertEqual(
                model.calls,
                [
                    ([0], [60, 61, 62, 63]),
                    ([0, 1], [61, 62, 63, 64]),
                    ([1, 2], [62, 63, 64, 65]),
                    ([1, 2, 3], [62, 63, 64, 65]),
                    ([1, 2, 3, 4], [62, 63, 64, 65]),
                    ([1, 2, 3, 4, 5], [62, 63, 64, 65]),
                ],
            )

            with np.load(output, allow_pickle=False) as artifact:
                probability = artifact["probability"]
                self.assertEqual(
                    str(artifact["schema_version"]),
                    "airpiano-inference-v1",
                )
                self.assertEqual(probability.shape, (6, 6))
                self.assertTrue(bool(artifact["frame_mask"].all()))
                self.assertTrue(bool(artifact["chord_mask"].all()))
                self.assertEqual(artifact["chord_ids"].tolist(), list(range(6)))
                self.assertEqual(artifact["chord_pitches"].shape, (6, 128))
                self.assertTrue(np.all(probability[0, :4] > 0.0))
                self.assertAlmostEqual(float(probability[0, 1]), 0.5)
                self.assertTrue(np.all(probability[0, 4:] == 0.0))
                self.assertEqual(float(probability[1, 0]), 0.0)
                self.assertTrue(np.all(probability[1, 1:5] > 0.0))
                self.assertTrue(np.all(probability[2:, :2] == 0.0))
                self.assertEqual(int(artifact["frame_start"]), 0)
                self.assertEqual(
                    str(artifact["record_id"]), "subject/performance.v1"
                )
                self.assertAlmostEqual(float(artifact["loss"]), loss)

            loaded = load_artifact(output)
            self.assertEqual(loaded.probability.shape, (6, 6))
            self.assertEqual(int(loaded.support.sum()), 36)

    def test_short_score_keeps_one_window_for_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _StreamingModel()
            output = Path(directory) / "short.npz"
            _, _, shifts = full_song_stage.infer_full_song(
                model,
                _song(3, frame_count=4),
                FutureChordSlicer(
                    mode="chord",
                    fps=10.0,
                    segment_frames=4,
                    stride_frames=2,
                ),
                _targets(),
                torch.device("cpu"),
                output,
                bf16=False,
                alignment_threshold=0.5,
                advance_after_chords=2,
            )

            self.assertEqual(shifts, 0)
            self.assertEqual(
                [frames for frames, _ in model.calls],
                [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
            )
            with np.load(output, allow_pickle=False) as artifact:
                self.assertTrue(bool((artifact["probability"] > 0.0).all()))

    def test_cross_entropy_keeps_pre_onset_frames_and_uses_softmax(self) -> None:
        base = _song(3)
        chords = tuple(
            _chord(
                chord.id,
                onset,
                chord.onset_notes[0].pitch,
                chord.onset_notes[0].velocity,
                chord.duration,
                chord.quantized_dur,
            )
            for chord, onset in zip(
                base.chords, (0.15, 0.35, 0.45), strict=True
            )
        )
        song = Song(
            record_id=base.record_id,
            notes=tuple(
                note for chord in chords for note in chord.onset_notes
            ),
            chords=chords,
            hands=base.hands,
            presence=base.presence,
            fps=base.fps,
        )
        slicer = FutureChordSlicer(
            mode="chord",
            fps=10.0,
            segment_frames=4,
            stride_frames=2,
            crop_before_first_chord=True,
        )
        target_generator = TargetGenerator(
            ChordOnsetEvents(),
            TriangularSoftener(0.1),
            target_mode="last_chord",
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "softmax.npz"
            model = _StreamingModel()
            loss, loss_weight, shifts = full_song_stage.infer_full_song(
                model,
                song,
                slicer,
                target_generator,
                torch.device("cpu"),
                output,
                bf16=False,
                alignment_threshold=2.0,
                advance_after_chords=2,
                alignment_loss="cross_entropy",
            )

            self.assertEqual(loss_weight, 4)
            self.assertEqual(shifts, 0)
            self.assertGreaterEqual(loss, 0.0)
            self.assertEqual(model.calls[0][0], [0])
            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(int(artifact["frame_start"]), 0)
                self.assertEqual(artifact["probability"].shape, (6, 3))
                np.testing.assert_allclose(
                    artifact["probability"].sum(axis=-1), 1.0, atol=1e-6
                )
                np.testing.assert_array_equal(
                    artifact["target"].sum(axis=-1),
                    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                )

    def test_rejects_frame_mode_and_noncausal_checkpoint(self) -> None:
        song = _song(3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.npz"
            with self.assertRaisesRegex(ValueError, "slicer.mode=chord"):
                full_song_stage.infer_full_song(
                    _StreamingModel(),
                    song,
                    FutureChordSlicer(mode="frame", segment_frames=4),
                    _targets(),
                    torch.device("cpu"),
                    output,
                    bf16=False,
                    alignment_threshold=0.2,
                    advance_after_chords=2,
                )

            model = _StreamingModel()
            model.causal_frame_attention = False
            with self.assertRaisesRegex(ValueError, "causal_frame_attention"):
                full_song_stage.infer_full_song(
                    model,
                    song,
                    FutureChordSlicer(mode="chord", segment_frames=4),
                    _targets(),
                    torch.device("cpu"),
                    output,
                    bf16=False,
                    alignment_threshold=0.2,
                    advance_after_chords=2,
                )


class FullSongStageTest(unittest.TestCase):
    def test_reuses_one_model_and_writes_weighted_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            components = {
                "slicer": FutureChordSlicer(
                    mode="chord", segment_frames=4, stride_frames=2
                ),
                "target_generator": _targets(),
            }
            songs = [_song(3), _song(4)]
            songs[1] = Song(
                record_id="second.take",
                notes=songs[1].notes,
                chords=songs[1].chords,
                hands=songs[1].hands,
                presence=songs[1].presence,
                fps=songs[1].fps,
            )
            model = SimpleNamespace(
                causal_frame_attention=True,
                include_chord_duration=False,
            )
            config = OmegaConf.create(
                {
                    "device": "cpu",
                    "output": {"dir": str(project)},
                    "dataset": {},
                    "preprocess": {},
                    "train": {"bf16": False},
                    "full_song": {
                        "split_file": str(project / "songs.txt"),
                        "checkpoint": str(
                            project / "checkpoints" / "last.pt"
                        ),
                        "output_dir": str(project / "full_song"),
                        "overwrite": True,
                        "alignment_threshold": 0.2,
                        "advance_after_chords": 2,
                    },
                }
            )

            with (
                patch.object(
                    full_song_stage.data,
                    "build_data_components",
                    return_value=components,
                ),
                patch.object(
                    full_song_stage,
                    "load_model_checkpoint",
                    return_value=model,
                ) as load_model,
                patch.object(
                    full_song_stage,
                    "iter_split_songs",
                    return_value=iter(
                        [
                            (1, 2, songs[0], components),
                            (2, 2, songs[1], components),
                        ]
                    ),
                ) as load_songs,
                patch.object(
                    full_song_stage,
                    "infer_full_song",
                    side_effect=[(1.0, 10, 1), (3.0, 30, 2)],
                ) as infer,
            ):
                full_song_stage.full_song(config)

            load_model.assert_called_once_with(
                (project / "checkpoints" / "last.pt").resolve(),
                torch.device("cpu"),
            )
            load_songs.assert_called_once_with(
                config,
                str(project / "songs.txt"),
                overwrite=True,
                label="full-song",
                components=components,
            )
            output_dir = project / "full_song"
            self.assertEqual(
                [item.args[1].record_id for item in infer.call_args_list],
                ["subject/performance.v1", "second.take"],
            )
            self.assertEqual(
                [item.args[5] for item in infer.call_args_list],
                [
                    output_dir / "subject/performance.v1.npz",
                    output_dir / "second.take.npz",
                ],
            )
            self.assertEqual(
                json.loads((output_dir / "full_song_loss.json").read_text()),
                {"full_song_loss": 2.5},
            )


if __name__ == "__main__":
    unittest.main()
