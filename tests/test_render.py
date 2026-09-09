from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import mido
import numpy as np
from omegaconf import OmegaConf

import airpiano.render.sonify as sonify_stage
import airpiano.render.visualize as visualize_stage
from airpiano.render.artifact import load_artifact
from airpiano.render.sonify import sonify_artifact, sonify_ground_truth
from airpiano.render.visualize import visualize_artifact


def _write_artifact(path: Path, **updates) -> None:
    probability = np.zeros((3, 96), dtype=np.float32)
    probability[0, 0] = 0.8
    probability[1, 0] = 0.8  # exact tie: frame zero must win
    probability[2, 0] = 1.0  # padded frame must be ignored
    probability[1, 1] = 0.19
    target = np.zeros_like(probability)
    target[0, 0] = 1.0
    frame_mask = np.asarray([True, True, False], dtype=np.bool_)
    chord_mask = np.zeros(96, dtype=np.bool_)
    chord_mask[:2] = True
    chord_pitches = np.zeros((96, 128), dtype=np.bool_)
    chord_pitches[0, [60, 64]] = True
    chord_pitches[1, 67] = True
    arrays = {
        "probability": probability,
        "target": target,
        "frame_mask": frame_mask,
        "chord_mask": chord_mask,
        "chord_pitches": chord_pitches,
        "fps": np.asarray(30.0, dtype=np.float64),
        "record_id": np.asarray("subject/session.v1/take.03"),
        "frame_start": np.asarray(0, dtype=np.int64),
        "schema_version": np.asarray("airpiano-inference-v1"),
    }
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


class RenderTest(unittest.TestCase):
    def test_config_handlers_own_argument_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "project"
            artifact_root = project / "inference" / "test"
            artifact_root.mkdir(parents=True)
            artifacts = [artifact_root / "a.npz", artifact_root / "take.v2.npz"]
            for artifact in artifacts:
                artifact.touch()
            nested = artifact_root / "nested"
            nested.mkdir()
            nested_artifact = nested / "inside.npz"
            nested_artifact.touch()
            midi_root = root / "midi"
            config = OmegaConf.create(
                {
                    "output": {"dir": str(project)},
                    "visualize": {
                        "root": str(artifact_root),
                    },
                    "sonify": {
                        "root": str(artifact_root),
                        "midi_root": str(midi_root),
                        "threshold": 0.35,
                        "duration_seconds": 0.75,
                        "velocity": 72,
                    },
                }
            )
            printed = io.StringIO()
            with (
                patch.object(
                    visualize_stage,
                    "visualize_artifact",
                    side_effect=lambda _, output: Path(output).resolve(),
                ) as draw,
                patch.object(
                    sonify_stage,
                    "sonify_artifact",
                    side_effect=lambda _, output, **__: Path(output).resolve(),
                ) as decode,
                patch.object(
                    sonify_stage,
                    "load_artifact",
                    side_effect=[
                        SimpleNamespace(record_id="subject/a"),
                        SimpleNamespace(record_id="subject/inside"),
                        SimpleNamespace(record_id="subject/take.v2"),
                    ],
                ),
                patch.object(
                    sonify_stage,
                    "sonify_ground_truth",
                    side_effect=lambda _, __, output: Path(output).resolve(),
                ) as ground_truth_decode,
                redirect_stdout(printed),
            ):
                visualize_stage.visualize(config)
                sonify_stage.sonify(config)

            self.assertEqual(
                draw.call_args_list,
                [
                    call(artifacts[0], artifact_root / "a.png"),
                    call(nested_artifact, nested / "inside.png"),
                    call(
                        artifacts[1],
                        artifact_root / "take.v2.png",
                    ),
                ],
            )
            self.assertEqual(
                [item.args[:2] for item in decode.call_args_list],
                [
                    (
                        artifacts[0],
                        artifact_root / "a.mid",
                    ),
                    (
                        nested_artifact,
                        nested / "inside.mid",
                    ),
                    (
                        artifacts[1],
                        artifact_root / "take.v2.mid",
                    ),
                ],
            )
            self.assertTrue(
                all(item.kwargs == {
                    "threshold": 0.35,
                    "duration_seconds": 0.75,
                    "velocity": 72,
                } for item in decode.call_args_list)
            )
            self.assertEqual(
                ground_truth_decode.call_args_list,
                [
                    call(
                        artifacts[0],
                        midi_root / "subject" / "a.mid",
                        artifact_root / "a.gt.mid",
                    ),
                    call(
                        nested_artifact,
                        midi_root / "subject" / "inside.mid",
                        nested / "inside.gt.mid",
                    ),
                    call(
                        artifacts[1],
                        midi_root / "subject" / "take.v2.mid",
                        artifact_root / "take.v2.gt.mid",
                    ),
                ],
            )
            self.assertIn("inside.mid", printed.getvalue())

    def test_config_handlers_preserve_absolute_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_root = root / "absolute-input"
            artifact_root.mkdir()
            artifact = artifact_root / "take.v2.npz"
            artifact.touch()
            midi_root = root / "absolute-midi"
            source_midi = midi_root / "nested" / "take.v2.mid"
            config = OmegaConf.create(
                {
                    "output": {"dir": str(root / "unused")},
                    "visualize": {
                        "root": str(artifact_root),
                    },
                    "sonify": {
                        "root": str(artifact_root),
                        "midi_root": str(midi_root),
                        "threshold": 0.2,
                        "duration_seconds": 1.0,
                        "velocity": 64,
                    },
                }
            )
            with (
                patch.object(
                    visualize_stage,
                    "visualize_artifact",
                    return_value=(artifact_root / "take.v2.png").resolve(),
                ) as draw,
                patch.object(
                    sonify_stage,
                    "sonify_artifact",
                    return_value=(artifact_root / "take.v2.mid").resolve(),
                ) as decode,
                patch.object(
                    sonify_stage,
                    "load_artifact",
                    return_value=SimpleNamespace(record_id="nested/take.v2"),
                ),
                patch.object(
                    sonify_stage,
                    "sonify_ground_truth",
                    return_value=(artifact_root / "take.v2.gt.mid").resolve(),
                ) as ground_truth_decode,
            ):
                visualize_stage.visualize(config)
                sonify_stage.sonify(config)

            draw.assert_called_once_with(artifact, artifact_root / "take.v2.png")
            decode.assert_called_once_with(
                artifact,
                artifact_root / "take.v2.mid",
                threshold=0.2,
                duration_seconds=1.0,
                velocity=64,
            )
            ground_truth_decode.assert_called_once_with(
                artifact,
                source_midi.resolve(),
                artifact_root / "take.v2.gt.mid",
            )

    def test_visualize_artifact_draws_two_masked_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "result.npz"
            output = root / "plots" / "result.png"
            _write_artifact(artifact)

            loaded = load_artifact(artifact)
            self.assertEqual(loaded.probability.shape, (3, 96))
            self.assertEqual(int(loaded.support.sum()), 4)
            self.assertEqual(loaded.record_id, "subject/session.v1/take.03")
            self.assertEqual(loaded.frame_start, 0)
            self.assertEqual(visualize_artifact(artifact, output), output.resolve())
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_sonify_uses_valid_earliest_argmax_and_all_chord_pitches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "result.npz"
            output = root / "audio" / "prediction.mid"
            _write_artifact(artifact)

            self.assertEqual(
                sonify_artifact(artifact, output, threshold=0.2), output.resolve()
            )
            midi = mido.MidiFile(output)
            absolute_tick = 0
            note_ons = []
            note_offs = []
            for message in midi.tracks[0]:
                absolute_tick += message.time
                if message.type == "note_on" and message.velocity > 0:
                    note_ons.append((absolute_tick, message.note, message.velocity))
                elif message.type == "note_off":
                    note_offs.append((absolute_tick, message.note))

            self.assertEqual(note_ons, [(0, 60, 64), (0, 64, 64)])
            self.assertEqual(note_offs, [(960, 60), (960, 64)])

    def test_ground_truth_uses_real_slice_and_preserves_note_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "result.npz"
            source = root / "source.mid"
            output = root / "audio" / "ground-truth.mid"
            _write_artifact(
                artifact,
                fps=np.asarray(10.0, dtype=np.float64),
                frame_start=np.asarray(10, dtype=np.int64),
            )

            events = [
                (864, "note_on", 60, 40),   # 0.90 s: already active, omit
                (960, "note_on", 61, 51),   # 1.00 s: include
                (1008, "note_off", 61, 0),  # 1.05 s
                (1056, "note_off", 60, 0),  # 1.10 s
                (1056, "note_on", 62, 62),  # 1.10 s: include, clip
                (1152, "note_on", 63, 63),  # 1.20 s: end boundary, omit
                (1200, "note_on", 64, 64),  # padded-tail time, omit
                (1248, "note_off", 63, 0),
                (1296, "note_off", 64, 0),
                (1344, "note_off", 62, 0),
            ]
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            previous_tick = 0
            for tick, message_type, pitch, velocity in events:
                track.append(
                    mido.Message(
                        message_type,
                        note=pitch,
                        velocity=velocity,
                        time=tick - previous_tick,
                    )
                )
                previous_tick = tick
            midi.save(source)

            self.assertEqual(
                sonify_ground_truth(artifact, source, output), output.resolve()
            )
            rendered = mido.MidiFile(output)
            absolute_tick = 0
            note_ons = []
            note_offs = []
            for message in rendered.tracks[0]:
                absolute_tick += message.time
                if message.type == "note_on" and message.velocity > 0:
                    note_ons.append((absolute_tick, message.note, message.velocity))
                elif message.type == "note_off":
                    note_offs.append((absolute_tick, message.note))

            self.assertEqual(note_ons, [(0, 61, 51), (96, 62, 62)])
            self.assertEqual(note_offs, [(48, 61), (192, 62)])


if __name__ == "__main__":
    unittest.main()
