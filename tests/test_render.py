from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
from omegaconf import OmegaConf

import airpiano.render.sonify as sonify_stage
import airpiano.render.visualize as visualize_stage
from airpiano.render.artifact import load_artifact
from airpiano.render.sonify import sonify_artifact
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
        "schema_version": np.asarray("airpiano-inference-v1"),
    }
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


class RenderTest(unittest.TestCase):
    def test_config_handlers_own_argument_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "project"
            artifact = project / "inference" / "result.npz"
            heatmap = project / "visualizations" / "result.png"
            midi = project / "sonifications" / "result.mid"
            config = OmegaConf.create(
                {
                    "output": {"dir": str(project)},
                    "visualize": {
                        "input": "inference/result.npz",
                        "output": "visualizations/result.png",
                    },
                    "sonify": {
                        "input": "inference/result.npz",
                        "output": "sonifications/result.mid",
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
                    return_value=heatmap.resolve(),
                ) as draw,
                patch.object(
                    sonify_stage,
                    "sonify_artifact",
                    return_value=midi.resolve(),
                ) as decode,
                redirect_stdout(printed),
            ):
                visualize_stage.visualize(config)
                sonify_stage.sonify(config)

            draw.assert_called_once_with(artifact.resolve(), heatmap.resolve())
            decode.assert_called_once_with(
                artifact.resolve(),
                midi.resolve(),
                threshold=0.35,
                duration_seconds=0.75,
                velocity=72,
            )
            self.assertIn(f"wrote {heatmap.resolve()}", printed.getvalue())
            self.assertIn(f"wrote {midi.resolve()}", printed.getvalue())

    def test_config_handlers_preserve_absolute_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "absolute.npz"
            heatmap = root / "absolute.png"
            midi = root / "absolute.mid"
            config = OmegaConf.create(
                {
                    "output": {"dir": str(root / "unused")},
                    "visualize": {"input": str(artifact), "output": str(heatmap)},
                    "sonify": {
                        "input": str(artifact),
                        "output": str(midi),
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
                    return_value=heatmap.resolve(),
                ) as draw,
                patch.object(
                    sonify_stage,
                    "sonify_artifact",
                    return_value=midi.resolve(),
                ) as decode,
            ):
                visualize_stage.visualize(config)
                sonify_stage.sonify(config)

            draw.assert_called_once_with(artifact.resolve(), heatmap.resolve())
            decode.assert_called_once_with(
                artifact.resolve(),
                midi.resolve(),
                threshold=0.2,
                duration_seconds=1.0,
                velocity=64,
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


if __name__ == "__main__":
    unittest.main()
