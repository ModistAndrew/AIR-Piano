from __future__ import annotations

import io
import json
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
from airpiano.render.sonify import (
    calculate_onset_deltas,
    plot_onset_delta_summary,
    sonify_artifact,
    sonify_ground_truth,
    write_onset_delta_summary,
)
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
    chord_ids = np.full(96, -1, dtype=np.int64)
    chord_ids[:2] = [0, 1]
    arrays = {
        "probability": probability,
        "target": target,
        "frame_mask": frame_mask,
        "chord_mask": chord_mask,
        "chord_pitches": chord_pitches,
        "chord_ids": chord_ids,
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
                        "chord_tolerance_seconds": 0.04,
                        "duration_seconds": 0.75,
                        "velocity": 72,
                    },
                }
            )
            printed = io.StringIO()
            onset_reports = [
                {"record_id": "subject/a"},
                {"record_id": "subject/inside"},
                {"record_id": "subject/take.v2"},
            ]
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
                patch.object(
                    sonify_stage,
                    "calculate_onset_deltas",
                    side_effect=onset_reports,
                ) as onset_delta_calculate,
                patch.object(
                    sonify_stage,
                    "write_onset_delta_summary",
                    side_effect=lambda _, output: Path(output).resolve(),
                ) as onset_summary_write,
                patch.object(
                    sonify_stage,
                    "plot_onset_delta_summary",
                    side_effect=lambda _, output: Path(output).resolve(),
                ) as onset_summary_plot,
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
            self.assertEqual(
                onset_delta_calculate.call_args_list,
                [
                    call(
                        artifacts[0],
                        midi_root / "subject" / "a.mid",
                        threshold=0.35,
                        chord_tolerance_seconds=0.04,
                    ),
                    call(
                        nested_artifact,
                        midi_root / "subject" / "inside.mid",
                        threshold=0.35,
                        chord_tolerance_seconds=0.04,
                    ),
                    call(
                        artifacts[1],
                        midi_root / "subject" / "take.v2.mid",
                        threshold=0.35,
                        chord_tolerance_seconds=0.04,
                    ),
                ],
            )
            onset_summary_write.assert_called_once_with(
                onset_reports,
                artifact_root / "onset_delta_summary.json",
            )
            onset_summary_plot.assert_called_once_with(
                artifact_root / "onset_delta_summary.json",
                artifact_root / "onset_delta_summary.png",
            )
            self.assertIn("inside.mid", printed.getvalue())
            self.assertNotIn(".onsets.json", printed.getvalue())
            self.assertIn("onset_delta_summary.json", printed.getvalue())
            self.assertIn("onset_delta_summary.png", printed.getvalue())

    def test_config_handlers_preserve_absolute_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_root = root / "absolute-input"
            artifact_root.mkdir()
            artifact = artifact_root / "take.v2.npz"
            artifact.touch()
            midi_root = root / "absolute-midi"
            source_midi = midi_root / "nested" / "take.v2.mid"
            onset_report = {"record_id": "nested/take.v2"}
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
                        "chord_tolerance_seconds": 0.05,
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
                patch.object(
                    sonify_stage,
                    "calculate_onset_deltas",
                    return_value=onset_report,
                ) as onset_delta_calculate,
                patch.object(
                    sonify_stage,
                    "write_onset_delta_summary",
                    return_value=(
                        artifact_root / "onset_delta_summary.json"
                    ).resolve(),
                ) as onset_summary_write,
                patch.object(
                    sonify_stage,
                    "plot_onset_delta_summary",
                    return_value=(
                        artifact_root / "onset_delta_summary.png"
                    ).resolve(),
                ) as onset_summary_plot,
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
            onset_delta_calculate.assert_called_once_with(
                artifact,
                source_midi.resolve(),
                threshold=0.2,
                chord_tolerance_seconds=0.05,
            )
            onset_summary_write.assert_called_once_with(
                [onset_report],
                artifact_root / "onset_delta_summary.json",
            )
            onset_summary_plot.assert_called_once_with(
                (artifact_root / "onset_delta_summary.json").resolve(),
                artifact_root / "onset_delta_summary.png",
            )

    def test_visualize_artifact_draws_three_masked_heatmaps(self) -> None:
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
            self.assertEqual(loaded.chord_ids[:2].tolist(), [0, 1])
            with patch.dict(
                "os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}
            ):
                import matplotlib.axes

                original_imshow = matplotlib.axes.Axes.imshow
                with patch.object(
                    matplotlib.axes.Axes,
                    "imshow",
                    autospec=True,
                    side_effect=original_imshow,
                ) as imshow:
                    self.assertEqual(
                        visualize_artifact(artifact, output), output.resolve()
                    )

            self.assertEqual(imshow.call_count, 3)
            delta = imshow.call_args_list[2].args[1]
            self.assertAlmostEqual(float(delta[0, 0]), -0.2)
            self.assertAlmostEqual(float(delta[0, 1]), 0.8)
            self.assertTrue(bool(delta.mask[0, 2]))
            self.assertEqual(imshow.call_args_list[2].kwargs["vmin"], -1.0)
            self.assertEqual(imshow.call_args_list[2].kwargs["vmax"], 1.0)
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

    def test_onset_deltas_use_source_chord_ids_and_mark_missed_chords(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "result.npz"
            source = root / "source.mid"
            probability = np.zeros((3, 96), dtype=np.float32)
            probability[0, 0] = 0.5
            probability[1, 0] = 0.5
            probability[2, 0] = 1.0
            probability[1, 1] = 0.49
            chord_ids = np.full(96, -1, dtype=np.int64)
            chord_ids[:2] = [1, 2]
            _write_artifact(
                artifact,
                probability=probability,
                chord_ids=chord_ids,
                fps=np.asarray(10.0, dtype=np.float64),
                frame_start=np.asarray(10, dtype=np.int64),
            )

            events = [
                (480, "note_on", 50, 50),
                (576, "note_off", 50, 0),
                (1008, "note_on", 60, 60),
                (1032, "note_on", 64, 64),
                (1104, "note_off", 60, 0),
                (1128, "note_off", 64, 0),
                (1200, "note_on", 67, 67),
                (1296, "note_off", 67, 0),
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

            report = calculate_onset_deltas(
                artifact,
                source,
                threshold=0.5,
                chord_tolerance_seconds=0.03,
            )
            self.assertEqual(report["frame_start"], 10)
            self.assertEqual(len(report["chords"]), 2)

            detected, missed = report["chords"]
            self.assertEqual(detected["chord_id"], 0)
            self.assertEqual(detected["source_chord_id"], 1)
            self.assertTrue(detected["detected"])
            self.assertAlmostEqual(detected["peak_probability"], 0.5)
            self.assertAlmostEqual(detected["real_onset_seconds"], 1.05)
            self.assertAlmostEqual(detected["predicted_onset_seconds"], 1.0)
            self.assertAlmostEqual(detected["delta_seconds"], -0.05)
            self.assertAlmostEqual(detected["absolute_delta_seconds"], 0.05)

            self.assertEqual(missed["chord_id"], 1)
            self.assertEqual(missed["source_chord_id"], 2)
            self.assertFalse(missed["detected"])
            self.assertAlmostEqual(missed["peak_probability"], 0.49)
            self.assertAlmostEqual(missed["real_onset_seconds"], 1.25)
            self.assertIsNone(missed["predicted_onset_seconds"])
            self.assertIsNone(missed["delta_seconds"])
            self.assertIsNone(missed["absolute_delta_seconds"])

    def test_onset_delta_summary_averages_relative_chord_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = [
                {
                    "record_id": "subject/a",
                    "chords": [
                        {
                            "chord_id": 0,
                            "source_chord_id": 7,
                            "detected": True,
                            "delta_seconds": -0.1,
                            "absolute_delta_seconds": 0.1,
                        },
                        {
                            "chord_id": 1,
                            "source_chord_id": 8,
                            "detected": False,
                            "delta_seconds": None,
                            "absolute_delta_seconds": None,
                        },
                    ],
                },
                {
                    "record_id": "subject/a",
                    "chords": [
                        {
                            "chord_id": 0,
                            "source_chord_id": 19,
                            "detected": True,
                            "delta_seconds": 0.3,
                            "absolute_delta_seconds": 0.3,
                        },
                        {
                            "chord_id": 1,
                            "source_chord_id": 20,
                            "detected": True,
                            "delta_seconds": 0.2,
                            "absolute_delta_seconds": 0.2,
                        },
                    ],
                },
                {
                    "record_id": "subject/b",
                    "chords": [
                        {
                            "chord_id": 0,
                            "source_chord_id": 7,
                            "detected": True,
                            "delta_seconds": -0.4,
                            "absolute_delta_seconds": 0.4,
                        },
                        {
                            "chord_id": 1,
                            "source_chord_id": 8,
                            "detected": False,
                            "delta_seconds": None,
                            "absolute_delta_seconds": None,
                        },
                        {
                            "chord_id": 2,
                            "source_chord_id": 9,
                            "detected": False,
                            "delta_seconds": None,
                            "absolute_delta_seconds": None,
                        },
                    ],
                },
            ]
            output = root / "onset_delta_summary.json"

            self.assertEqual(
                write_onset_delta_summary(reports, output),
                output.resolve(),
            )
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_count"], 3)
            self.assertEqual(
                summary["delta_definition"],
                "predicted_onset_seconds - real_onset_seconds",
            )
            self.assertEqual(
                [chord["chord_id"] for chord in summary["chords"]],
                [0, 1, 2],
            )

            first, second, third = summary["chords"]
            self.assertEqual(first["observation_count"], 3)
            self.assertEqual(first["detected_count"], 3)
            self.assertEqual(first["missed_count"], 0)
            self.assertAlmostEqual(first["mean_delta_seconds"], -0.2 / 3)
            self.assertAlmostEqual(first["mean_absolute_delta_seconds"], 0.8 / 3)

            self.assertEqual(second["observation_count"], 3)
            self.assertEqual(second["detected_count"], 1)
            self.assertEqual(second["missed_count"], 2)
            self.assertAlmostEqual(second["mean_delta_seconds"], 0.2)
            self.assertAlmostEqual(second["mean_absolute_delta_seconds"], 0.2)

            self.assertEqual(third["observation_count"], 1)
            self.assertEqual(third["detected_count"], 0)
            self.assertEqual(third["missed_count"], 1)
            self.assertIsNone(third["mean_delta_seconds"])
            self.assertIsNone(third["mean_absolute_delta_seconds"])

            chart = root / "onset_delta_summary.png"
            with patch.dict(
                "os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}
            ):
                import matplotlib.axes

                original_bar = matplotlib.axes.Axes.bar
                with patch.object(
                    matplotlib.axes.Axes,
                    "bar",
                    autospec=True,
                    side_effect=original_bar,
                ) as bar:
                    self.assertEqual(
                        plot_onset_delta_summary(output, chart),
                        chart.resolve(),
                    )

            self.assertEqual(bar.call_count, 2)
            timing_call, missing_call = bar.call_args_list
            self.assertEqual(timing_call.args[1], [0, 1])
            self.assertAlmostEqual(timing_call.args[2][0], 0.8 / 3)
            self.assertAlmostEqual(timing_call.args[2][1], 0.2)
            self.assertEqual(missing_call.args[1], [0, 1, 2])
            self.assertEqual(missing_call.args[2], [0, 2, 1])
            self.assertTrue(chart.is_file())
            self.assertGreater(chart.stat().st_size, 0)

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
