"""Tests for artifact-based alignment evaluation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

import airpiano.evaluation as evaluation_stage
from airpiano.evaluation import (
    aggregate_alignment_metrics,
    calculate_alignment_metrics,
)
from airpiano.render.artifact import InferenceArtifact


TOLERANCES = (0, 2, 4, 8, 16)


def _artifact(
    predicted_frames: list[int | None],
    *,
    target_frames: list[int | None] | None = None,
    frame_count: int = 10,
    fps: float = 10.0,
    threshold: float = 0.5,
    record_id: str = "subject/take",
    frame_start: int = 0,
) -> InferenceArtifact:
    chord_count = len(predicted_frames)
    probability = np.zeros((frame_count, chord_count), dtype=np.float32)
    target = np.zeros_like(probability)
    if target_frames is None:
        target_frames = list(range(1, 3 * chord_count + 1, 3))
    for chord_slot, frame in enumerate(predicted_frames):
        if frame is not None:
            probability[frame, chord_slot] = threshold
    for chord_slot, frame in enumerate(target_frames):
        if frame is not None:
            target[frame, chord_slot] = 1.0
    return InferenceArtifact(
        probability=probability,
        target=target,
        frame_mask=np.ones(frame_count, dtype=np.bool_),
        chord_mask=np.ones(chord_count, dtype=np.bool_),
        chord_pitches=np.zeros((chord_count, 128), dtype=np.bool_),
        chord_ids=np.arange(chord_count, dtype=np.int64),
        fps=fps,
        record_id=record_id,
        frame_start=frame_start,
    )


def _write_artifact(
    path: Path,
    *,
    record_id: str = "subject/take",
    frame_start: int = 0,
    fps: float = 10.0,
) -> None:
    artifact = _artifact(
        [1, 4, 7],
        record_id=record_id,
        frame_start=frame_start,
        fps=fps,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("airpiano-inference-v1"),
        probability=artifact.probability,
        target=artifact.target,
        frame_mask=artifact.frame_mask,
        chord_mask=artifact.chord_mask,
        chord_pitches=artifact.chord_pitches,
        chord_ids=artifact.chord_ids,
        fps=np.asarray(artifact.fps),
        record_id=np.asarray(artifact.record_id),
        frame_start=np.asarray(artifact.frame_start),
        loss=np.asarray(0.25),
    )


class AlignmentMetricTest(unittest.TestCase):
    def test_perfect_alignment_and_threshold_tie(self) -> None:
        artifact = _artifact([1, 4, 7])
        artifact.probability[2, 0] = 0.5
        metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertEqual(metrics["chord_acc"], 1.0)
        self.assertTrue(all(value == 1.0 for value in metrics["c_ar"].values()))
        self.assertEqual(metrics["onset_mae_frames"], 0.0)
        self.assertEqual(metrics["onset_mae_seconds"], 0.0)
        self.assertEqual(metrics["chord_coverage"], 1.0)

    def test_missing_chord_lowers_coverage_and_ratios(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([1, None, 8]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertAlmostEqual(metrics["chord_acc"], 5 / 9)
        self.assertAlmostEqual(metrics["c_ar"]["0"], 1 / 3)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 2 / 3)
        self.assertEqual(metrics["onset_mae_frames"], 0.5)
        self.assertEqual(metrics["onset_mae_seconds"], 0.05)
        self.assertAlmostEqual(metrics["chord_coverage"], 2 / 3)

    def test_tolerances_are_inclusive_and_alignment_stays_monotonic(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([1, 8, 3]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertAlmostEqual(metrics["chord_acc"], 5 / 9)
        self.assertAlmostEqual(metrics["c_ar"]["0"], 1 / 3)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 1 / 3)
        self.assertEqual(metrics["c_ar"]["4"], 1.0)
        self.assertAlmostEqual(metrics["onset_mae_frames"], 8 / 3)

    def test_target_uses_earliest_positive_peak_and_excludes_zero_columns(
        self,
    ) -> None:
        artifact = _artifact(
            [1, 0],
            target_frames=[1, None],
            frame_count=4,
            frame_start=20,
        )
        artifact.target[2, 0] = 1.0
        metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertEqual(metrics["ground_truth_chord_count"], 1)
        self.assertEqual(metrics["detected_chord_count"], 1)
        self.assertEqual(metrics["onset_mae_frames"], 0.0)
        self.assertEqual(metrics["chord_acc"], 1.0)

    def test_micro_and_macro_use_different_artifact_weighting(self) -> None:
        first = calculate_alignment_metrics(
            _artifact([1, None, 8]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        second = calculate_alignment_metrics(
            _artifact([1], frame_count=3),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        micro, macro = aggregate_alignment_metrics(
            [first, second], TOLERANCES
        )

        self.assertAlmostEqual(micro["chord_coverage"], 3 / 4)
        self.assertAlmostEqual(macro["chord_coverage"], 5 / 6)
        self.assertAlmostEqual(macro["chord_acc_variance"], 4 / 81)
        self.assertEqual(macro["artifact_count"], 2)
        self.assertEqual(macro["onset_mae_artifact_count"], 2)

    def test_chord_acc_variance_excludes_null_artifacts(self) -> None:
        valid = calculate_alignment_metrics(
            _artifact([1], frame_count=3),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        empty = calculate_alignment_metrics(
            _artifact([None], target_frames=[None]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        _, one_valid = aggregate_alignment_metrics(
            [valid, empty], TOLERANCES
        )
        _, no_valid = aggregate_alignment_metrics([empty], TOLERANCES)

        self.assertEqual(one_valid["chord_acc"], 1.0)
        self.assertEqual(one_valid["chord_acc_variance"], 0.0)
        self.assertEqual(one_valid["chord_acc_artifact_count"], 1)
        self.assertIsNone(no_valid["chord_acc"])
        self.assertIsNone(no_valid["chord_acc_variance"])

    def test_indexed_rates_and_plot_follow_relative_positions(self) -> None:
        observations = [
            evaluation_stage._alignment_observations(
                _artifact([1, None, 9]), 0.5
            ),
            evaluation_stage._alignment_observations(
                _artifact([1], frame_count=3), 0.5
            ),
        ]
        indexed = evaluation_stage.aggregate_indexed_alignment_metrics(
            observations, (0, 2)
        )

        np.testing.assert_array_equal(
            indexed["frame_observation_count"],
            [2, 2, 1, 1, 1, 1, 1, 1, 1],
        )
        np.testing.assert_allclose(
            indexed["frame_chord_acc"],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_array_equal(
            indexed["chord_observation_count"], [2, 1, 1]
        )
        np.testing.assert_allclose(indexed["chord_c_ar"]["0"], [1, 0, 0])
        np.testing.assert_allclose(indexed["chord_c_ar"]["2"], [1, 0, 1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "summary.png"
            with patch.dict(
                "os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}
            ):
                import matplotlib.axes

                chord_acc_values = (0.1, 0.55, 1.0)
                original_plot = matplotlib.axes.Axes.plot
                original_hist = matplotlib.axes.Axes.hist
                with (
                    patch.object(
                        matplotlib.axes.Axes,
                        "plot",
                        autospec=True,
                        side_effect=original_plot,
                    ) as plot,
                    patch.object(
                        matplotlib.axes.Axes,
                        "hist",
                        autospec=True,
                        side_effect=original_hist,
                    ) as hist,
                ):
                    self.assertEqual(
                        evaluation_stage.plot_indexed_alignment_summary(
                            indexed,
                            (0, 2),
                            chart,
                            chord_acc_values=chord_acc_values,
                        ),
                        chart.resolve(),
                    )
                    self.assertTrue(chart.is_file())

        self.assertEqual(plot.call_count, 3)
        frame_call, exact_call, tolerant_call = plot.call_args_list
        np.testing.assert_array_equal(frame_call.args[1], np.arange(9))
        np.testing.assert_allclose(
            frame_call.args[2], indexed["frame_chord_acc"]
        )
        np.testing.assert_allclose(exact_call.args[2], [1, 0, 0])
        np.testing.assert_allclose(tolerant_call.args[2], [1, 0, 1])
        hist.assert_called_once()
        np.testing.assert_allclose(hist.call_args.args[1], chord_acc_values)
        np.testing.assert_allclose(
            hist.call_args.kwargs["bins"], np.linspace(0.0, 1.0, 51)
        )

    def test_no_detections_keep_zero_rates_and_null_mae(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([None]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        micro, macro = aggregate_alignment_metrics([metrics], TOLERANCES)

        self.assertEqual(micro["chord_coverage"], 0.0)
        self.assertTrue(all(value == 0.0 for value in micro["c_ar"].values()))
        self.assertIsNone(micro["onset_mae_frames"])
        self.assertEqual(macro["chord_coverage"], 0.0)
        self.assertIsNone(macro["onset_mae_frames"])
        self.assertEqual(macro["onset_mae_artifact_count"], 0)


class EvaluationStageTest(unittest.TestCase):
    @staticmethod
    def _config(
        root: Path,
        remove: bool,
        *,
        summary_only: bool = True,
    ) -> object:
        return OmegaConf.create(
            {
                "evaluation": {
                    "root": str(root / "artifacts"),
                    "output_path": str(root / "metrics.json"),
                    "onset_threshold": 0.5,
                    "tolerances_frames": list(TOLERANCES),
                    "remove_inference_artifacts": remove,
                    "summary_only": summary_only,
                }
            }
        )

    @staticmethod
    def _populate(root: Path) -> list[Path]:
        paths = [
            root / "artifacts" / "first.npz",
            root / "artifacts" / "nested" / "second.npz",
            root / "artifacts" / "nested" / "third.npz",
        ]
        _write_artifact(paths[0], record_id="subject/a", frame_start=0)
        _write_artifact(paths[1], record_id="subject/a", frame_start=5)
        _write_artifact(paths[2], record_id="subject/b", frame_start=0, fps=5.0)
        (root / "artifacts" / "test_loss.json").write_text(
            '{"test_loss": 0.25}\n',
            encoding="utf-8",
        )
        return paths

    def test_recursively_evaluates_each_artifact_and_preserves_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._populate(root)
            config = self._config(root, False, summary_only=False)

            evaluation_stage.evaluation(config)

            self.assertTrue(all(path.is_file() for path in paths))
            report = json.loads(
                (root / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["schema_version"], "airpiano-evaluation-v5")
            self.assertEqual(report["artifact_count"], 3)
            self.assertAlmostEqual(report["average_duration_seconds"], 4 / 3)
            self.assertEqual(report["total_duration_seconds"], 4.0)
            self.assertFalse(report["summary_only"])
            self.assertEqual(report["unique_record_count"], 2)
            self.assertEqual(report["unique_record_ids"], ["subject/a", "subject/b"])
            self.assertEqual(report["micro"]["ground_truth_chord_count"], 9)
            self.assertEqual(report["macro"]["artifact_count"], 3)
            self.assertEqual(report["macro"]["chord_acc_variance"], 0.0)
            self.assertFalse(report["inference_artifacts_removed"])
            self.assertEqual(
                [item["duration_seconds"] for item in report["artifacts"]],
                [1.0, 1.0, 2.0],
            )
            self.assertEqual(
                [item["artifact_path"] for item in report["artifacts"]],
                ["first.npz", "nested/second.npz", "nested/third.npz"],
            )

    def test_summary_only_omits_artifact_details_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._populate(root)
            output = io.StringIO()

            with redirect_stdout(output):
                evaluation_stage.evaluation(self._config(root, False))

            report = json.loads(
                (root / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["summary_only"])
            self.assertAlmostEqual(report["average_duration_seconds"], 4 / 3)
            self.assertEqual(report["total_duration_seconds"], 4.0)
            self.assertEqual(
                Path(report["summary_plot_path"]),
                (root / "metrics.png").resolve(),
            )
            self.assertTrue((root / "metrics.png").is_file())
            self.assertNotIn("artifacts", report)
            self.assertNotIn("indexed_metrics", report)
            self.assertNotIn("unique_record_ids", report)
            console = output.getvalue()
            self.assertNotIn("evaluation 1/3", console)
            self.assertIn("evaluation micro:", console)
            self.assertIn("evaluation macro:", console)
            self.assertIn(
                "evaluation average_duration_seconds=", console
            )
            self.assertIn("wrote evaluation summary chart", console)

    def test_cleanup_removes_only_npzs_after_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._populate(root)
            config = self._config(root, True)

            evaluation_stage.evaluation(config)

            self.assertTrue((root / "metrics.json").is_file())
            self.assertTrue((root / "artifacts" / "test_loss.json").is_file())
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertTrue((root / "metrics.png").is_file())
            report = json.loads(
                (root / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["inference_artifacts_removed"])

    def test_output_failure_preserves_inference_artifacts(self) -> None:
        failures = (
            ("plot_indexed_alignment_summary", "plot failed"),
            ("write_json", "write failed"),
        )
        for function_name, message in failures:
            with self.subTest(function_name=function_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = self._populate(root)
                    config = self._config(root, True)

                    with patch.object(
                        evaluation_stage,
                        function_name,
                        side_effect=OSError(message),
                    ):
                        with self.assertRaisesRegex(OSError, message):
                            evaluation_stage.evaluation(config)

                    self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
