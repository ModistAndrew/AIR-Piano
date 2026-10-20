"""Tests for artifact-based alignment evaluation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
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
    release_times: list[float] | None = None,
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
        release_time=(
            None if release_times is None else np.asarray(release_times)
        ),
        onset_prediction_offset=0.1 if release_times is not None else None,
    )


def _write_artifact(
    path: Path,
    *,
    record_id: str = "subject/take",
    frame_start: int = 0,
    fps: float = 10.0,
    release_times: list[float] | None = None,
) -> None:
    artifact = _artifact(
        [1, 4, 7],
        record_id=record_id,
        frame_start=frame_start,
        fps=fps,
        release_times=release_times,
    )
    release_metadata = {}
    if artifact.release_time is not None:
        present = np.isfinite(artifact.release_time)
        release_metadata = {
            "release_time": artifact.release_time,
            "predicted_onset": artifact.release_time.copy(),
            "release_frame_index": np.where(
                present, np.rint((artifact.release_time - 0.05) * fps), -1
            ).astype(np.int64),
            "release_frame_time": artifact.release_time - 0.05,
            "note_duration": np.asarray(1.0),
            "note_velocity": np.asarray(64),
            "onset_prediction_offset": np.asarray(0.1),
        }
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
        **release_metadata,
    )


class AlignmentMetricTest(unittest.TestCase):
    def test_perfect_alignment_and_threshold_tie(self) -> None:
        artifact = _artifact([1, 4, 7])
        artifact.probability[2, 0] = 0.5
        metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        self.assertEqual(metrics["chord_acc"], 1.0)
        self.assertTrue(all(value == 1.0 for value in metrics["c_ar"].values()))
        self.assertEqual(metrics["onset_mae_frames"], 0.0)
        self.assertEqual(metrics["onset_mae_seconds"], 0.0)
        self.assertEqual(metrics["onset_mean_delta_frames"], 0.0)
        self.assertEqual(metrics["onset_mean_delta_seconds"], 0.0)
        self.assertEqual(metrics["chord_coverage"], 1.0)

    def test_first_threshold_uses_radius_compensated_future_frame(self) -> None:
        artifact = _artifact(
            [5, 6],
            target_frames=[3, 5],
            frame_count=8,
            fps=30.0,
        )
        artifact.probability[1, :] = 0.5
        artifact.probability[5, 0] = 0.8
        artifact.probability[6, 1] = 0.8

        first_threshold = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            target_radius_seconds=0.2,
        )
        peak = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        self.assertEqual(first_threshold["onset_mean_delta_frames"], 0.0)
        self.assertEqual(first_threshold["onset_mae_frames"], 1.0)
        self.assertAlmostEqual(first_threshold["chord_acc"], 3 / 5)
        self.assertEqual(peak["onset_mean_delta_frames"], 1.5)
        self.assertAlmostEqual(peak["chord_acc"], 2 / 5)

    def test_missing_chord_lowers_coverage_and_ratios(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([1, None, 8]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        self.assertAlmostEqual(metrics["chord_acc"], 5 / 9)
        self.assertAlmostEqual(metrics["c_ar"]["0"], 1 / 3)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 2 / 3)
        self.assertEqual(metrics["onset_mae_frames"], 0.5)
        self.assertEqual(metrics["onset_mae_seconds"], 0.05)
        self.assertEqual(metrics["onset_mean_delta_frames"], 0.5)
        self.assertEqual(metrics["onset_mean_delta_seconds"], 0.05)
        self.assertAlmostEqual(metrics["chord_coverage"], 2 / 3)

    def test_tolerances_are_inclusive_and_alignment_stays_monotonic(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([1, 8, 3]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        self.assertAlmostEqual(metrics["chord_acc"], 5 / 9)
        self.assertAlmostEqual(metrics["c_ar"]["0"], 1 / 3)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 1 / 3)
        self.assertEqual(metrics["c_ar"]["4"], 1.0)
        self.assertAlmostEqual(metrics["onset_mae_frames"], 8 / 3)
        self.assertEqual(metrics["onset_mean_delta_frames"], 0.0)
        self.assertEqual(metrics["onset_mean_delta_seconds"], 0.0)

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
            onset_mode="peak",
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
            onset_mode="peak",
        )
        second = calculate_alignment_metrics(
            _artifact([1], frame_count=3),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )
        micro, macro = aggregate_alignment_metrics(
            [first, second], TOLERANCES
        )

        self.assertAlmostEqual(micro["chord_coverage"], 3 / 4)
        self.assertAlmostEqual(macro["chord_coverage"], 5 / 6)
        self.assertAlmostEqual(micro["onset_mean_delta_frames"], 1 / 3)
        self.assertAlmostEqual(macro["onset_mean_delta_frames"], 0.25)
        self.assertAlmostEqual(macro["chord_acc_variance"], 4 / 81)
        self.assertEqual(macro["artifact_count"], 2)
        self.assertEqual(macro["onset_mae_artifact_count"], 2)
        self.assertEqual(macro["onset_mean_delta_artifact_count"], 2)

    def test_chord_acc_variance_excludes_null_artifacts(self) -> None:
        valid = calculate_alignment_metrics(
            _artifact([1], frame_count=3),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )
        empty = calculate_alignment_metrics(
            _artifact([None], target_frames=[None]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
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
                _artifact([1, None, 9]), 0.5, "peak"
            ),
            evaluation_stage._alignment_observations(
                _artifact([1], frame_count=3), 0.5, "peak"
            ),
        ]
        indexed = evaluation_stage.aggregate_indexed_alignment_metrics(
            observations, (2, 0)
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

                c_ar_values = (0.0, 0.55, 1.0)
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
                            (2, 0),
                            chart,
                            c_ar_values=c_ar_values,
                        ),
                        chart.resolve(),
                    )
                    self.assertTrue(chart.is_file())

        self.assertEqual(plot.call_count, 3)
        frame_call, tolerant_call, exact_call = plot.call_args_list
        np.testing.assert_array_equal(frame_call.args[1], np.arange(9))
        np.testing.assert_allclose(
            frame_call.args[2], indexed["frame_chord_acc"]
        )
        np.testing.assert_allclose(exact_call.args[2], [1, 0, 0])
        np.testing.assert_allclose(tolerant_call.args[2], [1, 0, 1])
        hist.assert_called_once()
        np.testing.assert_allclose(hist.call_args.args[1], c_ar_values)
        np.testing.assert_allclose(
            hist.call_args.kwargs["bins"], np.linspace(0.0, 1.0, 51)
        )
        histogram_axis = hist.call_args.args[0]
        self.assertEqual(histogram_axis.get_xlabel(), "C-AR (delta=2 frames)")
        self.assertEqual(histogram_axis.get_ylabel(), "Artifact count")
        self.assertEqual(
            histogram_axis.get_title(),
            "Per-artifact C-AR (delta=2 frames) distribution",
        )
        self.assertEqual(histogram_axis.get_xlim(), (0.0, 1.0))

    def test_empty_histogram_identifies_largest_tolerance(self) -> None:
        indexed = evaluation_stage.aggregate_indexed_alignment_metrics(
            [], (2, 0)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}
            ):
                import matplotlib.axes

                with patch.object(
                    matplotlib.axes.Axes,
                    "text",
                    autospec=True,
                    side_effect=matplotlib.axes.Axes.text,
                ) as text:
                    evaluation_stage.plot_indexed_alignment_summary(
                        indexed, (2, 0), root / "empty.png"
                    )

        self.assertIn(
            "No artifact C-AR (delta=2 frames) values",
            [call.args[3] for call in text.call_args_list],
        )

    def test_no_detections_keep_zero_rates_and_null_mae(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([None]),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )
        micro, macro = aggregate_alignment_metrics([metrics], TOLERANCES)

        self.assertEqual(micro["chord_coverage"], 0.0)
        self.assertTrue(all(value == 0.0 for value in micro["c_ar"].values()))
        self.assertIsNone(micro["onset_mae_frames"])
        self.assertIsNone(micro["onset_mean_delta_frames"])
        self.assertEqual(macro["chord_coverage"], 0.0)
        self.assertIsNone(macro["onset_mae_frames"])
        self.assertIsNone(macro["onset_mean_delta_frames"])
        self.assertEqual(macro["onset_mae_artifact_count"], 0)
        self.assertEqual(macro["onset_mean_delta_artifact_count"], 0)


class ReleaseTimeMetricTest(unittest.TestCase):
    def test_release_times_override_probability_and_predicted_onset(self) -> None:
        artifact = replace(
            _artifact([1, 4, 7], release_times=[0.25, 0.6, np.nan]),
            predicted_onset=np.asarray([0.1, 0.4, 0.7]),
        )
        with patch.object(
            evaluation_stage, "decode_chord_onsets", side_effect=AssertionError
        ):
            metrics = calculate_alignment_metrics(
                artifact,
                onset_threshold=0.5,
                tolerances_frames=TOLERANCES,
                onset_mode="first_threshold",
                target_radius_seconds=0.4,
            )

        self.assertEqual(metrics["detected_chord_count"], 2)
        self.assertAlmostEqual(metrics["chord_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["chord_acc"], 2 / 9)
        self.assertEqual(metrics["onset_mae_frames"], 1.75)
        self.assertAlmostEqual(metrics["onset_mae_seconds"], 0.175)
        self.assertEqual(metrics["onset_mean_delta_frames"], 1.75)
        self.assertEqual(metrics["c_ar"]["0"], 0.0)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 2 / 3)

    def test_all_missing_releases_do_not_fall_back_to_probability(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([1, 4, 7], release_times=[np.nan] * 3),
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        self.assertEqual(metrics["ground_truth_chord_count"], 3)
        self.assertEqual(metrics["detected_chord_count"], 0)
        self.assertEqual(metrics["chord_coverage"], 0.0)
        self.assertEqual(metrics["chord_acc"], 0.0)
        self.assertTrue(all(value == 0.0 for value in metrics["c_ar"].values()))
        self.assertIsNone(metrics["onset_mae_frames"])
        self.assertIsNone(metrics["onset_mean_delta_frames"])

    def test_fractional_frames_preserve_inclusive_tolerance_and_activation(self) -> None:
        artifact = _artifact(
            [1, 4, 7], fps=8.0,
            release_times=[3 / 8, 6.125 / 8, 6.875 / 8],
        )
        metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertEqual(metrics["c_ar"]["0"], 0.0)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 2 / 3)
        self.assertEqual(metrics["c_ar"]["4"], 1.0)
        self.assertAlmostEqual(metrics["chord_acc"], 4 / 9)
        self.assertEqual(metrics["absolute_error_frames_sum"], 4.25)
        self.assertEqual(metrics["onset_delta_frames_sum"], 4.0)
        self.assertAlmostEqual(metrics["onset_mae_frames"], 4.25 / 3)
        self.assertAlmostEqual(metrics["onset_mae_seconds"], 4.25 / 24)
        self.assertAlmostEqual(metrics["onset_mean_delta_frames"], 4 / 3)

    def test_absolute_release_times_respect_frame_and_chord_masks(self) -> None:
        artifact = _artifact(
            [0, 0, 0, 0], target_frames=[1, 4, 6, None],
            frame_count=8, frame_start=20, fps=8.0,
            release_times=[22.5 / 8, 24 / 8, 0.0, 0.0],
        )
        artifact.frame_mask[[1, 3]] = False
        artifact.chord_mask[2] = False
        artifact.target[2, 0] = 0.75
        artifact.target[3, 0] = 1.0
        frame_correct, errors = evaluation_stage._alignment_observations(
            artifact, 0.5, "peak"
        )
        metrics = calculate_alignment_metrics(
            artifact,
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
            onset_mode="peak",
        )

        np.testing.assert_array_equal(frame_correct, [False, True, True, True, True])
        self.assertEqual(errors, {0: 0.5, 1: 0.0})
        self.assertEqual(metrics["ground_truth_chord_count"], 2)
        self.assertEqual(metrics["evaluated_frame_count"], 5)
        self.assertEqual(metrics["chord_acc"], 0.8)
        self.assertEqual(metrics["chord_coverage"], 1.0)
        self.assertEqual(metrics["onset_mae_frames"], 0.25)

    def test_out_of_order_releases_keep_highest_active_chord(self) -> None:
        artifact = _artifact(
            [1, 4, 7], fps=8.0,
            release_times=[1.125 / 8, 8.125 / 8, 3.125 / 8],
        )
        frame_correct, errors = evaluation_stage._alignment_observations(
            artifact, 0.5, "peak"
        )

        np.testing.assert_array_equal(
            frame_correct,
            [False, True, True, False, False, False, True, True, True],
        )
        self.assertEqual(errors, {0: 0.125, 1: 4.125, 2: -3.875})

    def test_fractional_errors_survive_micro_macro_and_indexed_aggregation(self) -> None:
        artifacts = [
            _artifact(
                [1, 4], frame_count=6, fps=8.0,
                release_times=[1.5 / 8, 3.75 / 8],
            ),
            _artifact([1], frame_count=4, release_times=[0.225]),
        ]
        metrics = [
            calculate_alignment_metrics(
                artifact, onset_threshold=0.5, tolerances_frames=(0, 1, 2)
            )
            for artifact in artifacts
        ]
        micro, macro = aggregate_alignment_metrics(metrics, (0, 1, 2))
        self.assertAlmostEqual(micro["onset_mae_frames"], 2 / 3)
        self.assertEqual(micro["onset_mean_delta_frames"], 0.5)
        self.assertAlmostEqual(micro["onset_mae_seconds"], 0.21875 / 3)
        self.assertAlmostEqual(micro["onset_mean_delta_seconds"], 0.15625 / 3)
        self.assertEqual(macro["onset_mae_frames"], 0.8125)
        self.assertEqual(macro["onset_mean_delta_frames"], 0.6875)
        observations = [
            evaluation_stage._alignment_observations(artifact, 0.5)
            for artifact in artifacts
        ]
        indexed = evaluation_stage.aggregate_indexed_alignment_metrics(
            observations, (0, 1, 2)
        )
        np.testing.assert_allclose(indexed["chord_c_ar"]["0"], [0.0, 0.0])
        np.testing.assert_allclose(indexed["chord_c_ar"]["1"], [0.5, 1.0])
        np.testing.assert_allclose(indexed["chord_c_ar"]["2"], [1.0, 1.0])


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
                    "onset_mode": "peak",
                    "target_radius_seconds": 0.2,
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
            self.assertEqual(report["schema_version"], "airpiano-evaluation-v8")
            self.assertEqual(report["onset_mode"], "peak")
            self.assertEqual(report["onset_threshold"], 0.5)
            self.assertEqual(report["target_radius_seconds"], 0.2)
            self.assertEqual(report["prediction_horizon_seconds"], 0.0)
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

    def test_histogram_uses_largest_c_ar_excluding_null_and_retaining_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._populate(root)
            config = self._config(root, False, summary_only=False)
            config.evaluation.tolerances_frames = [2, 0, 1]
            artifacts = [
                _artifact([3], target_frames=[1]),
                _artifact([None]),
                _artifact([None], target_frames=[None]),
            ]
            with (
                patch.object(
                    evaluation_stage, "load_artifact", side_effect=artifacts
                ),
                patch.object(
                    evaluation_stage,
                    "plot_indexed_alignment_summary",
                    return_value=root / "metrics.png",
                ) as plot,
            ):
                evaluation_stage.evaluation(config)

            plot.assert_called_once()
            self.assertEqual(plot.call_args.args[1], (2, 0, 1))
            self.assertEqual(plot.call_args.kwargs["c_ar_values"], [1.0, 0.0])
            report = json.loads(
                (root / "metrics.json").read_text(encoding="utf-8")
            )
            metrics = [item["metrics"] for item in report["artifacts"]]
            self.assertEqual(metrics[0]["c_ar"]["0"], 0.0)
            self.assertEqual(metrics[0]["c_ar"]["2"], 1.0)
            self.assertNotEqual(metrics[0]["chord_acc"], 1.0)
            self.assertIsNone(metrics[2]["c_ar"]["2"])

    def test_release_only_stage_skips_target_validation_and_reports_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(
                root / "artifacts" / "release.npz",
                release_times=[0.15, 0.45, np.nan],
            )
            config = self._config(root, False, summary_only=False)
            config.evaluation.onset_mode = "first_threshold"
            self.assertNotIn("dataset", config)
            with patch.object(
                evaluation_stage, "plot_indexed_alignment_summary",
                return_value=root / "metrics.png",
            ):
                evaluation_stage.evaluation(config)
            report = json.loads((root / "metrics.json").read_text())

        self.assertEqual(report["schema_version"], "airpiano-evaluation-v8")
        self.assertEqual(report["onset_mode"], "release_time")
        self.assertEqual(report["onset_source_counts"], {"release_time": 1, "probability": 0})
        self.assertIsNone(report["onset_threshold"])
        self.assertEqual(report["target_radius_seconds"], 0.0)
        self.assertEqual(report["prediction_horizon_seconds"], 0.0)
        self.assertEqual(report["artifacts"][0]["onset_mode"], "release_time")
        self.assertEqual(report["artifacts"][0]["prediction_horizon_seconds"], 0.0)
        self.assertEqual(report["micro"]["onset_mae_frames"], 0.5)
        self.assertAlmostEqual(report["micro"]["chord_coverage"], 2 / 3)

    def test_mixed_sources_report_each_mode_and_validate_probability_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(
                root / "artifacts" / "a_release.npz",
                release_times=[0.1, 0.4, 0.7],
            )
            _write_artifact(root / "artifacts" / "b_probability.npz")
            _write_artifact(root / "artifacts" / "c_probability.npz")
            config = self._config(root, False, summary_only=False)
            config.evaluation.onset_mode = "first_threshold"
            config.dataset = {
                "target_generator": {
                    "target_mode": "onset",
                    "velocity_target": False,
                    "softener": {
                        "_target_": "airpiano.data.targets.TriangularSoftener"
                    },
                }
            }
            with (
                patch.object(
                    evaluation_stage, "validate_onset_target_config",
                    wraps=evaluation_stage.validate_onset_target_config,
                ) as validate,
                patch.object(
                    evaluation_stage, "plot_indexed_alignment_summary",
                    return_value=root / "metrics.png",
                ),
            ):
                evaluation_stage.evaluation(config)
            validate.assert_called_once_with(config, "first_threshold")
            report = json.loads((root / "metrics.json").read_text())

        self.assertEqual(report["onset_mode"], "mixed")
        self.assertEqual(report["onset_source_counts"], {"release_time": 1, "probability": 2})
        self.assertIsNone(report["prediction_horizon_seconds"])
        self.assertEqual(report["onset_threshold"], 0.5)
        self.assertEqual(report["target_radius_seconds"], 0.2)
        self.assertEqual(
            [item["onset_mode"] for item in report["artifacts"]],
            ["release_time", "first_threshold", "first_threshold"],
        )
        np.testing.assert_allclose(
            [item["prediction_horizon_seconds"] for item in report["artifacts"]],
            [0.0, 0.1, 0.1],
        )
        self.assertEqual(
            [item["metrics"]["onset_mae_frames"] for item in report["artifacts"]],
            [0.0, 1.0, 1.0],
        )

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
