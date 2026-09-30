"""Tests for split-based full-song evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

import airpiano.evaluation as evaluation_stage
from airpiano.evaluation import (
    aggregate_alignment_metrics,
    calculate_alignment_metrics,
    iter_evaluation_songs,
)
from airpiano.render.artifact import InferenceArtifact


TOLERANCES = (0, 2, 4, 8, 16)


def _artifact(
    predicted_frames: list[int | None],
    *,
    frame_count: int = 10,
    fps: float = 10.0,
    threshold: float = 0.5,
) -> InferenceArtifact:
    chord_count = len(predicted_frames)
    probability = np.zeros((frame_count, chord_count), dtype=np.float32)
    for chord_slot, frame in enumerate(predicted_frames):
        if frame is not None:
            probability[frame, chord_slot] = threshold
    return InferenceArtifact(
        probability=probability,
        target=np.zeros_like(probability),
        frame_mask=np.ones(frame_count, dtype=np.bool_),
        chord_mask=np.ones(chord_count, dtype=np.bool_),
        chord_pitches=np.zeros((chord_count, 128), dtype=np.bool_),
        chord_ids=np.arange(chord_count, dtype=np.int64),
        fps=fps,
        record_id="subject/take",
        frame_start=0,
    )


def _write_artifact(path: Path) -> None:
    artifact = _artifact([1, 4, 7])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probability=artifact.probability,
        target=artifact.target,
        frame_mask=artifact.frame_mask,
        chord_mask=artifact.chord_mask,
        chord_pitches=artifact.chord_pitches,
        chord_ids=artifact.chord_ids,
        fps=np.asarray(artifact.fps),
        record_id=np.asarray(artifact.record_id),
        frame_start=np.asarray(artifact.frame_start),
    )


class AlignmentMetricTest(unittest.TestCase):
    def test_perfect_alignment_and_threshold_tie(self) -> None:
        artifact = _artifact([1, 4, 7])
        artifact.probability[2, 0] = 0.5
        metrics = calculate_alignment_metrics(
            artifact,
            [0.1, 0.4, 0.7],
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
            [0.1, 0.4, 0.7],
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
            [0.1, 0.4, 0.7],
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )

        self.assertAlmostEqual(metrics["chord_acc"], 5 / 9)
        self.assertAlmostEqual(metrics["c_ar"]["0"], 1 / 3)
        self.assertAlmostEqual(metrics["c_ar"]["2"], 1 / 3)
        self.assertEqual(metrics["c_ar"]["4"], 1.0)
        self.assertAlmostEqual(metrics["onset_mae_frames"], 8 / 3)

    def test_micro_and_macro_use_different_record_weighting(self) -> None:
        first = calculate_alignment_metrics(
            _artifact([1, None, 8]),
            [0.1, 0.4, 0.7],
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        second = calculate_alignment_metrics(
            _artifact([1], frame_count=3),
            [0.1],
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        micro, macro = aggregate_alignment_metrics(
            [first, second], TOLERANCES
        )

        self.assertAlmostEqual(micro["chord_coverage"], 3 / 4)
        self.assertAlmostEqual(macro["chord_coverage"], 5 / 6)
        self.assertEqual(macro["record_count"], 2)
        self.assertEqual(macro["onset_mae_record_count"], 2)

    def test_no_detections_keep_zero_rates_and_null_mae(self) -> None:
        metrics = calculate_alignment_metrics(
            _artifact([None]),
            [0.1],
            onset_threshold=0.5,
            tolerances_frames=TOLERANCES,
        )
        micro, macro = aggregate_alignment_metrics([metrics], TOLERANCES)

        self.assertEqual(micro["chord_coverage"], 0.0)
        self.assertTrue(all(value == 0.0 for value in micro["c_ar"].values()))
        self.assertIsNone(micro["onset_mae_frames"])
        self.assertEqual(macro["chord_coverage"], 0.0)
        self.assertIsNone(macro["onset_mae_frames"])
        self.assertEqual(macro["onset_mae_record_count"], 0)


class EvaluationInputTest(unittest.TestCase):
    def test_streams_canonical_records_and_lazily_falls_back_to_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = ["subject/a", "subject/b"]
            for record in records:
                video = root / "video" / f"{record}.mp4"
                midi = root / "midi" / f"{record}.mid"
                video.parent.mkdir(parents=True, exist_ok=True)
                midi.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
                midi.touch()
            beat = root / "beats" / "subject/b.txt"
            beat.parent.mkdir(parents=True)
            beat.touch()

            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "midi_root": str(root / "midi"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "hands"),
                        "beat_root": str(root / "beats"),
                    },
                    "preprocess": {
                        "ragged_to_canonical": {"kind": "canonical"},
                        "video_to_ragged": {"kind": "video"},
                    },
                }
            )
            canonical = Mock()
            canonical.run.side_effect = [
                root / "hands/subject/a.npz",
                FileNotFoundError("raw missing"),
                root / "hands/subject/b.npz",
            ]
            video = Mock()
            reader = Mock(side_effect=lambda pair, *_: pair)
            components = {
                "file_loader": SimpleNamespace(
                    reader=SimpleNamespace(read=reader)
                ),
                "midi_loader": object(),
                "hand_loader": object(),
            }

            with patch.object(
                evaluation_stage,
                "instantiate",
                side_effect=[canonical, video],
            ) as instantiate:
                loaded = list(
                    iter_evaluation_songs(
                        config,
                        records,
                        overwrite=False,
                        components=components,
                    )
                )

            self.assertEqual([item[0] for item in loaded], [1, 2])
            self.assertEqual(instantiate.call_count, 2)
            video.run.assert_called_once_with(
                root / "video/subject/b.mp4",
                root / "raw/subject/b",
                False,
            )
            self.assertIsNone(loaded[0][2].beat_path)
            self.assertEqual(loaded[1][2].beat_path, beat)


class EvaluationStageTest(unittest.TestCase):
    @staticmethod
    def _config(root: Path, remove: bool) -> object:
        split = root / "splits" / "valid.txt"
        split.parent.mkdir(parents=True)
        split.write_text("subject/take\n", encoding="utf-8")
        return OmegaConf.create(
            {
                "device": "cpu",
                "alignment_loss": "bce",
                "train": {"bf16": False},
                "evaluation": {
                    "split_file": str(split),
                    "output_dir": str(root / "eval"),
                    "checkpoint": str(root / "last.pt"),
                    "overwrite": False,
                    "alignment_threshold": 0.2,
                    "advance_after_chords": 2,
                    "onset_threshold": 0.5,
                    "tolerances_frames": list(TOLERANCES),
                    "remove_inference_artifacts": remove,
                },
            }
        )

    @staticmethod
    def _song() -> object:
        return SimpleNamespace(
            record_id="subject/take",
            hands=np.zeros((10, 1), dtype=np.float32),
            chords=[
                SimpleNamespace(time=0.1),
                SimpleNamespace(time=0.4),
                SimpleNamespace(time=0.7),
            ],
            fps=10.0,
        )

    def _run(self, root: Path, remove: bool) -> Path:
        config = self._config(root, remove)
        song = self._song()

        def infer(*args, **kwargs):
            del kwargs
            _write_artifact(Path(args[5]))
            return 0.25, 30, 2

        with (
            patch.object(
                evaluation_stage.data,
                "build_data_components",
                return_value={"slicer": object(), "target_generator": object()},
            ),
            patch.object(
                evaluation_stage,
                "load_model_checkpoint",
                return_value=object(),
            ) as load_model,
            patch.object(
                evaluation_stage,
                "iter_evaluation_songs",
                return_value=iter([(1, 1, song)]),
            ),
            patch.object(
                evaluation_stage,
                "infer_full_song",
                side_effect=infer,
            ) as infer_song,
        ):
            evaluation_stage.evaluation(config)

        load_model.assert_called_once()
        infer_song.assert_called_once()
        return root / "eval" / "valid.json"

    def test_writes_split_metrics_and_optionally_removes_artifacts(self) -> None:
        for remove in (True, False):
            with self.subTest(
                remove=remove
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metrics_path = self._run(root, remove)
                artifact_dir = root / "eval" / "valid"
                self.assertTrue(metrics_path.is_file())
                self.assertEqual(artifact_dir.exists(), not remove)
                report = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(report["split_name"], "valid")
                self.assertEqual(report["full_song_loss"], 0.25)
                self.assertEqual(report["micro"]["chord_acc"], 1.0)
                self.assertEqual(report["inference_artifacts_removed"], remove)

    def test_write_failure_preserves_inference_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, True)
            song = self._song()

            def infer(*args, **kwargs):
                del kwargs
                _write_artifact(Path(args[5]))
                return 0.25, 30, 2

            with (
                patch.object(
                    evaluation_stage.data,
                    "build_data_components",
                    return_value={
                        "slicer": object(),
                        "target_generator": object(),
                    },
                ),
                patch.object(
                    evaluation_stage,
                    "load_model_checkpoint",
                    return_value=object(),
                ),
                patch.object(
                    evaluation_stage,
                    "iter_evaluation_songs",
                    return_value=iter([(1, 1, song)]),
                ),
                patch.object(
                    evaluation_stage,
                    "infer_full_song",
                    side_effect=infer,
                ),
                patch.object(
                    evaluation_stage,
                    "write_json",
                    side_effect=OSError("write failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "write failed"):
                    evaluation_stage.evaluation(config)

            self.assertTrue((root / "eval" / "valid").is_dir())


if __name__ == "__main__":
    unittest.main()
