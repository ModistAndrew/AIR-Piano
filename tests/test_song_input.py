"""Tests for shared split-song preparation and merged test inference."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mido
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import airpiano.song_input as song_input_stage
from airpiano import data, inference
from airpiano.preprocessing.canonical import CANONICAL_SCHEMA_VERSION
from airpiano.render.artifact import load_artifact
from airpiano.render.sonify import sonify_artifact
from airpiano.render.visualize import visualize_artifact


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.include_chord_duration = False
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        batch, frames = hands.shape[:2]
        return self.bias.expand(batch, frames, chords.shape[1])


class SplitSongInputTest(unittest.TestCase):
    def test_empty_split_does_not_construct_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            split = project / "empty.txt"
            split.write_text("", encoding="utf-8")
            config = OmegaConf.create({})
            with (
                patch.object(song_input_stage, "instantiate") as instantiate,
                patch.object(data, "build_data_components") as build_components,
            ):
                rows = list(
                    song_input_stage.iter_split_songs(
                        config,
                        split,
                        overwrite=False,
                        label="test",
                    )
                )

            self.assertEqual(rows, [])
            instantiate.assert_not_called()
            build_components.assert_not_called()

    def test_existing_canonical_does_not_construct_video_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("midi", "raw", "canonical"):
                (root / name).mkdir()
            split = root / "split.txt"
            split.write_text("take\n", encoding="utf-8")
            (root / "midi/take.mid").touch()
            canonical_path = root / "canonical/take.npz"
            canonical_path.touch()
            song = object()
            components = {
                "file_loader": SimpleNamespace(
                    reader=SimpleNamespace(read=lambda *args: song)
                ),
                "midi_loader": object(),
                "hand_loader": object(),
            }
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "midi_root": str(root / "midi"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "video_to_ragged": {"_target_": "example.Video"},
                        "ragged_to_canonical": {
                            "_target_": "example.Canonical"
                        },
                    },
                }
            )

            canonical_calls = []

            class Canonicalizer:
                def run(self, raw_record_dir, output_path, overwrite=False):
                    canonical_calls.append(Path(output_path))
                    return canonical_path

            canonicalizer = Canonicalizer()
            with patch.object(
                song_input_stage, "instantiate", return_value=canonicalizer
            ) as instantiate:
                rows = list(
                    song_input_stage.iter_split_songs(
                        config,
                        split,
                        overwrite=False,
                        label="test",
                        components=components,
                    )
                )

            instantiate.assert_called_once_with(
                config.preprocess.ragged_to_canonical
            )
            self.assertEqual(canonical_calls, [canonical_path])
            self.assertEqual(rows, [(1, 1, song, components)])

    def test_existing_raw_cache_is_canonicalized_without_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("midi", "raw/take", "canonical"):
                (root / name).mkdir(parents=True)
            split = root / "split.txt"
            split.write_text("take\n", encoding="utf-8")
            (root / "midi/take.mid").touch()
            (root / "raw/take/manifest.json").touch()
            (root / "raw/take/_SUCCESS").touch()
            song = object()
            components = {
                "file_loader": SimpleNamespace(
                    reader=SimpleNamespace(read=lambda *args: song)
                ),
                "midi_loader": object(),
                "hand_loader": object(),
            }
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "midi_root": str(root / "midi"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "video_to_ragged": {"_target_": "example.Video"},
                        "ragged_to_canonical": {
                            "_target_": "example.Canonical"
                        },
                    },
                }
            )
            canonicalizer = SimpleNamespace(
                run=lambda **kwargs: Path(kwargs["output_path"])
            )

            with patch.object(
                song_input_stage,
                "instantiate",
                return_value=canonicalizer,
            ) as instantiate:
                rows = list(
                    song_input_stage.iter_split_songs(
                        config,
                        split,
                        overwrite=False,
                        label="test",
                        components=components,
                    )
                )

            instantiate.assert_called_once_with(
                config.preprocess.ragged_to_canonical
            )
            self.assertEqual(rows, [(1, 1, song, components)])

    def test_malformed_completed_raw_cache_error_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("midi", "raw/take", "canonical"):
                (root / name).mkdir(parents=True)
            split = root / "split.txt"
            split.write_text("take\n", encoding="utf-8")
            (root / "midi/take.mid").touch()
            (root / "raw/take/manifest.json").touch()
            (root / "raw/take/_SUCCESS").touch()
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "midi_root": str(root / "midi"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                    },
                    "preprocess": {
                        "video_to_ragged": {"_target_": "example.Video"},
                        "ragged_to_canonical": {
                            "_target_": "example.Canonical"
                        },
                    },
                }
            )

            class MalformedCanonicalizer:
                def run(self, **kwargs):
                    del kwargs
                    raise ValueError("malformed cache")

            with patch.object(
                song_input_stage,
                "instantiate",
                return_value=MalformedCanonicalizer(),
            ):
                with self.assertRaisesRegex(ValueError, "malformed cache"):
                    list(
                        song_input_stage.iter_split_songs(
                            config,
                            split,
                            overwrite=False,
                            label="test",
                            components={"unused": object()},
                        )
                    )

    def test_configured_beat_files_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("midi", "beats", "raw", "canonical"):
                (root / name).mkdir()
            split = root / "split.txt"
            split.write_text("complete\nmissing-beats\n", encoding="utf-8")
            for record_id in ("complete", "missing-beats"):
                (root / "midi" / f"{record_id}.mid").touch()
                (root / "canonical" / f"{record_id}.npz").touch()
            beat_path = root / "beats" / "complete.txt"
            beat_path.touch()

            pairs = []
            song = object()

            def read_pair(pair, midi_loader, hand_loader):
                del midi_loader, hand_loader
                pairs.append(pair)
                return song

            components = {
                "file_loader": SimpleNamespace(
                    reader=SimpleNamespace(read=read_pair)
                ),
                "midi_loader": object(),
                "hand_loader": object(),
            }
            config = OmegaConf.create(
                {
                    "data": {
                        "video_root": str(root / "video"),
                        "midi_root": str(root / "midi"),
                        "raw_wilor_root": str(root / "raw"),
                        "hand_root": str(root / "canonical"),
                        "beat_root": str(root / "beats"),
                    },
                    "preprocess": {
                        "video_to_ragged": {"_target_": "example.Video"},
                        "ragged_to_canonical": {
                            "_target_": "example.Canonical"
                        },
                    },
                }
            )
            canonicalizer = SimpleNamespace(run=lambda *args, **kwargs: None)

            with patch.object(
                song_input_stage,
                "instantiate",
                return_value=canonicalizer,
            ):
                rows = list(
                    song_input_stage.iter_split_songs(
                        config,
                        split,
                        overwrite=False,
                        label="test",
                        components=components,
                    )
                )

        self.assertEqual(
            rows,
            [(1, 2, song, components)],
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].record_id, "complete")
        self.assertEqual(pairs[0].beat_path, beat_path)

    def test_test_stage_prepares_split_and_dumps_nested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            bundle = root / "runtime"
            video_root = bundle / "video"
            midi_root = bundle / "midi"
            raw_root = bundle / "wilor"
            hand_root = bundle / "airpiano_wilor_2p5d_30fps"
            video_root.mkdir(parents=True)
            midi_root.mkdir()
            stems = ["artist/alpha", "take.03"]
            for stem in stems:
                video = video_root / f"{stem}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
                midi = mido.MidiFile(ticks_per_beat=480)
                track = mido.MidiTrack()
                midi.tracks.append(track)
                track.append(mido.Message("note_on", note=60, velocity=64, time=0))
                track.append(mido.Message("note_off", note=60, velocity=0, time=480))
                midi_path = midi_root / f"{stem}.mid"
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                midi.save(midi_path)
            (video_root / "video-only.mp4").touch()
            midi = mido.MidiFile()
            midi.save(midi_root / "midi-only.mid")
            split = root / "runtime.txt"
            split.write_text(
                "artist/alpha\ntake.03\nvideo-only\nmidi-only\n",
                encoding="utf-8",
            )
            checkpoint = project / "checkpoints" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint loading is patched")

            output_root = project / "inference" / "test"
            run_order: list[tuple[str, str]] = []
            test_case = self

            class CachedVideoToWiLoR:
                def run(
                    self,
                    video_path,
                    output_dir,
                    overwrite=False,
                ):
                    video = Path(video_path)
                    record_id = (
                        video.relative_to(video_root).as_posix()[: -len(".mp4")]
                    )
                    test_case.assertEqual(
                        video, video_root / f"{record_id}.mp4"
                    )
                    test_case.assertEqual(
                        Path(output_dir), raw_root / record_id
                    )
                    test_case.assertTrue(overwrite)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    (Path(output_dir) / "manifest.json").touch()
                    run_order.append(("video", record_id))
                    return Path(output_dir)

            class CanonicalWiLoR:
                def run(self, raw_record_dir, output_path, overwrite=False):
                    if not (Path(raw_record_dir) / "manifest.json").is_file():
                        raise FileNotFoundError(raw_record_dir)
                    output = Path(output_path)
                    record_id = (
                        output.relative_to(hand_root).as_posix()[: -len(".npz")]
                    )
                    test_case.assertEqual(
                        Path(raw_record_dir), raw_root / record_id
                    )
                    test_case.assertEqual(
                        output,
                        hand_root / f"{record_id}.npz",
                    )
                    test_case.assertTrue(overwrite)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        output,
                        pose=np.zeros((4, 2, 21, 3), dtype=np.float32),
                        presence=np.ones((4, 2), dtype=np.bool_),
                        fps=np.asarray(30.0, dtype=np.float64),
                        schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
                    )
                    run_order.append(("canonical", record_id))
                    return output

            config = OmegaConf.load(
                Path(__file__).parents[1] / "configs" / "baseline.yaml"
            )
            config.device = "cpu"
            config.output.dir = str(project)
            config.train.batch_size = 1
            config.train.bf16 = False
            config.test.split_file = str(split)
            config.data.video_root = str(video_root)
            config.data.midi_root = str(midi_root)
            config.data.raw_wilor_root = str(raw_root)
            config.data.hand_root = str(hand_root)
            config.dataset.file_loader.pairer._target_ = (
                "airpiano.data.files.RelativeStemPairer"
            )
            config.test.overwrite = True

            targets: list[str] = []

            def instantiate_preprocessor(node):
                target = str(node._target_)
                targets.append(target)
                if target == "airpiano.preprocessing.video.VideoToWiLoR":
                    return CachedVideoToWiLoR()
                if target == "airpiano.preprocessing.canonical.WiLoRToCanonical":
                    return CanonicalWiLoR()
                raise AssertionError(target)

            printed = io.StringIO()
            with (
                patch.object(
                    song_input_stage,
                    "instantiate",
                    side_effect=instantiate_preprocessor,
                ),
                patch.object(
                    inference,
                    "load_model_checkpoint",
                    return_value=_TinyModel(),
                ) as load_checkpoint,
                redirect_stdout(printed),
            ):
                inference.test(config)

            self.assertEqual(
                targets,
                [
                    "airpiano.preprocessing.canonical.WiLoRToCanonical",
                    "airpiano.preprocessing.video.VideoToWiLoR",
                ],
            )
            self.assertEqual(
                run_order,
                [
                    ("video", "artist/alpha"),
                    ("canonical", "artist/alpha"),
                    ("video", "take.03"),
                    ("canonical", "take.03"),
                ],
            )
            load_checkpoint.assert_called_once_with(
                checkpoint.resolve(), torch.device("cpu")
            )
            for stem in stems:
                output_dir = output_root / stem
                artifacts = sorted(output_dir.glob("*.npz"))
                self.assertEqual(len(artifacts), 1)
                self.assertEqual(
                    artifacts[0].name,
                    "window_000000__f000000000.npz",
                )
                with np.load(artifacts[0], allow_pickle=False) as payload:
                    self.assertEqual(str(payload["record_id"]), stem)
                artifact = load_artifact(artifacts[0])
                self.assertEqual(
                    artifact.probability.shape,
                    (4, 1),
                )
                self.assertEqual(int(artifact.frame_mask.sum()), 4)
                self.assertEqual(int(artifact.chord_mask.sum()), 1)
                heatmap = root / f"{stem}.png"
                sonification = root / f"{stem}.mid"
                self.assertEqual(
                    visualize_artifact(artifacts[0], heatmap), heatmap.resolve()
                )
                self.assertEqual(
                    sonify_artifact(artifacts[0], sonification),
                    sonification.resolve(),
                )
                self.assertGreater(heatmap.stat().st_size, 0)
                self.assertGreater(sonification.stat().st_size, 0)
            metrics = json.loads(
                (output_root / "test_loss.json").read_text()
            )
            self.assertGreater(metrics["test_loss"], 0.0)
            self.assertIn("video-only", printed.getvalue())
            self.assertIn("midi-only", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
