"""Integration test for the config-owned realtime workflow."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import airpiano.realtime as realtime_stage
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


class RealtimeTest(unittest.TestCase):
    def test_empty_root_does_not_construct_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "realtime").mkdir()
            config = OmegaConf.create(
                {
                    "output": {"dir": str(project)},
                    "realtime": {
                        "root": str(project / "realtime"),
                        "checkpoint": "checkpoints/last.pt",
                        "overwrite": False,
                    },
                }
            )
            with (
                patch.object(realtime_stage, "instantiate") as instantiate,
                patch.object(data, "build_data_components") as build_components,
            ):
                realtime_stage.realtime(config)

            instantiate.assert_not_called()
            build_components.assert_not_called()

    def test_composes_preprocessors_and_dumps_renderable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            bundle = project / "realtime"
            video_root = bundle / "video"
            midi_root = bundle / "midi"
            video_root.mkdir(parents=True)
            midi_root.mkdir()
            stems = ["alpha", "take.03"]
            for stem in stems:
                (video_root / f"{stem}.mp4").touch()
                midi = mido.MidiFile(ticks_per_beat=480)
                track = mido.MidiTrack()
                midi.tracks.append(track)
                track.append(mido.Message("note_on", note=60, velocity=64, time=0))
                track.append(mido.Message("note_off", note=60, velocity=0, time=480))
                midi.save(midi_root / f"{stem}.mid")
            (video_root / "video-only.mp4").touch()
            midi = mido.MidiFile()
            midi.save(midi_root / "midi-only.mid")
            checkpoint = project / "checkpoints" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint loading is patched")

            output_root = bundle / "realtime"
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
                    record_id = video.name[: -len(".mp4")]
                    test_case.assertEqual(
                        video, video_root / f"{record_id}.mp4"
                    )
                    test_case.assertEqual(
                        Path(output_dir), bundle / "raw_wilor" / record_id
                    )
                    test_case.assertTrue(overwrite)
                    run_order.append(("video", record_id))
                    return Path(output_dir)

            class CanonicalWiLoR:
                def run(self, raw_record_dir, output_path, overwrite=False):
                    output = Path(output_path)
                    record_id = output.name[: -len(".npz")]
                    test_case.assertEqual(
                        Path(raw_record_dir), bundle / "raw_wilor" / record_id
                    )
                    test_case.assertEqual(
                        output, bundle / "canonical" / f"{record_id}.npz"
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
            config.realtime.root = str(bundle)
            config.dataset.file_loader.pairer._target_ = (
                "airpiano.data.files.RelativeStemPairer"
            )
            config.realtime.overwrite = True

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
                    realtime_stage,
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
                realtime_stage.realtime(config)

            self.assertEqual(
                targets,
                [
                    "airpiano.preprocessing.video.VideoToWiLoR",
                    "airpiano.preprocessing.canonical.WiLoRToCanonical",
                ],
            )
            self.assertEqual(
                run_order,
                [
                    ("video", "alpha"),
                    ("canonical", "alpha"),
                    ("video", "take.03"),
                    ("canonical", "take.03"),
                ],
            )
            self.assertEqual(load_checkpoint.call_count, 2)
            for checkpoint_call in load_checkpoint.call_args_list:
                self.assertEqual(
                    checkpoint_call.args,
                    (checkpoint.resolve(), torch.device("cpu")),
                )
            for stem in stems:
                output_dir = output_root / stem
                artifacts = sorted(output_dir.glob("*.npz"))
                self.assertEqual(len(artifacts), 1)
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
                    (output_dir / "realtime_loss.json").read_text()
                )
                self.assertGreater(metrics["loss"], 0.0)
            self.assertIn("video-only", printed.getvalue())
            self.assertIn("midi-only", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
