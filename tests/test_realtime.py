"""Integration test for the config-owned realtime workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import airpiano.realtime as realtime_stage
from airpiano import train
from airpiano.preprocessing.canonical import CANONICAL_SCHEMA_VERSION
from airpiano.render.artifact import load_artifact
from airpiano.render.sonify import sonify_artifact
from airpiano.render.visualize import visualize_artifact


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        batch, frames = hands.shape[:2]
        return self.bias.expand(batch, frames, chords.shape[1])


class RealtimeTest(unittest.TestCase):
    def test_composes_preprocessors_and_dumps_renderable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            bundle = project / "realtime"
            stem = Path("session.alpha") / "take.03"
            missing_video = bundle / "video" / "session.alpha" / "take.03.mp4"
            midi_path = bundle / "midi" / "session.alpha" / "take.03.mid"
            midi_path.parent.mkdir(parents=True)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.Message("note_on", note=60, velocity=64, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, time=480))
            midi.save(midi_path)
            checkpoint = project / "checkpoints" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint loading is patched")

            raw_dir = bundle / "raw_wilor" / stem
            canonical_path = bundle / "canonical" / "session.alpha" / "take.03.npz"
            output_dir = project / "inference" / "realtime" / stem
            run_order: list[str] = []
            test_case = self

            class CachedVideoToWiLoR:
                def run(
                    self,
                    video_path,
                    output_dir,
                    overwrite=False,
                ):
                    test_case.assertFalse(Path(video_path).exists())
                    test_case.assertEqual(Path(video_path), missing_video)
                    test_case.assertEqual(Path(output_dir), raw_dir)
                    test_case.assertTrue(overwrite)
                    run_order.append("video")
                    return raw_dir

            class CanonicalWiLoR:
                def run(self, raw_record_dir, output_path, overwrite=False):
                    test_case.assertEqual(Path(raw_record_dir), raw_dir)
                    output = Path(output_path)
                    test_case.assertEqual(output, canonical_path)
                    test_case.assertTrue(overwrite)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        output,
                        pose=np.zeros((4, 2, 21, 3), dtype=np.float32),
                        presence=np.ones((4, 2), dtype=np.bool_),
                        fps=np.asarray(30.0, dtype=np.float64),
                        schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
                    )
                    run_order.append("canonical")
                    return output

            config = OmegaConf.load(
                Path(__file__).parents[1] / "configs" / "baseline.yaml"
            )
            config.device = "cpu"
            config.output.dir = str(project)
            config.train.batch_size = 1
            config.train.bf16 = False
            config.dataset.file_loader.pairer._target_ = (
                "airpiano.data.files.RelativeStemPairer"
            )
            config.realtime.stem = stem.as_posix()
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

            with (
                patch.object(
                    realtime_stage,
                    "instantiate",
                    side_effect=instantiate_preprocessor,
                ),
                patch.object(
                    train,
                    "load_model_checkpoint",
                    return_value=_TinyModel(),
                ) as load_checkpoint,
            ):
                realtime_stage.realtime(config)

            self.assertEqual(
                targets,
                [
                    "airpiano.preprocessing.video.VideoToWiLoR",
                    "airpiano.preprocessing.canonical.WiLoRToCanonical",
                ],
            )
            self.assertEqual(run_order, ["video", "canonical"])
            load_checkpoint.assert_called_once_with(
                checkpoint.resolve(), torch.device("cpu")
            )
            artifacts = sorted(output_dir.glob("*.npz"))
            self.assertEqual(len(artifacts), 1)
            with np.load(artifacts[0], allow_pickle=False) as payload:
                self.assertEqual(str(payload["record_id"]), stem.as_posix())
            artifact = load_artifact(artifacts[0])
            self.assertEqual(
                artifact.probability.shape,
                (210, int(config.dataset.slicer.chord_count)),
            )
            self.assertEqual(int(artifact.frame_mask.sum()), 4)
            self.assertEqual(int(artifact.chord_mask.sum()), 1)
            heatmap = root / "realtime.png"
            sonification = root / "realtime.mid"
            self.assertEqual(
                visualize_artifact(artifacts[0], heatmap), heatmap.resolve()
            )
            self.assertEqual(
                sonify_artifact(artifacts[0], sonification),
                sonification.resolve(),
            )
            self.assertGreater(heatmap.stat().st_size, 0)
            self.assertGreater(sonification.stat().st_size, 0)
            metrics = json.loads((output_dir / "realtime_loss.json").read_text())
            self.assertGreater(metrics["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
