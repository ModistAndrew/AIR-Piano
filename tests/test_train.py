"""Integration tests for training and test-set inference artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from airpiano import train


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "hands": torch.zeros(2, 2, 21, 3),
            "hand_presence": torch.ones(2, 2, dtype=torch.bool),
            "frame_mask": torch.ones(2, dtype=torch.bool),
            "chords": torch.eye(128)[:2],
            "chord_mask": torch.ones(2, dtype=torch.bool),
            "target": torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
            "chord_ids": torch.tensor([0, 1]),
            "record_id": f"subject/song{index}",
            "frame_start": index * 2,
            "fps": 30.0,
        }


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        batch, frames = hands.shape[:2]
        return self.bias.expand(batch, frames, chords.shape[1])


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, float]] = []
        self.finished = False

    def log(self, values) -> None:
        self.logs.append(dict(values))

    def finish(self) -> None:
        self.finished = True


class TrainTest(unittest.TestCase):
    def test_baseline_constructs_every_configured_component(self) -> None:
        config = OmegaConf.load(Path(__file__).parents[1] / "configs" / "baseline.yaml")
        components = train.build_data_components(config)
        self.assertEqual(
            {name: type(value).__name__ for name, value in components.items()},
            {
                "file_loader": "FileLoader",
                "midi_loader": "MidiLoader",
                "hand_loader": "HandLoader",
                "slicer": "FutureChordSlicer",
                "target_generator": "TargetGenerator",
            },
        )
        self.assertEqual(
            type(instantiate(config.preprocess.video_to_ragged)).__name__,
            "VideoToWiLoR",
        )
        self.assertEqual(
            type(instantiate(config.preprocess.ragged_to_canonical)).__name__,
            "WiLoRToCanonical",
        )
        self.assertEqual(str(config.output.dir), "outputs/baseline")
        self.assertEqual(str(config.wandb.project), "air_piano")
        self.assertEqual(str(config.realtime.checkpoint), "checkpoints/last.pt")

    @staticmethod
    def _config(output: Path):
        return OmegaConf.create(
            {
                "seed": 1,
                "device": "cpu",
                "output": {
                    "root": str(output.parent),
                    "project_name": output.name,
                    "dir": str(output),
                },
                "dataset": {},
                "data": {"splits": {}, "midi_root": ".", "hand_root": "."},
                "train": {
                    "epochs": 1,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "warmup_fraction": 0.05,
                    "bf16": False,
                },
                "wandb": {
                    "project": "air_piano",
                    "name": "baseline",
                    "entity": None,
                    "dir": "logs",
                },
            }
        )

    def test_train_logs_three_loss_series_and_dumps_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fake_run = _FakeRun()
            dataset = _TinyDataset()
            with (
                patch.object(wandb, "init", return_value=fake_run) as wandb_init,
                patch.object(train, "build_data_components", return_value={}),
                patch.object(train, "build_dataset", return_value=dataset),
                patch.object(train, "AirPianoTransformer", _TinyModel),
                patch.object(
                    train,
                    "load_model_checkpoint",
                    wraps=train.load_model_checkpoint,
                ) as load_checkpoint,
            ):
                train.train(self._config(output))

            self.assertEqual(
                [next(iter(item)) for item in fake_run.logs],
                ["train/loss", "valid/loss", "test/loss"],
            )
            self.assertTrue(fake_run.finished)
            wandb_init.assert_called_once()
            self.assertEqual(wandb_init.call_args.kwargs["project"], "air_piano")
            self.assertEqual(wandb_init.call_args.kwargs["name"], "baseline")
            self.assertEqual(
                Path(wandb_init.call_args.kwargs["dir"]),
                (output / "logs").resolve(),
            )
            checkpoint_dir = output / "checkpoints"
            self.assertTrue((checkpoint_dir / "best.pt").is_file())
            self.assertTrue((checkpoint_dir / "last.pt").is_file())
            load_checkpoint.assert_called_once_with(
                (checkpoint_dir / "last.pt").resolve(), torch.device("cpu")
            )
            checkpoint = torch.load(
                checkpoint_dir / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertTrue(
                {
                    "model",
                    "optimizer",
                    "scheduler",
                    "epoch",
                    "global_step",
                    "best_valid_loss",
                    "config",
                }.issubset(checkpoint)
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["global_step"], 1)
            inference_dir = output / "inference" / "test"
            metrics = json.loads((inference_dir / "test_loss.json").read_text())
            self.assertGreater(metrics["test_loss"], 0.0)
            artifacts = sorted(inference_dir.glob("*.npz"))
            self.assertEqual(len(artifacts), 2)
            with np.load(artifacts[0], allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema_version"]), "airpiano-inference-v1"
                )
                self.assertEqual(artifact["probability"].shape, (2, 2))
                self.assertEqual(artifact["chord_pitches"].shape, (2, 128))
                self.assertEqual(artifact["chord_pitches"].dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
