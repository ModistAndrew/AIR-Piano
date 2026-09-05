"""Integration tests for the config-owned test inference stage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

import airpiano.inference as inference_stage
from airpiano import data


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


class TestInferenceTest(unittest.TestCase):
    def test_uses_configured_checkpoint_and_dumps_test_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint loading is patched")
            config = OmegaConf.create(
                {
                    "seed": 1,
                    "device": "cpu",
                    "output": {"dir": str(output)},
                    "data": {"splits": {}, "midi_root": ".", "hand_root": "."},
                    "dataset": {},
                    "train": {"batch_size": 1, "bf16": False},
                    "test": {
                        "checkpoint": "checkpoints/best.pt",
                        "output_dir": "inference/custom-test",
                    },
                }
            )

            with (
                patch.object(
                    data, "build_dataset", return_value=_TinyDataset()
                ) as build_dataset,
                patch.object(
                    inference_stage,
                    "load_model_checkpoint",
                    return_value=_TinyModel(),
                ) as load_checkpoint,
            ):
                inference_stage.test(config)

            build_dataset.assert_called_once_with(config, "test")
            load_checkpoint.assert_called_once_with(
                checkpoint.resolve(), torch.device("cpu")
            )
            inference_dir = output / "inference" / "custom-test"
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
