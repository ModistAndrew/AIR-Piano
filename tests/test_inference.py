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
        chords = torch.zeros(2, 456)
        chords[0, 60] = 1.0
        chords[1, 64] = 1.0
        chords[0, 256 + 10] = 1.0
        chords[1, 256 + 20] = 1.0
        return {
            "hands": torch.zeros(2, 2, 21, 3),
            "hand_presence": torch.ones(2, 2, dtype=torch.bool),
            "frame_mask": torch.ones(2, dtype=torch.bool),
            "chords": chords,
            "chord_mask": torch.ones(2, dtype=torch.bool),
            "target": torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
            "chord_ids": torch.tensor([0, 1]),
            "record_id": f"subject/song{index}",
            "frame_start": index * 2,
            "fps": 30.0,
        }


class _TinyCategoricalDataset(_TinyDataset):
    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample["target"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return sample


class _TinyModel(nn.Module):
    def __init__(
        self,
        causal_frame_attention: bool = False,
        include_chord_duration: bool = False,
        include_chord_offsets: bool = False,
        zero_hand_input: bool = False,
    ) -> None:
        super().__init__()
        self.causal_frame_attention = causal_frame_attention
        self.include_chord_duration = include_chord_duration
        self.include_chord_offsets = include_chord_offsets
        self.zero_hand_input = zero_hand_input
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, hands, hand_presence, chords, frame_mask, chord_mask):
        batch, frames = hands.shape[:2]
        return self.bias.expand(batch, frames, chords.shape[1])


class TestInferenceTest(unittest.TestCase):
    def test_checkpoint_restores_causal_mode_and_legacy_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            causal = root / "causal.pt"
            duration = root / "duration.pt"
            offsets = root / "offsets.pt"
            zero_hand = root / "zero_hand.pt"
            legacy = root / "legacy.pt"
            torch.save(
                {
                    "model": _TinyModel().state_dict(),
                    "config": {
                        "model": {"causal_frame_attention": True}
                    },
                },
                causal,
            )
            torch.save(
                {
                    "model": _TinyModel().state_dict(),
                    "config": {
                        "model": {"include_chord_duration": True}
                    },
                },
                duration,
            )
            torch.save(
                {
                    "model": _TinyModel().state_dict(),
                    "config": {"model": {"include_chord_offsets": True}},
                },
                offsets,
            )
            torch.save(
                {
                    "model": _TinyModel().state_dict(),
                    "config": {"model": {"zero_hand_input": True}},
                },
                zero_hand,
            )
            torch.save({"model": _TinyModel().state_dict()}, legacy)

            def instantiate_model(config):
                return _TinyModel(
                    causal_frame_attention=bool(
                        config.causal_frame_attention
                    ),
                    include_chord_duration=bool(
                        config.include_chord_duration
                    ),
                    include_chord_offsets=bool(
                        config.include_chord_offsets
                    ),
                    zero_hand_input=bool(config.zero_hand_input),
                )

            with patch.object(
                inference_stage,
                "instantiate",
                side_effect=instantiate_model,
            ):
                causal_model = inference_stage.load_model_checkpoint(
                    causal, torch.device("cpu")
                )
                duration_model = inference_stage.load_model_checkpoint(
                    duration, torch.device("cpu")
                )
                offset_model = inference_stage.load_model_checkpoint(
                    offsets, torch.device("cpu")
                )
                zero_hand_model = inference_stage.load_model_checkpoint(
                    zero_hand, torch.device("cpu")
                )
                legacy_model = inference_stage.load_model_checkpoint(
                    legacy, torch.device("cpu")
                )

            self.assertTrue(causal_model.causal_frame_attention)
            self.assertTrue(duration_model.include_chord_duration)
            self.assertTrue(offset_model.include_chord_offsets)
            self.assertTrue(zero_hand_model.zero_hand_input)
            self.assertFalse(legacy_model.causal_frame_attention)
            self.assertFalse(legacy_model.include_chord_duration)
            self.assertFalse(legacy_model.include_chord_offsets)
            self.assertFalse(legacy_model.zero_hand_input)

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
                        "split_file": "assets/splits/custom/test.txt",
                        "checkpoint": str(checkpoint),
                        "output_dir": str(
                            output / "inference" / "custom-test"
                        ),
                    },
                }
            )

            with (
                patch.object(
                    data,
                    "build_dataset",
                    return_value=_TinyDataset(),
                ) as build_dataset,
                patch.object(
                    inference_stage,
                    "load_model_checkpoint",
                    return_value=_TinyModel(include_chord_duration=True),
                ) as load_checkpoint,
            ):
                inference_stage.test(config)

            build_dataset.assert_called_once_with(
                config,
                "test",
                split_file="assets/splits/custom/test.txt",
            )
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

    def test_cross_entropy_inference_writes_softmax_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            loader = data.make_loader(
                _TinyCategoricalDataset(),
                batch_size=2,
                shuffle=False,
                seed=1,
                device=torch.device("cpu"),
            )

            loss = inference_stage.dump_inference(
                _TinyModel(),
                loader,
                torch.device("cpu"),
                output,
                bf16=False,
                alignment_loss="cross_entropy",
            )

            self.assertAlmostEqual(loss, float(np.log(2.0)), places=6)
            artifacts = sorted(output.glob("*.npz"))
            self.assertEqual(len(artifacts), 2)
            with np.load(artifacts[0], allow_pickle=False) as artifact:
                np.testing.assert_allclose(
                    artifact["probability"].sum(axis=-1), 1.0
                )
                np.testing.assert_allclose(artifact["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
