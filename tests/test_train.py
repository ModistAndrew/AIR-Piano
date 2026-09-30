"""Integration tests for training and checkpoint creation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from airpiano import data, train


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        chords = torch.zeros(2, 456)
        chords[:, :128] = torch.eye(128)[:2]
        chords[:, 256] = 1.0
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


class _CategoricalTinyDataset(_TinyDataset):
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
    ) -> None:
        super().__init__()
        self.causal_frame_attention = causal_frame_attention
        self.include_chord_duration = include_chord_duration
        self.include_chord_offsets = include_chord_offsets
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
        components = data.build_data_components(config)
        canonicalizer = instantiate(config.preprocess.ragged_to_canonical)
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
        video_extractor = instantiate(config.preprocess.video_to_ragged)
        self.assertEqual(type(video_extractor).__name__, "VideoToWiLoR")
        self.assertIsNone(video_extractor.target_fps)
        self.assertEqual(type(canonicalizer).__name__, "WiLoRToCanonical")
        self.assertIsNone(canonicalizer.postprocessor)
        self.assertEqual(str(config.output.dir), "outputs/baseline")
        self.assertEqual(
            str(config.data.root), "/storage/jinxuan/data/AirPianoFull"
        )
        self.assertEqual(
            str(config.data.processed_root),
            "/storage2/jinxuan/data/AirPianoFull",
        )
        self.assertEqual(str(config.data.midi_root), f"{config.data.root}/MIDI")
        self.assertEqual(str(config.data.video_root), f"{config.data.root}/Video")
        self.assertEqual(
            str(config.data.raw_wilor_root), f"{config.data.processed_root}/wilor"
        )
        self.assertEqual(
            str(config.data.hand_root),
            f"{config.data.processed_root}/airpiano_wilor_2p5d_30fps",
        )
        self.assertEqual(
            type(instantiate(config.model)).__name__, "AirPianoTransformer"
        )
        self.assertEqual(components["slicer"].mode, "frame")
        self.assertFalse(bool(config.model.causal_frame_attention))
        self.assertFalse(bool(config.model.include_chord_duration))
        self.assertFalse(bool(config.model.include_chord_offsets))
        self.assertFalse(bool(config.model.zero_hand_input))
        self.assertFalse(components["midi_loader"].include_offsets)
        self.assertEqual(
            components["midi_loader"].chord_merger.time_mode,
            "first",
        )
        self.assertEqual(float(config.dataset.random_hand_shift_seconds), 0.0)
        self.assertEqual(components["slicer"].end_padding_seconds, 0.0)
        self.assertEqual(components["slicer"].hand_context_chords, 0)
        self.assertFalse(components["slicer"].crop_to_chord_bounds)
        self.assertIsNone(components["slicer"].max_zero_hand_fraction)
        self.assertEqual(
            components["target_generator"].event_generator.delay_seconds,
            0.0,
        )
        self.assertEqual(
            float(config.sonify.chord_tolerance_seconds),
            components["midi_loader"].chord_merger.tolerance_seconds,
        )
        self.assertEqual(str(config.wandb.project), "air_piano")
        self.assertEqual(
            str(config.test.checkpoint),
            "outputs/baseline/checkpoints/last.pt",
        )
        self.assertEqual(
            str(config.realtime.checkpoint),
            "outputs/baseline/checkpoints/last.pt",
        )
        self.assertEqual(
            str(config.realtime.output_dir),
            "outputs/baseline/realtime",
        )

    def test_smoothed_config_constructs_opt_in_postprocessor(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="smoothed")

        canonicalizer = instantiate(config.preprocess.ragged_to_canonical)

        self.assertEqual(str(config.output.dir), "outputs/smoothed")
        self.assertEqual(
            str(config.data.hand_root),
            f"{config.data.processed_root}/airpiano_wilor_2p5d_smoothed_30fps",
        )
        self.assertEqual(
            type(canonicalizer.postprocessor).__name__,
            "PianoMotion10MPostprocessor",
        )
        self.assertEqual(
            type(canonicalizer.selector).__name__,
            "LowestWristHandSelector",
        )
        self.assertEqual(
            canonicalizer.selector.minimum_detector_confidence,
            0.75,
        )
        self.assertEqual(canonicalizer.postprocessor.hampel_window_frames, 20)
        self.assertEqual(canonicalizer.postprocessor.hampel_n_sigma, 3.0)
        self.assertEqual(canonicalizer.postprocessor.max_missing_run_frames, 29)
        self.assertEqual(canonicalizer.postprocessor.min_presence_run_frames, 15)
        self.assertEqual(canonicalizer.postprocessor.savgol_window_frames, 11)
        self.assertEqual(canonicalizer.postprocessor.savgol_polyorder, 3)
        self.assertEqual(canonicalizer.minimum_presence, 0.90)

    def test_lowest_v_config_changes_only_candidate_selection(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="lowest_v")

        canonicalizer = instantiate(config.preprocess.ragged_to_canonical)

        self.assertEqual(str(config.output.dir), "outputs/lowest_v")
        self.assertEqual(
            str(config.data.hand_root),
            f"{config.data.processed_root}/airpiano_wilor_2p5d_lowest_v_30fps",
        )
        self.assertEqual(
            type(canonicalizer.selector).__name__,
            "LowestWristHandSelector",
        )
        self.assertEqual(
            canonicalizer.selector.minimum_detector_confidence,
            0.0,
        )
        self.assertIsNone(canonicalizer.postprocessor)
        self.assertIsNone(canonicalizer.minimum_presence)
        self.assertEqual(canonicalizer.target_fps, 30.0)
        self.assertEqual(canonicalizer.coordinates.xy_source, "normalized_2d")
        self.assertEqual(
            canonicalizer.coordinates.z_source,
            "wrist_relative_3d",
        )

    def test_wrist_xyz10_config_constructs_ablation_components(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="wrist_xyz10")

        components = data.build_data_components(config)
        canonicalizer = instantiate(config.preprocess.ragged_to_canonical)

        self.assertEqual(str(config.output.dir), "outputs/wrist_xyz10")
        self.assertEqual(str(config.wandb.name), "wrist_xyz10")
        self.assertTrue(
            str(config.data.hand_root).endswith("/airpiano_wilor_wrist_xyz_30fps")
        )
        self.assertEqual(canonicalizer.coordinates.xy_source, "wrist_relative_3d")
        self.assertEqual(canonicalizer.coordinates.z_source, "wrist_relative_3d")
        self.assertEqual(
            components["hand_loader"].transform.offset, (0.0, 0.0, 0.0)
        )
        self.assertEqual(
            components["hand_loader"].transform.scale, (10.0, 10.0, 10.0)
        )

    def test_wrist_delta_configs_construct_camera_motion_ablation(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        config_names = (
            "wrist_xyz10_wrist_delta",
            "wrist_xyz10_wrist_delta_pianovam",
            "wrist_xyz10_causal_chord24_extended_wrist_delta",
            "wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20",
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            configurations = {
                name: compose(config_name=name) for name in config_names
            }

        for name, config in configurations.items():
            components = data.build_data_components(config)
            canonicalizer = instantiate(config.preprocess.ragged_to_canonical)
            self.assertEqual(str(config.output.dir), f"outputs/{name}")
            self.assertEqual(str(config.wandb.name), name)
            self.assertTrue(
                str(config.data.hand_root).endswith(
                    "/airpiano_wilor_wrist_xyz_delta_30fps"
                )
            )
            self.assertEqual(
                canonicalizer.coordinates.xy_source, "wrist_relative_3d"
            )
            self.assertEqual(
                canonicalizer.coordinates.z_source, "wrist_relative_3d"
            )
            self.assertEqual(
                canonicalizer.coordinates.wrist_source, "camera_3d"
            )
            self.assertEqual(
                type(canonicalizer.postprocessor).__name__,
                "WristDeltaPostprocessor",
            )
            self.assertEqual(
                components["hand_loader"].transform.scale,
                (10.0, 10.0, 10.0),
            )

        extended = configurations[
            "wrist_xyz10_causal_chord24_extended_wrist_delta"
        ]
        extended_components = data.build_data_components(extended)
        self.assertEqual(
            str(extended.data.root), "/storage/jinxuan/data/AirPianoFull"
        )
        self.assertEqual(
            {key: str(value) for key, value in extended.data.splits.items()},
            {
                "train": "assets/splits/extended/train.txt",
                "valid": "assets/splits/extended/valid.txt",
                "test": "assets/splits/extended/test.txt",
            },
        )
        self.assertTrue(bool(extended.model.causal_frame_attention))
        self.assertEqual(extended_components["slicer"].mode, "chord")
        self.assertEqual(extended_components["slicer"].segment_frames, 24)
        self.assertEqual(extended_components["slicer"].stride_frames, 12)
        self.assertFalse(extended_components["slicer"].crop_to_chord_bounds)
        self.assertIsNone(
            extended_components["slicer"].max_zero_hand_fraction
        )

        filtered = configurations[
            "wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20"
        ]
        filtered_components = data.build_data_components(filtered)
        filtered_slicer = filtered_components["slicer"]
        self.assertEqual(str(filtered.data.root), str(extended.data.root))
        self.assertEqual(
            {key: str(value) for key, value in filtered.data.splits.items()},
            {key: str(value) for key, value in extended.data.splits.items()},
        )
        self.assertTrue(bool(filtered.model.causal_frame_attention))
        self.assertEqual(filtered_slicer.mode, "chord")
        self.assertEqual(filtered_slicer.segment_frames, 24)
        self.assertEqual(filtered_slicer.stride_frames, 12)
        self.assertTrue(filtered_slicer.crop_to_chord_bounds)
        self.assertEqual(filtered_slicer.max_zero_hand_fraction, 0.2)

    def test_pianovam_config_keeps_baseline_behavior_at_30_fps(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="pianovam")

        components = data.build_data_components(config)
        canonicalizer = instantiate(config.preprocess.ragged_to_canonical)
        video_extractor = instantiate(config.preprocess.video_to_ragged)
        slicer = components["slicer"]

        self.assertEqual(str(config.output.dir), "outputs/pianovam")
        self.assertEqual(str(config.wandb.name), "pianovam")
        self.assertEqual(
            str(config.data.root), "/storage/jinxuan/data/AirPianoFull"
        )
        self.assertEqual(
            str(config.data.processed_root),
            "/storage2/jinxuan/data/AirPianoFull",
        )
        self.assertEqual(str(config.data.midi_root), f"{config.data.root}/MIDI")
        self.assertEqual(str(config.data.video_root), f"{config.data.root}/Video")
        self.assertEqual(
            str(config.data.raw_wilor_root), f"{config.data.processed_root}/wilor"
        )
        self.assertEqual(
            str(config.data.hand_root),
            f"{config.data.processed_root}/airpiano_wilor_2p5d_30fps",
        )
        self.assertEqual(
            {name: str(path) for name, path in config.data.splits.items()},
            {
                "train": "assets/splits/pianovam/train.txt",
                "valid": "assets/splits/pianovam/valid.txt",
                "test": "assets/splits/pianovam/test.txt",
            },
        )
        self.assertEqual(canonicalizer.target_fps, 30.0)
        self.assertEqual(video_extractor.target_fps, 30.0)
        self.assertEqual(slicer.fps, 30.0)
        self.assertEqual(slicer.segment_frames, 210)
        self.assertEqual(slicer.stride_frames, 105)
        self.assertEqual(slicer.chord_count, 128)
        self.assertEqual(
            components["hand_loader"].transform.offset, (0.5, 0.5, 0.0)
        )
        self.assertEqual(
            components["hand_loader"].transform.scale, (1.0, 1.0, 10.0)
        )

    def test_only_pianovam_downsamples_before_wilor(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            configurations = {
                name: compose(config_name=name)
                for name in (
                    "baseline",
                    "causal_chord24",
                    "chord_duration",
                    "full",
                    "pianovam",
                    "smoothed",
                    "wrist_xyz10",
                )
            }

        self.assertEqual(
            {
                name: config.preprocess.video_to_ragged.target_fps
                for name, config in configurations.items()
            },
            {
                "baseline": None,
                "causal_chord24": None,
                "chord_duration": None,
                "full": None,
                "pianovam": 30.0,
                "smoothed": None,
                "wrist_xyz10": None,
            },
        )

    def test_full_config_combines_training_roots_and_keeps_baseline_behavior(
        self,
    ) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="full")

        components = data.build_data_components(config)
        root = "/storage/jinxuan/data/AirPianoFull"
        processed_root = "/storage2/jinxuan/data/AirPianoFull"
        self.assertEqual(str(config.output.dir), "outputs/full")
        self.assertEqual(str(config.wandb.name), "full")
        self.assertEqual(str(config.data.root), root)
        self.assertEqual(str(config.data.processed_root), processed_root)
        self.assertEqual(str(config.data.midi_root), f"{root}/MIDI")
        self.assertEqual(
            str(config.data.hand_root),
            f"{processed_root}/airpiano_wilor_2p5d_30fps",
        )
        self.assertEqual(
            {name: str(path) for name, path in config.data.splits.items()},
            {
                "train": "assets/splits/full/train.txt",
                "valid": "assets/splits/full/valid.txt",
                "test": "assets/splits/full/test.txt",
            },
        )
        self.assertEqual(components["slicer"].fps, 30.0)
        self.assertEqual(components["slicer"].segment_frames, 210)
        self.assertEqual(components["slicer"].stride_frames, 105)
        self.assertEqual(components["slicer"].chord_count, 128)
        self.assertEqual(
            components["hand_loader"].transform.offset, (0.5, 0.5, 0.0)
        )
        self.assertEqual(
            components["hand_loader"].transform.scale, (1.0, 1.0, 10.0)
        )

    def test_causal_chord24_config_combines_both_ablation_modes(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="causal_chord24")

        slicer = data.build_data_components(config)["slicer"]
        self.assertEqual(str(config.output.dir), "outputs/causal_chord24")
        self.assertEqual(str(config.wandb.name), "causal_chord24")
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertEqual(int(config.full_song.advance_after_chords), 12)
        self.assertEqual(float(config.full_song.alignment_threshold), 0.2)
        self.assertEqual(
            str(config.full_song.output_dir),
            "outputs/causal_chord24/full_song",
        )
        self.assertEqual(
            str(config.evaluation.split_file),
            "assets/splits/pianomotion10mseg/valid.txt",
        )
        self.assertEqual(
            str(config.evaluation.output_dir),
            "outputs/causal_chord24/eval",
        )
        self.assertEqual(
            list(config.evaluation.tolerances_frames), [0, 2, 4, 8, 16]
        )
        self.assertTrue(bool(config.evaluation.remove_inference_artifacts))

    def test_wrist_chord24_random_hand_shift_config(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = "wrist_xyz10_chord24_random_hand_shift2s"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        slicer = components["slicer"]
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(
            str(config.test.checkpoint),
            f"outputs/{name}/checkpoints/last.pt",
        )
        self.assertEqual(str(config.test.output_dir), f"outputs/{name}/test")
        self.assertFalse(bool(config.model.causal_frame_attention))
        self.assertEqual(float(config.dataset.random_hand_shift_seconds), 2.0)
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (10.0, 10.0, 10.0),
        )

    def test_extended_configs_preserve_their_ablation_behavior(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        config_names = (
            "causal_chord24_extended",
            "wrist_xyz10_causal_chord24_extended",
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            configurations = {
                name: compose(config_name=name) for name in config_names
            }

        root = "/storage/jinxuan/data/AirPianoFull"
        processed_root = "/storage2/jinxuan/data/AirPianoFull"
        expected_splits = {
            "train": "assets/splits/extended/train.txt",
            "valid": "assets/splits/extended/valid.txt",
            "test": "assets/splits/extended/test.txt",
        }
        expected_hand_roots = {
            "causal_chord24_extended": (
                f"{processed_root}/airpiano_wilor_2p5d_30fps"
            ),
            "wrist_xyz10_causal_chord24_extended": (
                f"{processed_root}/airpiano_wilor_wrist_xyz_30fps"
            ),
        }
        expected_scales = {
            "causal_chord24_extended": (1.0, 1.0, 10.0),
            "wrist_xyz10_causal_chord24_extended": (10.0, 10.0, 10.0),
        }
        for name, config in configurations.items():
            components = data.build_data_components(config)
            slicer = components["slicer"]
            self.assertEqual(str(config.output.dir), f"outputs/{name}")
            self.assertEqual(str(config.wandb.name), name)
            self.assertEqual(str(config.data.root), root)
            self.assertEqual(str(config.data.processed_root), processed_root)
            self.assertEqual(str(config.data.midi_root), f"{root}/MIDI")
            self.assertEqual(str(config.data.hand_root), expected_hand_roots[name])
            self.assertEqual(
                {key: str(value) for key, value in config.data.splits.items()},
                expected_splits,
            )
            self.assertTrue(bool(config.model.causal_frame_attention))
            self.assertEqual(slicer.mode, "chord")
            self.assertEqual(slicer.segment_frames, 24)
            self.assertEqual(slicer.stride_frames, 12)
            self.assertEqual(
                components["hand_loader"].transform.scale,
                expected_scales[name],
            )

    def test_wrist_delta_filter20_quantized_and_offset_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        quantized_name = (
            "wrist_xyz10_causal_chord24_extended_wrist_delta_"
            "zero_hand_filter20_quantized"
        )
        offset_name = f"{quantized_name}_offset"
        offset_softmax_name = f"{offset_name}_last_chord_softmax"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            quantized = compose(config_name=quantized_name)
            offset = compose(config_name=offset_name)
            offset_softmax = compose(config_name=offset_softmax_name)

        components = data.build_data_components(quantized)
        offset_components = data.build_data_components(offset)
        offset_softmax_components = data.build_data_components(offset_softmax)
        reader = components["midi_loader"].reader
        slicer = components["slicer"]
        self.assertEqual(
            str(quantized.output.dir), f"outputs/{quantized_name}"
        )
        self.assertEqual(str(quantized.wandb.name), quantized_name)
        self.assertEqual(
            str(quantized.data.beat_root),
            "/storage2/jinxuan/data/AirPianoFull/beats",
        )
        self.assertTrue(reader.quantization)
        self.assertEqual(reader.subdivision, 12)
        self.assertFalse(components["midi_loader"].include_offsets)
        self.assertEqual(
            components["midi_loader"].chord_merger.time_mode,
            "first",
        )
        self.assertTrue(bool(quantized.model.include_chord_duration))
        self.assertFalse(bool(quantized.model.include_chord_offsets))
        self.assertTrue(bool(quantized.model.causal_frame_attention))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertTrue(slicer.crop_to_chord_bounds)
        self.assertEqual(slicer.max_zero_hand_fraction, 0.2)
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (10.0, 10.0, 10.0),
        )
        self.assertTrue(
            str(quantized.data.hand_root).endswith(
                "/airpiano_wilor_wrist_xyz_delta_30fps"
            )
        )
        self.assertEqual(str(offset.output.dir), f"outputs/{offset_name}")
        self.assertEqual(str(offset.wandb.name), offset_name)
        self.assertTrue(offset_components["midi_loader"].include_offsets)
        self.assertEqual(
            offset_components["midi_loader"].chord_merger.time_mode,
            "mean",
        )
        self.assertTrue(bool(offset.model.include_chord_duration))
        self.assertTrue(bool(offset.model.include_chord_offsets))
        self.assertTrue(offset_components["midi_loader"].reader.quantization)
        self.assertTrue(offset_components["slicer"].crop_to_chord_bounds)
        self.assertEqual(
            offset_components["slicer"].max_zero_hand_fraction,
            0.2,
        )
        self.assertEqual(
            str(offset_softmax.output.dir),
            f"outputs/{offset_softmax_name}",
        )
        self.assertEqual(
            str(offset_softmax.wandb.name), offset_softmax_name
        )
        self.assertEqual(
            str(offset_softmax.alignment_loss), "cross_entropy"
        )
        self.assertTrue(
            offset_softmax_components["midi_loader"].include_offsets
        )
        self.assertEqual(
            offset_softmax_components["midi_loader"].chord_merger.time_mode,
            "mean",
        )
        self.assertTrue(bool(offset_softmax.model.include_chord_duration))
        self.assertTrue(bool(offset_softmax.model.include_chord_offsets))
        self.assertTrue(
            offset_softmax_components["midi_loader"].reader.quantization
        )
        self.assertEqual(
            offset_softmax_components["target_generator"].target_mode,
            "last_chord",
        )
        self.assertTrue(
            offset_softmax_components["slicer"].crop_before_first_chord
        )
        self.assertTrue(
            offset_softmax_components["slicer"].crop_to_chord_bounds
        )
        self.assertEqual(
            offset_softmax_components["slicer"].max_zero_hand_fraction,
            0.2,
        )

    def test_pianovam_non_quantized_offset_last_chord_config(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = (
            "wrist_xyz10_causal_chord24_wrist_delta_zero_hand_filter20_"
            "offset_last_chord_softmax_pianovam"
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        midi_loader = components["midi_loader"]
        slicer = components["slicer"]
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(str(config.alignment_loss), "cross_entropy")
        self.assertEqual(
            {key: str(value) for key, value in config.data.splits.items()},
            {
                "train": "assets/splits/pianovam/train.txt",
                "valid": "assets/splits/pianovam/valid.txt",
                "test": "assets/splits/pianovam/test.txt",
            },
        )
        self.assertEqual(
            float(config.preprocess.video_to_ragged.target_fps), 30.0
        )
        self.assertNotIn("beat_root", config.data)
        self.assertFalse(midi_loader.reader.quantization)
        self.assertTrue(midi_loader.include_offsets)
        self.assertEqual(midi_loader.chord_merger.time_mode, "mean")
        self.assertFalse(bool(config.model.include_chord_duration))
        self.assertTrue(bool(config.model.include_chord_offsets))
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertEqual(
            components["target_generator"].target_mode,
            "last_chord",
        )
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertTrue(slicer.crop_before_first_chord)
        self.assertTrue(slicer.crop_to_chord_bounds)
        self.assertEqual(slicer.max_zero_hand_fraction, 0.2)
        self.assertTrue(
            str(config.data.hand_root).endswith(
                "/airpiano_wilor_wrist_xyz_delta_30fps"
            )
        )

    def test_pianovam_non_quantized_onset_target_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        offset_name = (
            "wrist_xyz10_causal_chord24_wrist_delta_zero_hand_filter20_"
            "offset_pianovam"
        )
        onset_name = (
            "wrist_xyz10_causal_chord24_wrist_delta_zero_hand_filter20_"
            "pianovam"
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            offset = compose(config_name=offset_name)
            onset = compose(config_name=onset_name)

        offset_components = data.build_data_components(offset)
        onset_components = data.build_data_components(onset)
        for name, config, components in (
            (offset_name, offset, offset_components),
            (onset_name, onset, onset_components),
        ):
            self.assertEqual(str(config.output.dir), f"outputs/{name}")
            self.assertEqual(str(config.wandb.name), name)
            self.assertEqual(str(config.alignment_loss), "bce")
            self.assertEqual(
                {
                    key: str(value)
                    for key, value in config.data.splits.items()
                },
                {
                    "train": "assets/splits/pianovam/train.txt",
                    "valid": "assets/splits/pianovam/valid.txt",
                    "test": "assets/splits/pianovam/test.txt",
                },
            )
            self.assertNotIn("beat_root", config.data)
            self.assertFalse(components["midi_loader"].reader.quantization)
            self.assertFalse(bool(config.model.include_chord_duration))
            self.assertEqual(
                components["target_generator"].target_mode,
                "onset",
            )
            self.assertTrue(components["slicer"].crop_before_first_chord)
            self.assertTrue(components["slicer"].crop_to_chord_bounds)
            self.assertEqual(
                components["slicer"].max_zero_hand_fraction,
                0.2,
            )

        self.assertTrue(offset_components["midi_loader"].include_offsets)
        self.assertEqual(
            offset_components["midi_loader"].chord_merger.time_mode,
            "mean",
        )
        self.assertTrue(bool(offset.model.include_chord_offsets))
        self.assertFalse(onset_components["midi_loader"].include_offsets)
        self.assertEqual(
            onset_components["midi_loader"].chord_merger.time_mode,
            "first",
        )
        self.assertFalse(bool(onset.model.include_chord_offsets))

    def test_wrist_delta_filter20_additional_duration_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        base_name = (
            "wrist_xyz10_causal_chord24_extended_wrist_delta_"
            "zero_hand_filter20"
        )
        normalized_name = f"{base_name}_chord_duration_mean_normalized"
        softmax_name = f"{base_name}_quantized_last_chord_softmax"
        mean_time_name = f"{base_name}_quantized_mean_time"
        names = (normalized_name, softmax_name, mean_time_name)
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            configs = [compose(config_name=name) for name in names]

        normalized, softmax, mean_time = configs
        normalized_components = data.build_data_components(normalized)
        softmax_components = data.build_data_components(softmax)
        mean_time_components = data.build_data_components(mean_time)

        for name, config, components in zip(
            names,
            configs,
            (
                normalized_components,
                softmax_components,
                mean_time_components,
            ),
            strict=True,
        ):
            slicer = components["slicer"]
            self.assertEqual(str(config.output.dir), f"outputs/{name}")
            self.assertEqual(str(config.wandb.name), name)
            self.assertTrue(bool(config.model.causal_frame_attention))
            self.assertEqual(slicer.mode, "chord")
            self.assertEqual(slicer.segment_frames, 24)
            self.assertEqual(slicer.stride_frames, 12)
            self.assertTrue(slicer.crop_to_chord_bounds)
            self.assertEqual(slicer.max_zero_hand_fraction, 0.2)
            self.assertEqual(
                components["hand_loader"].transform.scale,
                (10.0, 10.0, 10.0),
            )
            self.assertTrue(
                str(config.data.hand_root).endswith(
                    "/airpiano_wilor_wrist_xyz_delta_30fps"
                )
            )

        normalized_loader = normalized_components["midi_loader"]
        self.assertFalse(normalized_loader.reader.quantization)
        self.assertEqual(
            normalized_loader.chord_merger.duration_normalization,
            "song_mean",
        )
        self.assertIsNone(
            normalized_loader.chord_merger.duration_scale_range
        )
        self.assertEqual(normalized_loader.chord_merger.time_mode, "first")
        self.assertTrue(bool(normalized.model.include_chord_duration))

        softmax_loader = softmax_components["midi_loader"]
        self.assertTrue(softmax_loader.reader.quantization)
        self.assertEqual(softmax_loader.reader.subdivision, 12)
        self.assertTrue(bool(softmax.model.include_chord_duration))
        self.assertEqual(str(softmax.alignment_loss), "cross_entropy")
        self.assertEqual(
            softmax_components["target_generator"].target_mode,
            "last_chord",
        )
        self.assertTrue(softmax_components["slicer"].crop_before_first_chord)

        mean_time_loader = mean_time_components["midi_loader"]
        self.assertTrue(mean_time_loader.reader.quantization)
        self.assertEqual(mean_time_loader.reader.subdivision, 12)
        self.assertFalse(mean_time_loader.include_offsets)
        self.assertEqual(mean_time_loader.chord_merger.time_mode, "mean")
        self.assertTrue(bool(mean_time.model.include_chord_duration))
        self.assertFalse(bool(mean_time.model.include_chord_offsets))

    def test_mean_time_subdivision_ablation_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        base_name = (
            "wrist_xyz10_causal_chord24_extended_wrist_delta_"
            "zero_hand_filter20_quantized_mean_time"
        )
        expectations = {
            f"{base_name}_subdivision8": (8, None, "extrapolate"),
            f"{base_name}_subdivision16": (16, None, "extrapolate"),
            f"{base_name}_subdivision12_triplet_quarter": (
                12,
                (0, 3, 4, 6, 8, 9),
                "extrapolate",
            ),
            f"{base_name}_subdivision12_triplet_quarter_clamped": (
                12,
                (0, 3, 4, 6, 8, 9),
                "clamp",
            ),
        }
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_directory.resolve()),
        ):
            configs = {
                name: compose(config_name=name) for name in expectations
            }

        for name, (subdivision, offsets, bounds_mode) in expectations.items():
            with self.subTest(name=name):
                config = configs[name]
                components = data.build_data_components(config)
                midi_loader = components["midi_loader"]
                slicer = components["slicer"]

                self.assertEqual(str(config.output.dir), f"outputs/{name}")
                self.assertEqual(str(config.wandb.name), name)
                self.assertTrue(midi_loader.reader.quantization)
                self.assertEqual(midi_loader.reader.subdivision, subdivision)
                self.assertEqual(
                    midi_loader.reader.allowed_tick_offsets,
                    offsets,
                )
                self.assertEqual(
                    midi_loader.reader.out_of_bounds_mode,
                    bounds_mode,
                )
                self.assertFalse(midi_loader.include_offsets)
                self.assertEqual(midi_loader.chord_merger.time_mode, "mean")
                self.assertTrue(bool(config.model.include_chord_duration))
                self.assertFalse(bool(config.model.include_chord_offsets))
                self.assertTrue(bool(config.model.causal_frame_attention))
                self.assertEqual(slicer.mode, "chord")
                self.assertEqual(slicer.segment_frames, 24)
                self.assertEqual(slicer.stride_frames, 12)
                self.assertTrue(slicer.crop_to_chord_bounds)
                self.assertEqual(slicer.max_zero_hand_fraction, 0.2)

    def test_mean_time_note_split_and_velocity_ablation_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        base_name = (
            "wrist_xyz10_causal_chord24_extended_wrist_delta_"
            "zero_hand_filter20_quantized_mean_time"
        )
        expectations = {
            f"{base_name}_note_split": (True, False),
            f"{base_name}_note_split_velocity": (True, True),
            f"{base_name}_velocity": (False, True),
        }
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_directory.resolve()),
        ):
            configs = {
                name: compose(config_name=name) for name in expectations
            }

        for name, (split_notes, velocity_target) in expectations.items():
            with self.subTest(name=name):
                config = configs[name]
                components = data.build_data_components(config)
                midi_loader = components["midi_loader"]
                target_generator = components["target_generator"]
                slicer = components["slicer"]

                self.assertEqual(str(config.output.dir), f"outputs/{name}")
                self.assertEqual(str(config.wandb.name), name)
                self.assertTrue(midi_loader.reader.quantization)
                self.assertEqual(midi_loader.reader.subdivision, 12)
                self.assertFalse(midi_loader.include_offsets)
                self.assertEqual(midi_loader.chord_merger.time_mode, "mean")
                self.assertEqual(
                    midi_loader.chord_merger.split_notes,
                    split_notes,
                )
                self.assertEqual(
                    target_generator.velocity_target,
                    velocity_target,
                )
                self.assertTrue(bool(config.model.include_chord_duration))
                self.assertFalse(bool(config.model.include_chord_offsets))
                self.assertTrue(bool(config.model.causal_frame_attention))
                self.assertEqual(slicer.mode, "chord")
                self.assertEqual(slicer.segment_frames, 24)
                self.assertEqual(slicer.stride_frames, 12)
                self.assertTrue(slicer.crop_to_chord_bounds)
                self.assertEqual(slicer.max_zero_hand_fraction, 0.2)

    def test_extended_last_chord_softmax_config(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = "wrist_xyz10_causal_chord24_extended_last_chord_softmax"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(str(config.alignment_loss), "cross_entropy")
        self.assertEqual(
            components["target_generator"].target_mode, "last_chord"
        )
        self.assertTrue(components["slicer"].crop_before_first_chord)
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertEqual(components["slicer"].mode, "chord")
        self.assertEqual(components["slicer"].segment_frames, 24)
        self.assertEqual(components["slicer"].stride_frames, 12)
        self.assertEqual(
            str(config.data.root), "/storage/jinxuan/data/AirPianoFull"
        )

    def test_wrist_chord_zero_hand_and_delayed_target_configs(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        names = (
            "wrist_xyz10_causal_chord24_zero_hand",
            "wrist_xyz10_causal_chord24_target_delay1s",
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            zero_hand, delayed = [
                compose(config_name=name) for name in names
            ]

        for name, config in zip(names, (zero_hand, delayed), strict=True):
            components = data.build_data_components(config)
            self.assertEqual(str(config.output.dir), f"outputs/{name}")
            self.assertEqual(str(config.wandb.name), name)
            self.assertEqual(
                str(config.test.checkpoint),
                f"outputs/{name}/checkpoints/last.pt",
            )
            self.assertEqual(str(config.test.output_dir), f"outputs/{name}/test")
            self.assertEqual(
                str(config.realtime.output_dir),
                f"outputs/{name}/realtime",
            )
            self.assertEqual(
                str(config.full_song.output_dir),
                f"outputs/{name}/full_song",
            )
            self.assertTrue(bool(config.model.causal_frame_attention))
            self.assertEqual(components["slicer"].mode, "chord")
            self.assertEqual(components["slicer"].segment_frames, 24)
            self.assertEqual(components["slicer"].stride_frames, 12)
            self.assertEqual(
                components["hand_loader"].transform.scale,
                (10.0, 10.0, 10.0),
            )

        self.assertTrue(bool(zero_hand.model.zero_hand_input))
        self.assertEqual(
            data.build_data_components(zero_hand)["slicer"].end_padding_seconds,
            0.0,
        )
        delayed_components = data.build_data_components(delayed)
        self.assertFalse(bool(delayed.model.zero_hand_input))
        self.assertEqual(float(delayed.target_delay_seconds), 1.0)
        self.assertEqual(
            delayed_components["slicer"].end_padding_seconds,
            1.0,
        )
        self.assertEqual(
            delayed_components[
                "target_generator"
            ].event_generator.delay_seconds,
            1.0,
        )

    def test_wrist_chord_stride1_config_changes_only_window_hop(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = "wrist_xyz10_causal_chord24_stride1"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        slicer = components["slicer"]
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(
            str(config.test.checkpoint),
            f"outputs/{name}/checkpoints/last.pt",
        )
        self.assertEqual(str(config.test.output_dir), f"outputs/{name}/test")
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertFalse(bool(config.model.zero_hand_input))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 1)
        self.assertEqual(int(config.full_song.advance_after_chords), 12)
        self.assertEqual(int(config.train.epochs), 20)
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (10.0, 10.0, 10.0),
        )
        self.assertEqual(
            components["target_generator"].event_generator.delay_seconds,
            0.0,
        )

    def test_wrist_chord_hand_context1_config_expands_only_hands(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = "wrist_xyz10_causal_chord24_hand_context1"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        slicer = components["slicer"]
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(
            str(config.test.checkpoint),
            f"outputs/{name}/checkpoints/last.pt",
        )
        self.assertEqual(str(config.test.output_dir), f"outputs/{name}/test")
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertFalse(bool(config.model.zero_hand_input))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertEqual(slicer.hand_context_chords, 1)
        self.assertEqual(slicer.end_padding_seconds, 0.0)
        self.assertEqual(int(config.full_song.advance_after_chords), 12)
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (10.0, 10.0, 10.0),
        )
        self.assertEqual(
            components["target_generator"].event_generator.delay_seconds,
            0.0,
        )

    def test_chord_duration_config_enables_duration_ablation(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name="chord_duration")

        slicer = data.build_data_components(config)["slicer"]
        self.assertEqual(str(config.output.dir), "outputs/chord_duration")
        self.assertEqual(str(config.wandb.name), "chord_duration")
        self.assertTrue(bool(config.model.include_chord_duration))
        self.assertEqual(slicer.mode, "frame")

    def test_relative_duration_config_combines_selected_ablations(self) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(
                config_name=(
                    "wrist_xyz10_causal_chord24_chord_duration_relative"
                )
            )

        components = data.build_data_components(config)
        merger = components["midi_loader"].chord_merger
        slicer = components["slicer"]
        self.assertEqual(
            str(config.output.dir),
            "outputs/wrist_xyz10_causal_chord24_chord_duration_relative",
        )
        self.assertEqual(merger.duration_scale_range, (0.5, 2.0))
        self.assertTrue(bool(config.model.include_chord_duration))
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (10.0, 10.0, 10.0),
        )

    def test_mean_normalized_duration_config_uses_extended_baseline_data(
        self,
    ) -> None:
        config_directory = Path(__file__).parents[1] / "configs"
        name = "causal_chord24_extended_chord_duration_mean_normalized"
        with initialize_config_dir(
            version_base=None, config_dir=str(config_directory.resolve())
        ):
            config = compose(config_name=name)

        components = data.build_data_components(config)
        merger = components["midi_loader"].chord_merger
        slicer = components["slicer"]
        root = "/storage/jinxuan/data/AirPianoFull"
        processed_root = "/storage2/jinxuan/data/AirPianoFull"
        self.assertEqual(str(config.output.dir), f"outputs/{name}")
        self.assertEqual(str(config.wandb.name), name)
        self.assertEqual(str(config.data.root), root)
        self.assertEqual(str(config.data.processed_root), processed_root)
        self.assertEqual(str(config.data.midi_root), f"{root}/MIDI")
        self.assertEqual(
            str(config.data.hand_root),
            f"{processed_root}/airpiano_wilor_2p5d_30fps",
        )
        self.assertEqual(
            {key: str(value) for key, value in config.data.splits.items()},
            {
                "train": "assets/splits/extended/train.txt",
                "valid": "assets/splits/extended/valid.txt",
                "test": "assets/splits/extended/test.txt",
            },
        )
        self.assertTrue(bool(config.model.include_chord_duration))
        self.assertTrue(bool(config.model.causal_frame_attention))
        self.assertEqual(slicer.mode, "chord")
        self.assertEqual(slicer.segment_frames, 24)
        self.assertEqual(slicer.stride_frames, 12)
        self.assertEqual(
            components["hand_loader"].transform.offset,
            (0.5, 0.5, 0.0),
        )
        self.assertEqual(
            components["hand_loader"].transform.scale,
            (1.0, 1.0, 10.0),
        )
        self.assertEqual(merger.duration_normalization, "song_mean")
        self.assertIsNone(merger.duration_scale_range)

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
                "model": {
                    "causal_frame_attention": False,
                    "include_chord_duration": False,
                    "include_chord_offsets": False,
                    "zero_hand_input": False,
                },
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
                    "dir": str(output / "logs"),
                },
            }
        )

    def test_train_logs_losses_and_writes_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fake_run = _FakeRun()
            dataset = _TinyDataset()
            with (
                patch.object(wandb, "init", return_value=fake_run) as wandb_init,
                patch.object(data, "build_data_components", return_value={}),
                patch.object(
                    data, "build_dataset", return_value=dataset
                ) as build_dataset,
                patch.object(
                    train, "instantiate", return_value=_TinyModel()
                ) as instantiate_model,
            ):
                train.train(self._config(output))

            self.assertEqual(
                [next(iter(item)) for item in fake_run.logs],
                ["train/loss", "valid/loss"],
            )
            self.assertEqual(
                build_dataset.call_args_list,
                [
                    call(self._config(output), "train", components={}),
                    call(self._config(output), "valid", components={}),
                ],
            )
            self.assertTrue(fake_run.finished)
            instantiate_model.assert_called_once()
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
            self.assertFalse(
                checkpoint["config"]["model"]["causal_frame_attention"]
            )
            self.assertFalse(
                checkpoint["config"]["model"]["include_chord_duration"]
            )
            self.assertFalse(
                checkpoint["config"]["model"]["include_chord_offsets"]
            )
            self.assertFalse(
                checkpoint["config"]["model"]["zero_hand_input"]
            )
            self.assertFalse((output / "inference" / "test").exists())

    def test_evaluate_supports_cross_entropy(self) -> None:
        loader = data.make_loader(
            _CategoricalTinyDataset(),
            batch_size=2,
            shuffle=False,
            seed=1,
            device=torch.device("cpu"),
        )

        loss = train.evaluate(
            _TinyModel(),
            loader,
            torch.device("cpu"),
            bf16=False,
            alignment_loss="cross_entropy",
        )

        self.assertAlmostEqual(loss, float(torch.log(torch.tensor(2.0))))


if __name__ == "__main__":
    unittest.main()
