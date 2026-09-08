"""Focused tests for the in-tree, resumable WiLoR integration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from airpiano.preprocessing import video as video_module
from airpiano.preprocessing.wilor import backend as backend_module
from airpiano.preprocessing.wilor import cache as cache_module


def _empty_chunk(
    indices: list[int], timestamps: list[float]
) -> dict[str, np.ndarray]:
    """Build the complete chunk schema for frames with no detections."""

    arrays = backend_module.empty_detection_arrays()
    arrays.update(
        frame_indices=np.asarray(indices, dtype=np.int64),
        frame_timestamps_seconds=np.asarray(timestamps, dtype=np.float64),
        frame_detection_offsets=np.zeros(len(indices) + 1, dtype=np.int64),
    )
    return arrays


class _Reader:
    """Small sequential reader supporting both decoding and resume grabs."""

    def __init__(self, frame_count: int, fps: float = 30.0) -> None:
        self.frames = [
            np.full((4, 6, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.fps = fps
        self.position = 0
        self.grabbed = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position == len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def grab(self) -> bool:
        if self.position == len(self.frames):
            return False
        self.position += 1
        self.grabbed += 1
        return True

    def timestamp_seconds(self) -> float:
        return (self.position - 1) / self.fps

    def release(self) -> None:
        self.released = True


class _Backend:
    """Backend protocol fake that emits valid zero-detection chunks."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[list[int]] = []
        self.frame_values: list[list[int]] = []
        self.timestamps: list[list[float]] = []

    def model_metadata(self) -> dict[str, str]:
        return {"model": "fake"}

    def inference_metadata(self, chunk_frames: int) -> dict[str, int]:
        return {"chunk_frames": chunk_frames}

    def build_chunk(self, indices, timestamps, frames):
        self.calls.append(list(indices))
        self.frame_values.append([int(frame[0, 0, 0]) for frame in frames])
        self.timestamps.append(list(timestamps))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return _empty_chunk(list(indices), list(timestamps))


def _video_metadata(frame_count: int, fps: float = 30.0) -> dict[str, object]:
    return {
        "codec_name": "h264",
        "width": 6,
        "height": 4,
        "fps": fps,
        "duration_seconds": frame_count / fps,
    }


class _Node(dict):
    """Attribute-access mapping matching the upstream configuration API."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class VideoComponentTest(unittest.TestCase):
    def test_complete_cache_is_reused_without_source_or_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "nested" / "cache"

            class LazyBackend:
                model_accessed = False

                @property
                def models(self):
                    self.model_accessed = True
                    raise AssertionError("model loaded")

            lazy_backend = LazyBackend()

            with (
                patch.object(
                    video_module,
                    "WiLoRBackend",
                    return_value=lazy_backend,
                ),
                patch.object(
                    video_module,
                    "completed_output_is_valid",
                    return_value=True,
                ),
                patch.object(video_module, "process_video") as process,
            ):
                extractor = video_module.VideoToWiLoR(
                    wilor_repo=root / "upstream",
                    device="cpu",
                    target_fps=30.0,
                )
                result = extractor.run(root / "deleted.mp4", output)

            self.assertEqual(result, output.resolve())
            self.assertFalse(lazy_backend.model_accessed)
            process.assert_not_called()

    def test_legacy_cache_is_rejected_and_overwrite_regenerates_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"
            output.mkdir()
            (output / "stale").touch()
            (output / "_SUCCESS").touch()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "pianomotion-wilor-v1",
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            fake_backend = _Backend()
            reader = _Reader(1)

            with patch.object(
                video_module, "WiLoRBackend", return_value=fake_backend
            ):
                extractor = video_module.VideoToWiLoR(
                    wilor_repo=root / "upstream",
                    device="cpu",
                    chunk_frames=2,
                )

            with self.assertRaisesRegex(
                ValueError,
                "airpiano-wilor-ragged-v1.*pianomotion-wilor-v1",
            ):
                extractor.run(video, output)

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(1),
                ),
                patch.object(
                    cache_module,
                    "open_video_reader",
                    return_value=reader,
                ),
            ):
                result = extractor.run(video, output, overwrite=True)

            self.assertEqual(result, output.resolve())
            self.assertFalse((output / "stale").exists())
            self.assertTrue(reader.released)
            self.assertTrue(cache_module.completed_output_is_valid(output))
            manifest = cache_module.read_manifest(output)
            self.assertEqual(manifest["schema_version"], cache_module.SCHEMA_VERSION)
            self.assertTrue(
                {"subject_id", "video_id", "source_url"}.isdisjoint(manifest)
            )


class CacheTest(unittest.TestCase):
    def test_ffprobe_preserves_fractional_source_fps(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24000/1001",
                    "r_frame_rate": "24/1",
                    "nb_frames": "120",
                    "duration": "5.005",
                    "time_base": "1/24000",
                }
            ],
            "format": {
                "duration": "5.005",
                "format_name": "mov,mp4",
                "size": "1234",
            },
        }
        completed = types.SimpleNamespace(stdout=json.dumps(payload))

        with patch.object(
            cache_module.subprocess,
            "run",
            return_value=completed,
        ) as run:
            metadata = cache_module.probe_video(Path("take.03.mp4"))

        self.assertAlmostEqual(metadata["fps"], 24000 / 1001)
        self.assertEqual(metadata["declared_frame_count"], 120)
        self.assertEqual(metadata["size_bytes"], 1234)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], "ffprobe")

    def test_first_due_reader_selects_fractional_rate_source_frames(self) -> None:
        native = _Reader(13, fps=60.0)
        reader = cache_module.FirstDueVideoReader(native, 60.0, 25.0)
        values = []
        timestamps = []

        while True:
            ok, frame = reader.read()
            if not ok:
                break
            values.append(int(frame[0, 0, 0]))
            timestamps.append(reader.timestamp_seconds())
        reader.release()

        self.assertEqual(values, [0, 3, 5, 8, 10, 12])
        np.testing.assert_allclose(
            timestamps, np.asarray(values, dtype=np.float64) / 60.0
        )
        self.assertTrue(native.released)

    def test_60_to_30_processing_halves_backend_inputs_and_records_rates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"
            reader = _Reader(6, fps=60.0)
            backend = _Backend()

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(6, fps=60.0),
                ),
                patch.object(
                    cache_module,
                    "OpenCVVideoReader",
                    return_value=reader,
                ),
            ):
                cache_module.process_video(
                    video,
                    output,
                    backend,
                    chunk_frames=8,
                    target_fps=30.0,
                )

            self.assertEqual(backend.calls, [[0, 1, 2]])
            self.assertEqual(backend.frame_values, [[0, 2, 4]])
            np.testing.assert_allclose(backend.timestamps, [[0.0, 2 / 60, 4 / 60]])
            manifest = cache_module.read_manifest(output)
            self.assertEqual(manifest["video"]["fps"], 30.0)
            self.assertEqual(manifest["video"]["source_fps"], 60.0)
            self.assertEqual(
                manifest["video"]["sampling_policy"],
                cache_module.FIRST_DUE_SAMPLING_POLICY,
            )
            self.assertEqual(manifest["decoded_frame_count"], 3)
            self.assertTrue(cache_module.completed_output_is_valid(output))

    def test_rate_validation_and_equal_rate_native_metadata(self) -> None:
        source = _video_metadata(6, fps=60.0)

        self.assertEqual(cache_module.video_metadata_for_cache(source, None), source)
        self.assertEqual(cache_module.video_metadata_for_cache(source, 60.0), source)
        with self.assertRaisesRegex(ValueError, "positive"):
            cache_module.video_metadata_for_cache(source, 0.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            cache_module.video_metadata_for_cache(source, 60.1)

    def test_completed_native_cache_is_reused_for_a_new_target_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(4, fps=60.0),
                ) as probe,
                patch.object(
                    cache_module,
                    "open_video_reader",
                    return_value=_Reader(4, fps=60.0),
                ),
            ):
                cache_module.process_video(video, output, _Backend(), chunk_frames=4)
                cache_module.process_video(
                    video,
                    output,
                    _Backend(),
                    chunk_frames=4,
                    target_fps=30.0,
                )

            self.assertEqual(probe.call_count, 1)
            manifest = cache_module.read_manifest(output)
            self.assertEqual(manifest["video"]["fps"], 60.0)
            self.assertNotIn("source_fps", manifest["video"])
            self.assertNotIn("sampling_policy", manifest["video"])

    def test_partial_cache_refuses_a_different_requested_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"
            reader = _Reader(6, fps=60.0)

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(6, fps=60.0),
                ),
                patch.object(
                    cache_module,
                    "open_video_reader",
                    return_value=reader,
                ) as open_reader,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    cache_module.process_video(
                        video,
                        output,
                        _Backend(fail_on_call=2),
                        chunk_frames=2,
                    )
                with self.assertRaisesRegex(RuntimeError, "different frame-rate"):
                    cache_module.process_video(
                        video,
                        output,
                        _Backend(),
                        chunk_frames=2,
                        target_fps=30.0,
                    )

            self.assertEqual(open_reader.call_count, 1)
            self.assertEqual(cache_module.read_manifest(output)["video"]["fps"], 60.0)

    def test_downsampled_partial_cache_resumes_on_the_same_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"
            first_reader = _Reader(10, fps=60.0)
            resumed_reader = _Reader(10, fps=60.0)
            interrupted = _Backend(fail_on_call=2)
            resumed = _Backend()

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(10, fps=60.0),
                ),
                patch.object(
                    cache_module,
                    "OpenCVVideoReader",
                    side_effect=[first_reader, resumed_reader],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    cache_module.process_video(
                        video,
                        output,
                        interrupted,
                        chunk_frames=2,
                        target_fps=30.0,
                    )
                cache_module.process_video(
                    video,
                    output,
                    resumed,
                    chunk_frames=2,
                    target_fps=30.0,
                )

            self.assertEqual(resumed.calls, [[2, 3], [4]])
            self.assertEqual(resumed.frame_values, [[4, 6], [8]])
            self.assertTrue(cache_module.completed_output_is_valid(output))

    def test_interrupted_cache_resumes_and_publishes_partial_final_chunk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "raw"
            first_reader = _Reader(5)
            resumed_reader = _Reader(5)
            interrupted = _Backend(fail_on_call=2)
            resumed = _Backend()

            with (
                patch.object(
                    cache_module,
                    "probe_video",
                    return_value=_video_metadata(5),
                ),
                patch.object(
                    cache_module,
                    "open_video_reader",
                    side_effect=[first_reader, resumed_reader],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    cache_module.process_video(
                        video,
                        output,
                        interrupted,
                        chunk_frames=2,
                    )

                processing = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(processing["status"], "processing")
                self.assertEqual(len(processing["chunks"]), 1)
                self.assertFalse((output / "_SUCCESS").exists())
                self.assertTrue(first_reader.released)

                result = cache_module.process_video(
                    video,
                    output,
                    resumed,
                    chunk_frames=2,
                )

            self.assertEqual(result, output.resolve())
            self.assertEqual(resumed_reader.grabbed, 2)
            self.assertTrue(resumed_reader.released)
            self.assertEqual(resumed.calls, [[2, 3], [4]])
            self.assertTrue(cache_module.completed_output_is_valid(output))

            manifest = cache_module.read_manifest(output)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["decoded_frame_count"], 5)
            self.assertEqual(manifest["detection_count"], 0)
            self.assertEqual(
                [chunk["frame_end"] - chunk["frame_start"] for chunk in manifest["chunks"]],
                [2, 2, 1],
            )
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(list(output.rglob("*.tmp")), [])

            chunk_paths = sorted((output / "chunks").glob("*.npz"))
            self.assertEqual(len(chunk_paths), 3)
            for expected_frames, chunk_path in zip([2, 2, 1], chunk_paths):
                with np.load(chunk_path, allow_pickle=False) as payload:
                    self.assertEqual(len(payload["frame_indices"]), expected_frames)
                    np.testing.assert_array_equal(
                        payload["frame_detection_offsets"],
                        np.zeros(expected_frames + 1, dtype=np.int64),
                    )
                    self.assertEqual(payload["pred_keypoints_3d"].shape, (0, 21, 3))

    def test_backend_keeps_offsets_for_zero_detection_frame_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Detector:
                def __init__(self) -> None:
                    self.batch_sizes: list[int] = []

                def __call__(self, frames, **kwargs):
                    self.batch_sizes.append(len(frames))
                    self.assert_detector_arguments(kwargs)
                    return [types.SimpleNamespace(boxes=None) for _ in frames]

                @staticmethod
                def assert_detector_arguments(kwargs) -> None:
                    if kwargs != {"conf": 0.3, "verbose": False}:
                        raise AssertionError(kwargs)

            detector = Detector()
            backend = backend_module.WiLoRBackend(
                wilor_repo=root,
                checkpoint=root / "model.ckpt",
                model_config=root / "model.yaml",
                detector_checkpoint=root / "detector.pt",
                mano_model=root / "MANO_RIGHT.pkl",
                mano_mean_params=root / "mean.npz",
                device="cpu",
                frame_batch_size=2,
            )
            backend._models = backend_module.Models(
                model=None,
                model_cfg=None,
                detector=detector,
                device=None,
                torch=None,
                fast=False,
            )
            frames = [np.zeros((4, 6, 3), np.uint8) for _ in range(5)]

            chunk = backend.build_chunk(
                list(range(5)),
                [index / 30 for index in range(5)],
                frames,
            )

            self.assertEqual(detector.batch_sizes, [2, 2, 1])
            np.testing.assert_array_equal(
                chunk["frame_detection_offsets"], np.zeros(6, dtype=np.int64)
            )
            self.assertEqual(chunk["boxes_xyxy"].shape, (0, 4))
            self.assertEqual(chunk["pred_hand_pose"].shape, (0, 15, 3, 3))


class BackendTest(unittest.TestCase):
    def test_headless_models_load_once_and_upstream_path_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "WiLoR"
            repository.mkdir()
            cfg = _Node(
                MODEL=_Node(
                    BACKBONE=_Node(TYPE="vit", PRETRAINED_WEIGHTS="unused")
                ),
                MANO=_Node(),
            )
            cfg.defrost = Mock()
            cfg.freeze = Mock()
            load_calls: list[tuple[str, dict[str, object]]] = []

            class FakeModel:
                def __init__(self) -> None:
                    self.devices = []
                    self.evaluated = False

                def to(self, device):
                    self.devices.append(device)
                    return self

                def eval(self):
                    self.evaluated = True
                    return self

            model = FakeModel()

            class FakeWiLoR:
                @staticmethod
                def load_from_checkpoint(path, **kwargs):
                    load_calls.append((path, kwargs))
                    return model

            class FakeYOLO:
                instances = []

                def __init__(self, path):
                    self.path = path
                    self.devices = []
                    self.instances.append(self)

                def to(self, device):
                    self.devices.append(device)
                    return self

            def get_config(path, update_cachedir):
                self.assertEqual(path, str((root / "model.yaml").resolve()))
                self.assertFalse(update_cachedir)
                self.assertEqual(sys.path[0], str(repository.resolve()))
                return cfg

            modules = {
                "ultralytics": types.ModuleType("ultralytics"),
                "wilor": types.ModuleType("wilor"),
                "wilor.configs": types.ModuleType("wilor.configs"),
                "wilor.models": types.ModuleType("wilor.models"),
            }
            modules["ultralytics"].YOLO = FakeYOLO
            modules["wilor"].__path__ = []
            modules["wilor.configs"].get_config = get_config
            modules["wilor.models"].WiLoR = FakeWiLoR

            backend = backend_module.WiLoRBackend(
                wilor_repo=repository,
                checkpoint=root / "model.ckpt",
                model_config=root / "model.yaml",
                detector_checkpoint=root / "detector.pt",
                mano_model=root / "mano" / "MANO_RIGHT.pkl",
                mano_mean_params=root / "mean.npz",
                device="cpu",
            )
            before = list(sys.path)
            with patch.dict(sys.modules, modules):
                first = backend.models
                second = backend.models

            self.assertIs(first, second)
            self.assertEqual(sys.path, before)
            self.assertEqual(len(load_calls), 1)
            _, arguments = load_calls[0]
            self.assertIs(arguments["cfg"], cfg)
            self.assertFalse(arguments["strict"])
            self.assertFalse(arguments["init_renderer"])
            self.assertNotIn("PRETRAINED_WEIGHTS", cfg.MODEL.BACKBONE)
            self.assertEqual(cfg.MODEL.BBOX_SHAPE, [192, 256])
            self.assertEqual(cfg.MANO.DATA_DIR, str((root / "mano").resolve()))
            self.assertEqual(cfg.MANO.MODEL_PATH, str((root / "mano").resolve()))
            self.assertEqual(cfg.MANO.MEAN_PARAMS, str((root / "mean.npz").resolve()))
            self.assertTrue(model.evaluated)
            self.assertEqual(len(FakeYOLO.instances), 1)

    def test_repository_path_cleanup_survives_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            before = list(sys.path)

            with self.assertRaisesRegex(RuntimeError, "inside repository"):
                with backend_module.repository_on_path(repository):
                    self.assertEqual(sys.path[0], str(repository))
                    raise RuntimeError("inside repository")

            self.assertEqual(sys.path, before)

    def test_crop_shape_normalization_flip_and_antialias_path(self) -> None:
        cfg = _Node(
            MODEL=_Node(
                IMAGE_SIZE=8,
                IMAGE_MEAN=[0.0, 0.0, 0.0],
                IMAGE_STD=[1.0, 1.0, 1.0],
                BBOX_SHAPE=None,
            )
        )
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        boxes = np.asarray([[0, 0, 40, 40]], dtype=np.float32)
        calls: dict[str, object] = {}

        def expand_to_aspect_ratio(size, target_aspect_ratio):
            self.assertIsNone(target_aspect_ratio)
            return np.asarray(size)

        def generate_image_patch_cv2(
            source,
            center_x,
            center_y,
            width,
            height,
            output_width,
            output_height,
            flip,
            scale,
            rotation,
            border_mode,
        ):
            del source, scale, rotation, border_mode
            calls.update(
                center=(center_x, center_y),
                size=(width, height),
                output=(output_width, output_height),
                flip=bool(flip),
            )
            patch = np.empty((output_height, output_width, 3), dtype=np.float32)
            patch[:] = [10.0, 20.0, 30.0]
            return patch, np.eye(3)

        utils = types.ModuleType("wilor.datasets.utils")
        utils.expand_to_aspect_ratio = expand_to_aspect_ratio
        utils.generate_image_patch_cv2 = generate_image_patch_cv2
        utils.convert_cvimg_to_tensor = lambda value: value.transpose(2, 0, 1)
        wilor = types.ModuleType("wilor")
        datasets = types.ModuleType("wilor.datasets")
        wilor.__path__ = []
        datasets.__path__ = []
        blurred = Mock(side_effect=lambda value, **kwargs: value)

        dataset = backend_module.WiLoRCropDataset(
            cfg,
            image,
            boxes,
            np.asarray([0], dtype=np.float32),
            rescale_factor=1.0,
            fp16=False,
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "wilor": wilor,
                    "wilor.datasets": datasets,
                    "wilor.datasets.utils": utils,
                },
            ),
            patch.object(backend_module.cv2, "GaussianBlur", blurred),
        ):
            item = dataset[0]

        self.assertEqual(item["img"].shape, (3, 8, 8))
        np.testing.assert_allclose(item["img"][:, 0, 0], [30 / 255, 20 / 255, 10 / 255])
        np.testing.assert_allclose(item["box_center"], [20, 20])
        np.testing.assert_allclose(item["img_size"], [60, 40])
        self.assertEqual(calls["center"], (20.0, 20.0))
        self.assertEqual(calls["size"], (40.0, 40.0))
        self.assertEqual(calls["output"], (8, 8))
        self.assertTrue(calls["flip"])
        blurred.assert_called_once()
        self.assertEqual(blurred.call_args.kwargs["ksize"], (0, 0))
        self.assertAlmostEqual(blurred.call_args.kwargs["sigmaX"], 0.75)
        self.assertAlmostEqual(blurred.call_args.kwargs["sigmaY"], 0.75)
        self.assertEqual(
            blurred.call_args.kwargs["borderType"],
            backend_module.cv2.BORDER_REPLICATE,
        )


class VendoredSnapshotTest(unittest.TestCase):
    def test_snapshot_contains_runtime_sources_config_and_license(self) -> None:
        project_root = Path(__file__).parents[1]
        repository = project_root / "third_party" / "WiLoR"
        required = [
            repository / "README.md",
            repository / "license.txt",
            repository / "wilor" / "models" / "wilor.py",
            repository / "pretrained_models" / "model_config.yaml",
            repository / "mano_data" / "mano_mean_params.npz",
        ]

        self.assertEqual(
            [path.relative_to(project_root).as_posix() for path in required if not path.is_file()],
            [],
        )

    def test_snapshot_has_no_nested_repository_or_python_bytecode(self) -> None:
        repository = Path(__file__).parents[1] / "third_party" / "WiLoR"
        forbidden = []
        for name in (".git", ".gitmodules", "__pycache__"):
            forbidden.extend(repository.rglob(name))
        forbidden.extend(repository.rglob("*.pyc"))

        self.assertEqual(
            [path.relative_to(repository).as_posix() for path in forbidden],
            [],
        )

    def test_project_gitignore_covers_vendor_bytecode(self) -> None:
        project_root = Path(__file__).parents[1]
        probe = "third_party/WiLoR/wilor/models/__pycache__/probe.pyc"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", probe],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
