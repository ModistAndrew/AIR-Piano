"""Headless WiLoR model loading and batched hand-pose inference."""

from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np


@contextmanager
def repository_on_path(repository: Path) -> Iterator[None]:
    """Temporarily expose the vendored upstream WiLoR snapshot."""

    repository_text = str(repository)
    sys.path.insert(0, repository_text)
    try:
        yield
    finally:
        sys.path.remove(repository_text)


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    """Return a stable model-asset fingerprint for cache provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Models:
    """The lazily loaded objects shared across videos and frame batches."""

    model: Any
    model_cfg: Any
    detector: Any
    device: Any
    torch: Any
    fast: bool


def empty_detection_arrays() -> dict[str, np.ndarray]:
    """Create correctly shaped arrays for a frame batch with no hands."""

    return {
        "boxes_xyxy": np.empty((0, 4), np.float32),
        "detector_confidence": np.empty((0,), np.float32),
        "is_right": np.empty((0,), np.uint8),
        "pred_global_orient": np.empty((0, 1, 3, 3), np.float32),
        "pred_hand_pose": np.empty((0, 15, 3, 3), np.float32),
        "pred_betas": np.empty((0, 10), np.float32),
        "pred_cam": np.empty((0, 3), np.float32),
        "pred_cam_t": np.empty((0, 3), np.float32),
        "pred_cam_t_full": np.empty((0, 3), np.float32),
        "focal_length": np.empty((0, 2), np.float32),
        "scaled_focal_length": np.empty((0,), np.float32),
        "pred_keypoints_3d": np.empty((0, 21, 3), np.float32),
        "pred_keypoints_2d": np.empty((0, 21, 2), np.float32),
        "pred_keypoints_2d_full": np.empty((0, 21, 2), np.float32),
        "box_center": np.empty((0, 2), np.float32),
        "box_size": np.empty((0,), np.float32),
        "image_size_wh": np.empty((0, 2), np.float32),
    }


class WiLoRCropDataset:
    """Produce WiLoR crops with the baseline OpenCV antialiasing path.

    This AirPiano-owned adapter preserves the established crop numerics without
    modifying the vendored upstream snapshot.
    """

    def __init__(
        self,
        cfg: Any,
        image: np.ndarray,
        boxes: np.ndarray,
        right: np.ndarray,
        rescale_factor: float,
        fp16: bool,
    ) -> None:
        self.cfg = cfg
        self.image = image
        self.boxes = boxes.astype(np.float32)
        self.center = (self.boxes[:, 2:4] + self.boxes[:, 0:2]) / 2.0
        self.scale = (
            rescale_factor * (self.boxes[:, 2:4] - self.boxes[:, 0:2]) / 200.0
        )
        self.right = right.astype(np.float32)
        self.image_size = int(cfg.MODEL.IMAGE_SIZE)
        self.mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD)
        self.fp16 = fp16

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from wilor.datasets.utils import (
            convert_cvimg_to_tensor,
            expand_to_aspect_ratio,
            generate_image_patch_cv2,
        )

        center = self.center[index].copy()
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = expand_to_aspect_ratio(
            self.scale[index] * 200.0,
            target_aspect_ratio=bbox_shape,
        ).max()
        right = self.right[index].copy()

        image = self.image.copy()
        downsampling_factor = (bbox_size / self.image_size) / 2.0
        if downsampling_factor > 1.1:
            sigma = (downsampling_factor - 1.0) / 2.0
            image = cv2.GaussianBlur(
                image,
                ksize=(0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )

        patch, _ = generate_image_patch_cv2(
            image,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            self.image_size,
            self.image_size,
            right == 0,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        patch = convert_cvimg_to_tensor(patch[:, :, ::-1])
        for channel in range(min(self.image.shape[2], 3)):
            patch[channel] = (patch[channel] - self.mean[channel]) / self.std[channel]

        if self.fp16:
            import torch

            patch = torch.from_numpy(patch).half()
        return {
            "img": patch,
            "personid": index,
            "box_center": center,
            "box_size": bbox_size,
            "img_size": np.asarray(
                [image.shape[1], image.shape[0]], dtype=np.float64
            ),
            "right": right,
        }


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def infer_frame_batch(
    frames: Sequence[np.ndarray],
    models: Models,
    detector_confidence: float,
    rescale_factor: float,
    hand_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Infer a frame batch into offsets plus flattened frame-major detections."""

    detector_outputs = models.detector(
        list(frames), conf=detector_confidence, verbose=False
    )
    detections_by_frame: list[list[tuple[np.ndarray, float, float]]] = []
    for output in detector_outputs:
        detections: list[tuple[np.ndarray, float, float]] = []
        if output.boxes is not None:
            boxes = output.boxes.xyxy.detach().cpu().numpy()
            confidence = output.boxes.conf.detach().cpu().numpy()
            classes = output.boxes.cls.detach().cpu().numpy()
            detections = [
                (box.astype(np.float32), float(score), float(handedness))
                for box, score, handedness in zip(boxes, confidence, classes)
            ]
        detections_by_frame.append(detections)

    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(items) for items in detections_by_frame])
    if offsets[-1] == 0:
        return offsets, empty_detection_arrays()

    from torch.utils.data import DataLoader
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    crop_items: list[dict[str, Any]] = []
    detector_boxes: list[np.ndarray] = []
    detector_scores: list[float] = []
    for frame, detections in zip(frames, detections_by_frame):
        if not detections:
            continue
        boxes = np.stack([item[0] for item in detections])
        right = np.asarray([item[2] for item in detections], dtype=np.float32)
        dataset = WiLoRCropDataset(
            models.model_cfg,
            frame,
            boxes,
            right,
            rescale_factor=rescale_factor,
            fp16=models.fast,
        )
        crop_items.extend(dataset[index] for index in range(len(dataset)))
        detector_boxes.extend(item[0] for item in detections)
        detector_scores.extend(item[1] for item in detections)

    loader = DataLoader(
        crop_items,
        batch_size=hand_batch_size,
        shuffle=False,
        num_workers=0,
    )
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in empty_detection_arrays()
        if key not in {"boxes_xyxy", "detector_confidence"}
    }
    for batch in loader:
        batch = recursive_to(batch, models.device)
        with models.torch.inference_mode():
            output = models.model(batch)

        right = batch["right"]
        multiplier = 2 * right - 1
        full_camera = output["pred_cam"].clone()
        full_camera[:, 1] *= multiplier
        image_size = batch["img_size"].float()
        scaled_focal = (
            models.model_cfg.EXTRA.FOCAL_LENGTH
            / models.model_cfg.MODEL.IMAGE_SIZE
            * image_size.max(dim=1).values
        )
        full_translation = cam_crop_to_full(
            full_camera,
            batch["box_center"].float(),
            batch["box_size"].float(),
            image_size,
            scaled_focal,
        )

        joints_3d = output["pred_keypoints_3d"].clone()
        joints_3d[:, :, 0] *= multiplier[:, None]
        camera_points = joints_3d + full_translation[:, None, :]
        normalized = camera_points / camera_points[:, :, 2:3]
        image_center = image_size / 2.0
        joints_full = (
            normalized[:, :, :2] * scaled_focal[:, None, None]
            + image_center[:, None, :]
        )

        values = {
            "is_right": _numpy(right).astype(np.uint8),
            "pred_global_orient": _numpy(
                output["pred_mano_params"]["global_orient"]
            ),
            "pred_hand_pose": _numpy(output["pred_mano_params"]["hand_pose"]),
            "pred_betas": _numpy(output["pred_mano_params"]["betas"]),
            "pred_cam": _numpy(output["pred_cam"]),
            "pred_cam_t": _numpy(output["pred_cam_t"]),
            "pred_cam_t_full": _numpy(full_translation),
            "focal_length": _numpy(output["focal_length"]),
            "scaled_focal_length": _numpy(scaled_focal),
            "pred_keypoints_3d": _numpy(joints_3d),
            "pred_keypoints_2d": _numpy(output["pred_keypoints_2d"]),
            "pred_keypoints_2d_full": _numpy(joints_full),
            "box_center": _numpy(batch["box_center"]),
            "box_size": _numpy(batch["box_size"]),
            "image_size_wh": _numpy(image_size),
        }
        for key, value in values.items():
            collected[key].append(value)

    arrays = {
        "boxes_xyxy": np.asarray(detector_boxes, dtype=np.float32),
        "detector_confidence": np.asarray(detector_scores, dtype=np.float32),
    }
    arrays.update(
        {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    )
    return offsets, arrays


class WiLoRBackend:
    """Own one lazily loaded WiLoR model bundle and its inference settings."""

    def __init__(
        self,
        wilor_repo: str | Path,
        checkpoint: str | Path,
        model_config: str | Path,
        detector_checkpoint: str | Path,
        mano_model: str | Path,
        mano_mean_params: str | Path,
        device: str = "cuda:1",
        detector_confidence: float = 0.3,
        rescale_factor: float = 2.0,
        frame_batch_size: int = 8,
        hand_batch_size: int = 16,
        fast: bool = False,
        compile: bool = False,
    ) -> None:
        self.wilor_repo = Path(wilor_repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.model_config = Path(model_config).expanduser().resolve()
        self.detector_checkpoint = Path(detector_checkpoint).expanduser().resolve()
        self.mano_model = Path(mano_model).expanduser().resolve()
        self.mano_mean_params = Path(mano_mean_params).expanduser().resolve()
        self.device = str(device)
        self.detector_confidence = float(detector_confidence)
        self.rescale_factor = float(rescale_factor)
        self.frame_batch_size = int(frame_batch_size)
        self.hand_batch_size = int(hand_batch_size)
        self.fast = bool(fast)
        self.compile = bool(compile)
        self._models: Models | None = None

    @property
    def models(self) -> Models:
        if self._models is None:
            self._models = self._load_models()
        return self._models

    def _load_models(self) -> Models:
        """Load the vendored upstream snapshot with absolute MANO asset paths."""

        with repository_on_path(self.wilor_repo):
            import torch
            from ultralytics import YOLO
            from wilor.configs import get_config
            from wilor.models import WiLoR

            model_cfg = get_config(str(self.model_config), update_cachedir=False)
            model_cfg.defrost()
            if (
                "vit" in model_cfg.MODEL.BACKBONE.TYPE
                and "BBOX_SHAPE" not in model_cfg.MODEL
            ):
                model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
                model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.MANO.DATA_DIR = str(self.mano_model.parent)
            model_cfg.MANO.MODEL_PATH = str(self.mano_model.parent)
            model_cfg.MANO.MEAN_PARAMS = str(self.mano_mean_params)
            model_cfg.freeze()

            model = WiLoR.load_from_checkpoint(
                str(self.checkpoint),
                strict=False,
                cfg=model_cfg,
                init_renderer=False,
            )
            device = torch.device(self.device)
            if self.fast:
                torch.set_float32_matmul_precision("high")
                model = model.half()
                if self.compile:
                    model.backbone = torch.compile(model.backbone)
                model.backbone.skip_blocks = True
            model = model.to(device).eval()
            detector = YOLO(str(self.detector_checkpoint)).to(device)
        return Models(model, model_cfg, detector, device, torch, self.fast)

    def build_chunk(
        self,
        frame_indices: Sequence[int],
        timestamps: Sequence[float],
        frames: Sequence[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Infer one cache chunk while retaining zero-detection frames."""

        all_offsets = [0]
        pieces: dict[str, list[np.ndarray]] = {
            key: [] for key in empty_detection_arrays()
        }
        with repository_on_path(self.wilor_repo):
            for start in range(0, len(frames), self.frame_batch_size):
                end = min(start + self.frame_batch_size, len(frames))
                offsets, detections = infer_frame_batch(
                    frames[start:end],
                    self.models,
                    self.detector_confidence,
                    self.rescale_factor,
                    self.hand_batch_size,
                )
                all_offsets.extend((offsets[1:] + all_offsets[-1]).tolist())
                for key, value in detections.items():
                    if len(value):
                        pieces[key].append(value)

        result = {
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "frame_timestamps_seconds": np.asarray(timestamps, dtype=np.float64),
            "frame_detection_offsets": np.asarray(all_offsets, dtype=np.int64),
        }
        empty = empty_detection_arrays()
        for key, parts in pieces.items():
            result[key] = np.concatenate(parts, axis=0) if parts else empty[key]
        return result

    def model_metadata(self) -> dict[str, Any]:
        """Describe exact upstream code and immutable assets used by a cache."""

        return {
            "wilor_repository": str(self.wilor_repo),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "config": str(self.model_config),
            "detector_checkpoint": str(self.detector_checkpoint),
            "detector_checkpoint_sha256": sha256_file(
                self.detector_checkpoint
            ),
            "mano_model": str(self.mano_model),
            "mano_model_sha256": sha256_file(self.mano_model),
            "mano_mean_params": str(self.mano_mean_params),
            "mano_mean_params_sha256": sha256_file(self.mano_mean_params),
        }

    def inference_metadata(self, chunk_frames: int) -> dict[str, Any]:
        return {
            "device": self.device,
            "fast": self.fast,
            "compiled": self.compile,
            "detector_confidence": self.detector_confidence,
            "rescale_factor": self.rescale_factor,
            "frame_batch_size": self.frame_batch_size,
            "hand_batch_size": self.hand_batch_size,
            "chunk_frames": int(chunk_frames),
        }
