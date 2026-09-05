"""Checkpoint loading, dataset inference, and test artifact dumping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from . import data
from .model.transformer import AirPianoTransformer
from .train import _autocast, _forward, _tensor_to_device
from .utils import (
    resolve_output_path,
    safe_record_name,
    write_json,
    write_npz,
)


def _batch_value(batch: dict[str, Any], name: str, index: int) -> Any:
    value = batch[name]
    if isinstance(value, torch.Tensor):
        item = value[index].detach().cpu()
        return item.item() if item.ndim == 0 else item.numpy()
    return value[index]


@torch.no_grad()
def dump_inference(
    model: AirPianoTransformer,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    bf16: bool,
) -> float:
    """Write one self-contained NPZ per slice and return aggregate BCE."""

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_loss = 0.0
    total_valid = 0
    for batch in loader:
        target_device = _tensor_to_device(batch["target"], device)
        frame_mask_device = _tensor_to_device(batch["frame_mask"], device)
        chord_mask_device = _tensor_to_device(batch["chord_mask"], device)
        with _autocast(device, bf16):
            logits = _forward(model, batch, device)
        probabilities = torch.sigmoid(logits.float())

        for index in range(logits.shape[0]):
            valid = (
                frame_mask_device[index, :, None]
                & chord_mask_device[index, None, :]
            )
            count = int(valid.sum())
            if count:
                element_loss = F.binary_cross_entropy_with_logits(
                    logits[index].float(),
                    target_device[index].float(),
                    reduction="none",
                )
                item_loss = float(element_loss[valid].mean())
                total_loss += float(element_loss[valid].sum())
                total_valid += count
            else:
                item_loss = float("nan")

            record_id = str(_batch_value(batch, "record_id", index))
            frame_start = int(_batch_value(batch, "frame_start", index))
            write_npz(
                output_dir / safe_record_name(record_id, frame_start),
                schema_version=np.asarray("airpiano-inference-v1"),
                probability=probabilities[index].cpu().numpy().astype(np.float32),
                target=batch["target"][index].cpu().numpy().astype(np.float32),
                frame_mask=batch["frame_mask"][index].cpu().numpy().astype(bool),
                chord_mask=batch["chord_mask"][index].cpu().numpy().astype(bool),
                chord_pitches=batch["chords"][index].cpu().numpy().astype(bool),
                chord_ids=np.asarray(
                    _batch_value(batch, "chord_ids", index), dtype=np.int64
                ),
                record_id=np.asarray(record_id),
                frame_start=np.asarray(frame_start, dtype=np.int64),
                fps=np.asarray(
                    float(_batch_value(batch, "fps", index)), dtype=np.float64
                ),
                loss=np.asarray(item_loss, dtype=np.float64),
            )
    return total_loss / total_valid


def load_model_checkpoint(
    path: str | Path, device: torch.device
) -> AirPianoTransformer:
    """Restore the fixed transformer from an AirPiano state-dict checkpoint."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = AirPianoTransformer().to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model


def infer_dataset(
    cfg: DictConfig,
    dataset: data.AlignmentDataset,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> float:
    """Run checkpoint inference over one dataset for test or realtime."""

    device = torch.device(str(cfg.device))
    loader = data.make_loader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        seed=int(cfg.seed),
        device=device,
    )
    model = load_model_checkpoint(checkpoint, device)
    return dump_inference(
        model,
        loader,
        device,
        Path(output_dir).expanduser().resolve(),
        bf16=bool(cfg.train.bf16),
    )


def test(cfg: DictConfig) -> None:
    """Load the configured checkpoint and dump every test-set slice."""

    checkpoint = resolve_output_path(cfg, cfg.test.checkpoint)
    output_dir = resolve_output_path(cfg, cfg.test.output_dir)
    dataset = data.build_dataset(cfg, "test")
    loss = infer_dataset(cfg, dataset, checkpoint, output_dir)
    write_json(output_dir / "test_loss.json", {"test_loss": loss})
    print(f"test_loss={loss:.8f}", flush=True)
