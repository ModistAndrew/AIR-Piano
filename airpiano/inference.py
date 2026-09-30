"""Checkpoint loading, dataset inference, and test artifact dumping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import data
from .model.loss import (
    alignment_loss_count,
    alignment_probabilities,
    masked_alignment_loss,
)
from .model.transformer import AirPianoTransformer
from .train import _autocast, _forward, _tensor_to_device
from .utils import (
    resolve_path,
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


def write_inference_artifact(
    output_path: str | Path,
    *,
    probability: Any,
    target: Any,
    frame_mask: Any,
    chord_mask: Any,
    chord_pitches: Any,
    chord_ids: Any,
    record_id: str,
    frame_start: int,
    fps: float,
    loss: float,
) -> None:
    """Write one artifact using the shared inference-v1 array contract."""

    write_npz(
        output_path,
        schema_version=np.asarray("airpiano-inference-v1"),
        probability=np.asarray(probability, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        frame_mask=np.asarray(frame_mask, dtype=bool),
        chord_mask=np.asarray(chord_mask, dtype=bool),
        chord_pitches=np.asarray(chord_pitches, dtype=bool),
        chord_ids=np.asarray(chord_ids, dtype=np.int64),
        record_id=np.asarray(record_id),
        frame_start=np.asarray(frame_start, dtype=np.int64),
        fps=np.asarray(fps, dtype=np.float64),
        loss=np.asarray(loss, dtype=np.float64),
    )


@torch.no_grad()
def dump_inference(
    model: AirPianoTransformer,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    bf16: bool,
    alignment_loss: str = "bce",
) -> float:
    """Write one NPZ per slice and return the aggregate configured loss."""

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
        probabilities = alignment_probabilities(
            logits, chord_mask_device, alignment_loss
        )

        for index in range(logits.shape[0]):
            item_frame_mask = frame_mask_device[index : index + 1]
            item_chord_mask = chord_mask_device[index : index + 1]
            count = alignment_loss_count(
                target_device[index : index + 1],
                item_frame_mask,
                item_chord_mask,
                alignment_loss,
            )
            if count:
                item_loss_tensor = masked_alignment_loss(
                    logits[index : index + 1],
                    target_device[index : index + 1],
                    item_frame_mask,
                    item_chord_mask,
                    alignment_loss,
                )
                item_loss = float(item_loss_tensor)
                total_loss += item_loss * count
                total_valid += count
            else:
                item_loss = float("nan")

            record_id = str(_batch_value(batch, "record_id", index))
            frame_start = int(_batch_value(batch, "frame_start", index))
            write_inference_artifact(
                output_dir / safe_record_name(record_id, frame_start),
                probability=probabilities[index].cpu().numpy(),
                target=batch["target"][index].cpu().numpy(),
                frame_mask=batch["frame_mask"][index].cpu().numpy(),
                chord_mask=batch["chord_mask"][index].cpu().numpy(),
                chord_pitches=batch["chords"][index, :, :128].cpu().numpy(),
                chord_ids=_batch_value(batch, "chord_ids", index),
                record_id=record_id,
                frame_start=frame_start,
                fps=float(_batch_value(batch, "fps", index)),
                loss=item_loss,
            )
    return total_loss / total_valid


def load_model_checkpoint(
    path: str | Path, device: torch.device
) -> AirPianoTransformer:
    """Instantiate the saved model configuration and restore its state."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", {})
    # for compatibility with older checkpoints
    model_config = OmegaConf.merge(
        {
            "_target_": "airpiano.model.transformer.AirPianoTransformer",
            "causal_frame_attention": False,
            "include_chord_duration": False,
            "include_chord_offsets": False,
            "zero_hand_input": False,
        },
        checkpoint_config.get("model", {}),
    )
    model = instantiate(model_config).to(device)
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
        resolve_path(output_dir),
        bf16=bool(cfg.train.bf16),
        alignment_loss=str(cfg.get("alignment_loss", "bce")),
    )


def test(cfg: DictConfig) -> None:
    """Load the configured checkpoint and dump every test-set slice."""

    checkpoint = resolve_path(cfg.test.checkpoint)
    output_dir = resolve_path(cfg.test.output_dir)
    dataset = data.build_dataset(
        cfg,
        "test",
        split_file=str(cfg.test.split_file),
    )
    loss = infer_dataset(cfg, dataset, checkpoint, output_dir)
    write_json(output_dir / "test_loss.json", {"test_loss": loss})
    print(f"test_loss={loss:.8f}", flush=True)
