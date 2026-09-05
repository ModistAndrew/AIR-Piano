"""Training, evaluation, checkpointing, and dataset inference."""

from __future__ import annotations

import contextlib
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data import AlignmentDataset, collate_alignment_batch
from .model.loss import masked_bce_loss
from .model.transformer import AirPianoTransformer
from .utils import resolve_output_path, safe_record_name, write_json, write_npz


TENSOR_INPUTS = ("hands", "hand_presence", "chords", "frame_mask", "chord_mask")


def seed_everything(seed: int) -> None:
    """Seed every random source used by the baseline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_components(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate the replaceable data components declared in YAML."""

    return {
        "file_loader": instantiate(cfg.dataset.file_loader),
        "midi_loader": instantiate(cfg.dataset.midi_loader),
        "hand_loader": instantiate(cfg.dataset.hand_loader),
        "slicer": instantiate(cfg.dataset.slicer),
        "target_generator": instantiate(cfg.dataset.target_generator),
    }


def build_dataset(
    cfg: DictConfig,
    split: str,
    *,
    components: dict[str, Any] | None = None,
) -> AlignmentDataset:
    """Eagerly load one split and precompute all slice/chord indices."""

    components = components or build_data_components(cfg)
    songs = components["file_loader"].load(
        split_file=Path(str(cfg.data.splits[split])).expanduser().resolve(),
        midi_root=Path(str(cfg.data.midi_root)).expanduser().resolve(),
        hand_root=Path(str(cfg.data.hand_root)).expanduser().resolve(),
        midi_loader=components["midi_loader"],
        hand_loader=components["hand_loader"],
    )
    return AlignmentDataset(
        songs=songs,
        slicer=components["slicer"],
        target_generator=components["target_generator"],
    )


def make_loader(
    dataset: AlignmentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Build the deterministic, single-process baseline DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_alignment_batch,
        generator=generator,
    )


def _tensor_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=device.type == "cuda")


def _forward(
    model: AirPianoTransformer,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    values = {name: _tensor_to_device(batch[name], device) for name in TENSOR_INPUTS}
    return model(**values)


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(
    model: AirPianoTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    bf16: bool,
) -> float:
    """Return BCE weighted by the number of valid frame/chord cells."""

    model.eval()
    loss_sum = 0.0
    valid_sum = 0
    for batch in loader:
        target = _tensor_to_device(batch["target"], device)
        frame_mask = _tensor_to_device(batch["frame_mask"], device)
        chord_mask = _tensor_to_device(batch["chord_mask"], device)
        valid_count = int(
            (frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)).sum()
        )
        with _autocast(device, bf16):
            logits = _forward(model, batch, device)
            loss = masked_bce_loss(logits, target, frame_mask, chord_mask)
        loss_sum += float(loss) * valid_count
        valid_sum += valid_count
    return loss_sum / valid_sum


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


def load_model_checkpoint(path: str | Path, device: torch.device) -> AirPianoTransformer:
    """Restore the fixed transformer from an AirPiano state-dict checkpoint."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = AirPianoTransformer().to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: DictConfig) -> None:
    """Train, validate each epoch, restore the last model, then infer test."""

    import wandb

    seed = int(cfg.seed)
    seed_everything(seed)
    device = torch.device(str(cfg.device))
    components = build_data_components(cfg)
    train_dataset = build_dataset(cfg, "train", components=components)
    valid_dataset = build_dataset(cfg, "valid", components=components)
    test_dataset = build_dataset(cfg, "test", components=components)
    batch_size = int(cfg.train.batch_size)
    train_loader = make_loader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed, device=device
    )
    valid_loader = make_loader(
        valid_dataset, batch_size=batch_size, shuffle=False, seed=seed, device=device
    )
    test_loader = make_loader(
        test_dataset, batch_size=batch_size, shuffle=False, seed=seed, device=device
    )

    model = AirPianoTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    accumulation = int(cfg.train.gradient_accumulation_steps)
    epochs = int(cfg.train.epochs)
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = max(1, epochs * updates_per_epoch)
    warmup_steps = int(total_steps * float(cfg.train.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    checkpoint_dir = resolve_output_path(cfg, "checkpoints")
    test_output_dir = resolve_output_path(cfg, "inference/test")
    wandb_dir = resolve_output_path(cfg, cfg.wandb.dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=str(cfg.wandb.project),
        name=str(cfg.wandb.name),
        entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
        dir=str(wandb_dir),
        mode="online",
        config=resolved_config,
    )
    best_loss = float("inf")
    global_step = 0
    bf16 = bool(cfg.train.bf16)
    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_batches = 0
            for batch_index, batch in enumerate(train_loader):
                target = _tensor_to_device(batch["target"], device)
                frame_mask = _tensor_to_device(batch["frame_mask"], device)
                chord_mask = _tensor_to_device(batch["chord_mask"], device)

                # The final accumulation group may contain fewer than `accumulation`
                # batches, so divide by its actual size.
                group_size = min(
                    accumulation,
                    len(train_loader)
                    - (batch_index // accumulation) * accumulation,
                )
                with _autocast(device, bf16):
                    logits = _forward(model, batch, device)
                    loss = masked_bce_loss(logits, target, frame_mask, chord_mask)
                (loss / group_size).backward()
                accumulated_loss += float(loss.detach())
                accumulated_batches += 1

                last_batch = batch_index + 1 == len(train_loader)
                if accumulated_batches == accumulation or last_batch:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.train.max_grad_norm)
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    run.log(
                        {"train/loss": accumulated_loss / accumulated_batches}
                    )
                    accumulated_loss = 0.0
                    accumulated_batches = 0

            valid_loss = evaluate(model, valid_loader, device, bf16=bf16)
            run.log({"valid/loss": valid_loss})
            print(
                f"epoch {epoch + 1}/{epochs}: valid_loss={valid_loss:.8f}",
                flush=True,
            )
            checkpoint = {
                "format": "airpiano-checkpoint-v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_valid_loss": min(best_loss, valid_loss),
                "config": resolved_config,
            }
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(checkpoint, checkpoint_dir / "best.pt")

        model = load_model_checkpoint(checkpoint_dir / "last.pt", device)
        test_loss = dump_inference(
            model, test_loader, device, test_output_dir, bf16=bf16
        )
        write_json(test_output_dir / "test_loss.json", {"test_loss": test_loss})
        run.log({"test/loss": test_loss})
        print(f"test_loss={test_loss:.8f}", flush=True)
    finally:
        run.finish()


def infer_dataset(
    cfg: DictConfig,
    dataset: AlignmentDataset,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> float:
    """Run checkpoint inference over one dataset, as used by realtime."""

    device = torch.device(str(cfg.device))
    loader = make_loader(
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
