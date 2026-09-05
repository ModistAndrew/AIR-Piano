"""Training, validation, and checkpoint creation."""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import data
from .model.loss import masked_bce_loss
from .model.transformer import AirPianoTransformer
from .utils import resolve_output_path


TENSOR_INPUTS = ("hands", "hand_presence", "chords", "frame_mask", "chord_mask")


def seed_everything(seed: int) -> None:
    """Seed every random source used by the baseline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: DictConfig) -> None:
    """Train the baseline, validate each epoch, and write checkpoints."""

    import wandb

    seed = int(cfg.seed)
    seed_everything(seed)
    device = torch.device(str(cfg.device))
    components = data.build_data_components(cfg)
    train_dataset = data.build_dataset(cfg, "train", components=components)
    valid_dataset = data.build_dataset(cfg, "valid", components=components)
    batch_size = int(cfg.train.batch_size)
    train_loader = data.make_loader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed, device=device
    )
    valid_loader = data.make_loader(
        valid_dataset, batch_size=batch_size, shuffle=False, seed=seed, device=device
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
    finally:
        run.finish()
