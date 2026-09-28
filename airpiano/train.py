"""Training, validation, and checkpoint creation."""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from . import data
from .model.loss import alignment_loss_count, masked_alignment_loss
from .model.transformer import AirPianoTransformer
from .utils import resolve_path


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


def _forward_heads(
    model: AirPianoTransformer,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = {name: _tensor_to_device(batch[name], device) for name in TENSOR_INPUTS}
    return model.forward_heads(**values)


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
    alignment_loss: str = "bce",
) -> float:
    """Return the configured loss weighted by its valid target units."""

    model.eval()
    loss_sum = 0.0
    valid_sum = 0
    for batch in loader:
        target = _tensor_to_device(batch["target"], device)
        frame_mask = _tensor_to_device(batch["frame_mask"], device)
        chord_mask = _tensor_to_device(batch["chord_mask"], device)
        valid_count = alignment_loss_count(
            target, frame_mask, chord_mask, alignment_loss
        )
        with _autocast(device, bf16):
            logits = _forward(model, batch, device)
            loss = masked_alignment_loss(
                logits,
                target,
                frame_mask,
                chord_mask,
                alignment_loss,
            )
        loss_sum += float(loss) * valid_count
        valid_sum += valid_count
    return loss_sum / valid_sum


@torch.no_grad()
def evaluate_multitask(
    model: AirPianoTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    bf16: bool,
    alignment_loss: str,
    auxiliary_alignment_loss: str,
    auxiliary_loss_weight: float,
) -> tuple[float, float, float]:
    """Return primary, auxiliary, and combined validation losses."""

    model.eval()
    primary_sum = 0.0
    primary_valid = 0
    auxiliary_sum = 0.0
    auxiliary_valid = 0
    for batch in loader:
        target = _tensor_to_device(batch["target"], device)
        auxiliary_target = _tensor_to_device(
            batch["auxiliary_target"], device
        )
        frame_mask = _tensor_to_device(batch["frame_mask"], device)
        chord_mask = _tensor_to_device(batch["chord_mask"], device)
        primary_count = alignment_loss_count(
            target, frame_mask, chord_mask, alignment_loss
        )
        auxiliary_count = alignment_loss_count(
            auxiliary_target,
            frame_mask,
            chord_mask,
            auxiliary_alignment_loss,
        )
        with _autocast(device, bf16):
            primary_logits, auxiliary_logits = _forward_heads(
                model, batch, device
            )
            primary_loss = masked_alignment_loss(
                primary_logits,
                target,
                frame_mask,
                chord_mask,
                alignment_loss,
            )
            auxiliary_loss = masked_alignment_loss(
                auxiliary_logits,
                auxiliary_target,
                frame_mask,
                chord_mask,
                auxiliary_alignment_loss,
            )
        primary_sum += float(primary_loss) * primary_count
        primary_valid += primary_count
        auxiliary_sum += float(auxiliary_loss) * auxiliary_count
        auxiliary_valid += auxiliary_count

    primary_mean = primary_sum / primary_valid
    auxiliary_mean = auxiliary_sum / auxiliary_valid
    combined = primary_mean + auxiliary_loss_weight * auxiliary_mean
    return primary_mean, auxiliary_mean, combined


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
    model = instantiate(cfg.model).to(device)
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
    checkpoint_dir = resolve_path(cfg.output.dir) / "checkpoints"
    wandb_dir = resolve_path(cfg.wandb.dir)
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
    alignment_loss = str(cfg.get("alignment_loss", "bce"))
    auxiliary_alignment_loss = cfg.train.get("auxiliary_alignment_loss")
    multitask = auxiliary_alignment_loss is not None
    auxiliary_loss_weight = float(
        cfg.train.get("auxiliary_loss_weight", 0.0)
    )
    if multitask:
        auxiliary_alignment_loss = str(auxiliary_alignment_loss)
    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_primary_loss = 0.0
            accumulated_auxiliary_loss = 0.0
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
                    if multitask:
                        primary_logits, auxiliary_logits = _forward_heads(
                            model, batch, device
                        )
                        primary_loss = masked_alignment_loss(
                            primary_logits,
                            target,
                            frame_mask,
                            chord_mask,
                            alignment_loss,
                        )
                        auxiliary_loss = masked_alignment_loss(
                            auxiliary_logits,
                            _tensor_to_device(
                                batch["auxiliary_target"], device
                            ),
                            frame_mask,
                            chord_mask,
                            auxiliary_alignment_loss,
                        )
                        loss = (
                            primary_loss
                            + auxiliary_loss_weight * auxiliary_loss
                        )
                    else:
                        logits = _forward(model, batch, device)
                        primary_loss = masked_alignment_loss(
                            logits,
                            target,
                            frame_mask,
                            chord_mask,
                            alignment_loss,
                        )
                        auxiliary_loss = None
                        loss = primary_loss
                (loss / group_size).backward()
                accumulated_loss += float(loss.detach())
                accumulated_primary_loss += float(primary_loss.detach())
                if auxiliary_loss is not None:
                    accumulated_auxiliary_loss += float(
                        auxiliary_loss.detach()
                    )
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
                    train_metrics = {
                        "train/loss": accumulated_loss / accumulated_batches
                    }
                    if multitask:
                        train_metrics.update(
                            {
                                "train/primary_loss": (
                                    accumulated_primary_loss
                                    / accumulated_batches
                                ),
                                "train/auxiliary_loss": (
                                    accumulated_auxiliary_loss
                                    / accumulated_batches
                                ),
                            }
                        )
                    run.log(train_metrics)
                    accumulated_loss = 0.0
                    accumulated_primary_loss = 0.0
                    accumulated_auxiliary_loss = 0.0
                    accumulated_batches = 0

            if multitask:
                valid_loss, valid_auxiliary_loss, valid_combined_loss = (
                    evaluate_multitask(
                        model,
                        valid_loader,
                        device,
                        bf16=bf16,
                        alignment_loss=alignment_loss,
                        auxiliary_alignment_loss=auxiliary_alignment_loss,
                        auxiliary_loss_weight=auxiliary_loss_weight,
                    )
                )
                run.log(
                    {
                        "valid/loss": valid_loss,
                        "valid/auxiliary_loss": valid_auxiliary_loss,
                        "valid/combined_loss": valid_combined_loss,
                    }
                )
                print(
                    f"epoch {epoch + 1}/{epochs}: "
                    f"valid_loss={valid_loss:.8f} "
                    f"valid_auxiliary_loss={valid_auxiliary_loss:.8f} "
                    f"valid_combined_loss={valid_combined_loss:.8f}",
                    flush=True,
                )
            else:
                valid_loss = evaluate(
                    model,
                    valid_loader,
                    device,
                    bf16=bf16,
                    alignment_loss=alignment_loss,
                )
                run.log({"valid/loss": valid_loss})
                print(
                    f"epoch {epoch + 1}/{epochs}: "
                    f"valid_loss={valid_loss:.8f}",
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
