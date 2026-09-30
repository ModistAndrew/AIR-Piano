"""Frame-causal full-song inference with a moving chord window."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from . import data
from .inference import load_model_checkpoint, write_inference_artifact
from .model.loss import alignment_probabilities, alignment_probability_loss
from .model.transformer import AirPianoTransformer
from .song_input import iter_split_songs
from .train import _autocast, _forward
from .utils import resolve_path, write_json


@torch.no_grad()
def infer_full_song(
    model: AirPianoTransformer,
    song: data.Song,
    slicer: data.FutureChordSlicer,
    target_generator: data.TargetGenerator,
    device: torch.device,
    output_path: str | Path,
    *,
    bf16: bool,
    alignment_threshold: float,
    advance_after_chords: int,
    alignment_loss: str = "bce",
) -> tuple[float, int, int]:
    """Infer one song online and return loss, loss weight, and shift count."""

    if slicer.mode != "chord":
        raise ValueError("full-song inference requires slicer.mode=chord")
    if not model.causal_frame_attention:
        raise ValueError(
            "full-song inference requires a causal_frame_attention checkpoint"
        )
    model.eval()

    frame_count = len(song.hands)
    chord_count = len(song.chords)
    chord_positions = tuple(range(chord_count))
    score = slicer.slice(song, 0, 0, chord_positions)
    chord_tokens = score["chords"]
    chord_ids = score["chord_ids"]
    probability = torch.zeros((frame_count, chord_count), dtype=torch.float32)

    window_width = slicer.segment_frames
    final_chord_start = max(0, chord_count - window_width)
    chord_start = 0
    history_start = 0
    shift_count = 0

    for frame_index in range(frame_count):
        chord_stop = min(chord_start + window_width, chord_count)
        history_stop = frame_index + 1
        history_length = history_stop - history_start
        active_chord_count = chord_stop - chord_start
        batch = {
            "hands": song.hands[history_start:history_stop].unsqueeze(0),
            "hand_presence": song.presence[
                history_start:history_stop
            ].unsqueeze(0),
            "chords": chord_tokens[chord_start:chord_stop].unsqueeze(0),
            "frame_mask": torch.ones(
                (1, history_length), dtype=torch.bool
            ),
            "chord_mask": torch.ones(
                (1, active_chord_count), dtype=torch.bool
            ),
        }
        with _autocast(device, bf16):
            logits = _forward(model, batch, device)
        history_probability = alignment_probabilities(
            logits,
            batch["chord_mask"].to(device),
            alignment_loss,
        )[0].cpu()
        probability[frame_index, chord_start:chord_stop] = history_probability[-1]

        if chord_start >= final_chord_start:
            continue
        trailing_peaks = history_probability[
            :, -advance_after_chords:
        ].amax(dim=0)
        if bool((trailing_peaks >= alignment_threshold).any()):
            # Local slot one becomes the beginning chord after advancing once.
            predicted_start = int(torch.argmax(history_probability[:, 1]))
            history_start += predicted_start
            chord_start += 1
            shift_count += 1

    target = target_generator.generate(
        song.chords,
        frame_start=0,
        frame_count=frame_count,
        fps=slicer.fps,
    )
    loss_tensor, loss_weight = alignment_probability_loss(
        probability, target, alignment_loss
    )
    loss = float(loss_tensor)
    write_inference_artifact(
        output_path,
        probability=probability.numpy(),
        target=target.numpy(),
        frame_mask=torch.ones(frame_count, dtype=torch.bool).numpy(),
        chord_mask=torch.ones(chord_count, dtype=torch.bool).numpy(),
        chord_pitches=chord_tokens[:, :128].numpy(),
        chord_ids=chord_ids.numpy(),
        record_id=song.record_id,
        frame_start=0,
        fps=song.fps,
        loss=loss,
    )
    return loss, loss_weight, shift_count


def full_song(cfg: DictConfig) -> None:
    """Preprocess split records and infer one artifact per full song."""

    if cfg.full_song.split_file is None:
        raise ValueError("full_song.split_file must be set")
    output_dir = resolve_path(cfg.full_song.output_dir)
    checkpoint = resolve_path(cfg.full_song.checkpoint)
    device = torch.device(str(cfg.device))
    components = data.build_data_components(cfg)
    slicer = components["slicer"]
    target_generator = components["target_generator"]
    model = load_model_checkpoint(checkpoint, device)
    advance_after_chords = int(cfg.full_song.advance_after_chords)
    alignment_loss = str(cfg.get("alignment_loss", "bce"))
    total_loss = 0.0
    total_weight = 0
    for index, record_count, song, _ in iter_split_songs(
        cfg,
        cfg.full_song.split_file,
        overwrite=bool(cfg.full_song.overwrite),
        label="full-song",
        components=components,
    ):
        print(
            f"full-song {index}/{record_count}: infer {song.record_id} "
            f"({len(song.hands)} frames, {len(song.chords)} chords)",
            flush=True,
        )
        loss, loss_weight, shift_count = infer_full_song(
            model,
            song,
            slicer,
            target_generator,
            device,
            output_dir / f"{song.record_id}.npz",
            bf16=bool(cfg.train.bf16),
            alignment_threshold=float(cfg.full_song.alignment_threshold),
            advance_after_chords=advance_after_chords,
            alignment_loss=alignment_loss,
        )
        if loss_weight:
            total_loss += loss * loss_weight
            total_weight += loss_weight
        print(
            f"full-song {index}/{record_count}: {song.record_id} "
            f"loss={loss:.8f} shifts={shift_count}",
            flush=True,
        )

    if total_weight:
        aggregate_loss = total_loss / total_weight
        write_json(
            output_dir / "full_song_loss.json",
            {"full_song_loss": aggregate_loss},
        )
        print(f"full_song_loss={aggregate_loss:.8f}", flush=True)
