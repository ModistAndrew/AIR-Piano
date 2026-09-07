"""Reusable layers for the fixed AirPiano alignment transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


MODEL_DIM = 512
NUM_HEADS = 8
FEEDFORWARD_DIM = 2048
DROPOUT = 0.1
NUM_CHORD_LAYERS = 4
NUM_FRAME_LAYERS = 4
PAIR_DIM = 32
ONSET_PRIOR = 0.0055


def sinusoidal_encoding(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a dynamic sinusoidal position table with shape ``[1, L, D]``."""

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros((length, width), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequencies)
    encoding[:, 1::2] = torch.cos(position * frequencies)
    return encoding.unsqueeze(0).to(dtype=dtype)


class FrameAlignmentBlock(nn.Module):
    """Apply pre-norm frame self-attention, chord attention, and an FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(MODEL_DIM)
        self.self_attention = nn.MultiheadAttention(
            MODEL_DIM,
            NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(MODEL_DIM)
        self.cross_attention = nn.MultiheadAttention(
            MODEL_DIM,
            NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(MODEL_DIM, FEEDFORWARD_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEEDFORWARD_DIM, MODEL_DIM),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(
        self,
        frames: torch.Tensor,
        chord_memory: torch.Tensor,
        *,
        frame_padding_mask: torch.Tensor,
        frame_attention_mask: torch.Tensor | None,
        chord_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update ``[B,T,D]`` frames using valid frame and chord positions."""

        normalized = self.self_norm(frames)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=frame_padding_mask,
            attn_mask=frame_attention_mask,
            need_weights=False,
        )
        frames = frames + self.dropout(update)

        normalized = self.cross_norm(frames)
        update, _ = self.cross_attention(
            normalized,
            chord_memory,
            chord_memory,
            key_padding_mask=chord_padding_mask,
            need_weights=False,
        )
        frames = frames + self.dropout(update)
        return frames + self.dropout(self.ffn(self.ffn_norm(frames)))


class PairScorer(nn.Module):
    """Produce one independent logit for every frame/chord pair."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_query = nn.Linear(MODEL_DIM, PAIR_DIM)
        self.chord_key = nn.Linear(MODEL_DIM, PAIR_DIM)
        self.bias = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.frame_query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.chord_key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.frame_query.bias, mean=0.0, std=0.002)
        nn.init.normal_(self.chord_key.bias, mean=0.0, std=0.002)
        nn.init.constant_(self.bias, math.log(ONSET_PRIOR / (1.0 - ONSET_PRIOR)))

    def forward(
        self,
        frame_states: torch.Tensor,
        chord_states: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``[B,T,D]`` against ``[B,C,D]`` as ``[B,T,C]``."""

        queries = self.frame_query(frame_states)
        keys = self.chord_key(chord_states)
        return (
            torch.einsum("btd,bcd->btc", queries, keys) / math.sqrt(PAIR_DIM)
            + self.bias
        )
