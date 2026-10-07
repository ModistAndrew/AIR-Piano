"""Reusable layers for the fixed AirPiano alignment transformer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


MODEL_DIM = 512
NUM_HEADS = 8
FEEDFORWARD_DIM = 2048
DROPOUT = 0.1
NUM_CHORD_LAYERS = 4
NUM_FRAME_LAYERS = 4
PAIR_DIM = 32
ONSET_PRIOR = 0.0055
ROTARY_BASE = 10_000.0


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


def apply_rotary_encoding(
    values: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``[B,H,L,D]`` values at integer positions ``[B,L]``."""

    width = values.shape[-1]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=values.device, dtype=torch.float32)
        * (-math.log(ROTARY_BASE) / width)
    )
    angles = positions.to(device=values.device, dtype=torch.float32)[
        :, None, :, None
    ] * frequencies
    cosine = torch.cos(angles).to(dtype=values.dtype)
    sine = torch.sin(angles).to(dtype=values.dtype)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.empty_like(values)
    rotated[..., 0::2] = even * cosine - odd * sine
    rotated[..., 1::2] = even * sine + odd * cosine
    return rotated


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to queries and keys."""

    def __init__(self) -> None:
        super().__init__()
        if MODEL_DIM % NUM_HEADS:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.head_dim = MODEL_DIM // NUM_HEADS
        if self.head_dim % 2:
            raise ValueError("rotary attention head width must be even")
        self.in_projection = nn.Linear(MODEL_DIM, 3 * MODEL_DIM)
        self.out_projection = nn.Linear(MODEL_DIM, MODEL_DIM)

    def forward(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = states.shape
        projected = self.in_projection(states).reshape(
            batch_size,
            length,
            3,
            NUM_HEADS,
            self.head_dim,
        )
        query, key, value = projected.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = apply_rotary_encoding(query, positions)
        key = apply_rotary_encoding(key, positions)

        attention_bias = torch.zeros(
            (batch_size, 1, 1, length),
            device=states.device,
            dtype=states.dtype,
        )
        attention_bias.masked_fill_(
            padding_mask[:, None, None, :],
            -torch.inf,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=DROPOUT if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, length, MODEL_DIM
        )
        return self.out_projection(attended)


class RotaryChordEncoderLayer(nn.Module):
    """Apply pre-norm rotary self-attention followed by an FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(MODEL_DIM)
        self.self_attention = RotarySelfAttention()
        self.self_dropout = nn.Dropout(DROPOUT)
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(MODEL_DIM, FEEDFORWARD_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEEDFORWARD_DIM, MODEL_DIM),
        )
        self.ffn_dropout = nn.Dropout(DROPOUT)

    def forward(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        update = self.self_attention(
            self.self_norm(states),
            positions,
            padding_mask,
        )
        states = states + self.self_dropout(update)
        return states + self.ffn_dropout(self.ffn(self.ffn_norm(states)))


class RotaryChordEncoder(nn.Module):
    """Four-layer chord encoder with absolute-tick rotary positions."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RotaryChordEncoderLayer() for _ in range(NUM_CHORD_LAYERS)]
        )
        self.norm = nn.LayerNorm(MODEL_DIM)

    def forward(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            states = layer(states, positions, padding_mask)
        return self.norm(states)


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
