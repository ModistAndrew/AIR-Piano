"""Fixed hand-to-chord alignment transformer."""

from __future__ import annotations

import torch
from torch import nn

from .layers import (
    DROPOUT,
    FEEDFORWARD_DIM,
    MODEL_DIM,
    NUM_CHORD_LAYERS,
    NUM_FRAME_LAYERS,
    NUM_HEADS,
    FrameAlignmentBlock,
    PairScorer,
    sinusoidal_encoding,
)


class AirPianoTransformer(nn.Module):
    """Align fixed-format two-hand frames with pitch-multihot chords.

    The architecture is intentionally fixed. Sequence lengths remain dynamic:
    ``T`` is the number of hand frames and ``C`` is the number of candidate
    chords in the current slice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hand_embedding = nn.Sequential(
            nn.Linear(128, MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM),
            nn.Dropout(DROPOUT),
        )
        self.chord_embedding = nn.Sequential(
            nn.Linear(128, MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM),
            nn.Dropout(DROPOUT),
        )

        chord_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chord_encoder = nn.TransformerEncoder(
            chord_layer,
            num_layers=NUM_CHORD_LAYERS,
            norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.global_chord = nn.Parameter(torch.empty(1, 1, MODEL_DIM))
        self.frame_layers = nn.ModuleList(
            [FrameAlignmentBlock() for _ in range(NUM_FRAME_LAYERS)]
        )
        self.frame_norm = nn.LayerNorm(MODEL_DIM)
        self.scorer = PairScorer()
        nn.init.normal_(self.global_chord, mean=0.0, std=0.02)

    @staticmethod
    def _safe_frame_padding_mask(frame_mask: torch.Tensor) -> torch.Tensor:
        """Expose one dummy key so all-padded slices remain finite in attention."""

        padding_mask = ~frame_mask
        all_padding = padding_mask.all(dim=1)
        padding_mask = padding_mask.clone()
        padding_mask[:, 0] &= ~all_padding
        return padding_mask

    def forward(
        self,
        hands: torch.Tensor,
        hand_presence: torch.Tensor,
        chords: torch.Tensor,
        frame_mask: torch.Tensor,
        chord_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return alignment logits with shape ``[B,T,C]``.

        ``hand_presence`` zeros an absent hand's coordinates and contributes two
        explicit embedding features. It is not a frame mask: ``frame_mask``
        alone marks padded time steps for attention and supervision.
        """

        batch_size, frame_count = hands.shape[:2]
        chord_count = chords.shape[1]

        hand_dtype = self.hand_embedding[0].weight.dtype
        presence = hand_presence.to(dtype=hand_dtype)
        masked_hands = hands.to(dtype=hand_dtype) * presence[..., None, None]
        hand_features = torch.cat(
            (masked_hands.flatten(start_dim=2), presence),
            dim=-1,
        )
        frames = self.hand_embedding(hand_features)
        frames = frames + sinusoidal_encoding(
            frame_count,
            MODEL_DIM,
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = frames.masked_fill(~frame_mask.unsqueeze(-1), 0.0)

        chord_dtype = self.chord_embedding[0].weight.dtype
        chord_states = self.chord_embedding(chords.to(dtype=chord_dtype))
        chord_states = chord_states.masked_fill(~chord_mask.unsqueeze(-1), 0.0)
        global_chord = self.global_chord.expand(batch_size, -1, -1)
        chord_memory = torch.cat((global_chord, chord_states), dim=1)
        chord_memory = chord_memory + sinusoidal_encoding(
            chord_count + 1,
            MODEL_DIM,
            device=chord_memory.device,
            dtype=chord_memory.dtype,
        )
        chord_padding_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=chord_mask.device,
                ),
                ~chord_mask,
            ),
            dim=1,
        )
        chord_memory = chord_memory.masked_fill(
            chord_padding_mask.unsqueeze(-1),
            0.0,
        )
        chord_memory = self.chord_encoder(
            chord_memory,
            src_key_padding_mask=chord_padding_mask,
        )
        chord_memory = chord_memory.masked_fill(
            chord_padding_mask.unsqueeze(-1),
            0.0,
        )

        frame_padding_mask = self._safe_frame_padding_mask(frame_mask)
        for layer in self.frame_layers:
            frames = layer(
                frames,
                chord_memory,
                frame_padding_mask=frame_padding_mask,
                chord_padding_mask=chord_padding_mask,
            )
            frames = frames.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        frames = self.frame_norm(frames)

        logits = self.scorer(frames, chord_memory[:, 1:])
        valid = frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)
        return logits.masked_fill(~valid, 0.0)


__all__ = ["AirPianoTransformer"]
