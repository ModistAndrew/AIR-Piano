"""Final pitch and duration encoding for raw score chords."""

from __future__ import annotations

import torch

from .types import Chord


class ChordTokenizer:
    """Encode pitch multi-hots followed by saturated one-based duration bins."""

    def __init__(self, chord_duration_bins: int = 96):
        self.chord_duration_bins = int(chord_duration_bins)

    def tokenize(self, chords: tuple[Chord, ...]) -> torch.Tensor:
        """Return float32 ``[N, 128 + B]`` tokens, including ``N == 0``."""

        tokens = torch.zeros(
            (len(chords), 128 + self.chord_duration_bins), dtype=torch.float32
        )
        for index, chord in enumerate(chords):
            for note in chord.onset_notes:
                # Duplicate physical notes collapse into one active pitch.
                tokens[index, note.pitch] = 1.0
            ticks = min(max(chord.duration_ticks, 1), self.chord_duration_bins)
            tokens[index, 128 + ticks - 1] = 1.0
        return tokens
