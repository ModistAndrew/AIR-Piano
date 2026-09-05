"""Selection shared by raw visualization and canonical preprocessing."""

from __future__ import annotations

import numpy as np


def select_best_hand_indices(
    frame_detection_offsets: np.ndarray,
    detector_confidence: np.ndarray,
    is_right: np.ndarray,
) -> np.ndarray:
    """Return absolute detection indices for the best Left/Right per frame.

    Columns are anatomical Left (``is_right == 0``) and Right
    (``is_right == 1``). ``np.argmax`` deliberately keeps the first detection
    when confidence values tie.
    """

    selected = np.full((len(frame_detection_offsets) - 1, 2), -1, np.int64)
    for frame in range(len(selected)):
        start = int(frame_detection_offsets[frame])
        end = int(frame_detection_offsets[frame + 1])
        for slot in range(2):
            candidates = np.flatnonzero(is_right[start:end] == slot) + start
            if candidates.size:
                selected[frame, slot] = int(
                    candidates[np.argmax(detector_confidence[candidates])]
                )
    return selected
