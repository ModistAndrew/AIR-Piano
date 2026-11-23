"""Fit the fixed left/right camera-wrist XY means used by the absolute ablations."""

from __future__ import annotations

import hashlib
import numpy as np
from omegaconf import DictConfig

from airpiano.data.files import read_record_stems
from airpiano.preprocessing.canonical import WiLoRToCanonical, raw_cache_has_markers
from airpiano.utils import resolve_path, write_json


def fit_hand_mean(cfg: DictConfig) -> None:
    split = resolve_path(cfg.data.splits.train)
    raw_root = resolve_path(cfg.data.raw_wilor_root)
    converter = WiLoRToCanonical(target_fps=float(cfg.dataset.fps))
    records = read_record_stems([split])
    sums = np.zeros((2, 2), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for index, record in enumerate(records, 1):
        raw = raw_root / record
        if not raw_cache_has_markers(raw):
            raise FileNotFoundError(f"incomplete training raw WiLoR cache: {raw}")
        pose, presence = converter.read_resampled(raw)
        for hand in range(2):
            xy = pose[presence[:, hand], hand, 0, :2]
            sums[hand] += xy.sum(axis=0, dtype=np.float64)
            counts[hand] += len(xy)
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"fit wrist mean {index}/{len(records)}", flush=True)
    if np.any(counts == 0):
        raise ValueError("training split must contain observations of both hands")
    mean = sums / counts[:, None]
    if not np.isfinite(mean).all():
        raise ValueError("training wrist mean contains nonfinite coordinates")
    write_json(resolve_path(cfg.data.hand_mean_path), {
        "mean_xy": mean.tolist(),
        "counts": counts.tolist(),
        "fps": converter.target_fps,
        "training_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    })
