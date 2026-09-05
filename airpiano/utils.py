"""Small filesystem helpers shared by AirPiano stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig


def resolve_output_path(cfg: DictConfig, value: str | Path) -> Path:
    """Resolve a stage path relative to the configured project output directory."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(str(cfg.output.dir)).expanduser() / path
    return path.resolve()


def write_json(path: str | Path, value: Any) -> None:
    """Write one JSON document, creating its parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_npz(path: str | Path, **arrays: Any) -> None:
    """Write one compressed NumPy archive, creating its parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def safe_record_name(record_id: str, frame_start: int) -> str:
    """Return the deterministic filename used for an inference slice."""

    record = record_id.replace("\\", "__").replace("/", "__")
    return f"{record}__f{frame_start:09d}.npz"
