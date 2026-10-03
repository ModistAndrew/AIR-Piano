"""Convert SKY-Piano imputed hand markers to AirPiano canonical poses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.preprocessing.skypiano import (
    AXIS_SIGNS,
    read_skypiano_markers,
)

DEFAULT_MOTION_ROOT = PROJECT_ROOT / "assets" / "sky_piano" / "motion_hand"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "assets"
    / "sky_piano"
    / "airpiano_wilor_wrist_xyz_30fps"
)
INPUT_SUFFIX = "_imputed.csv"
CANONICAL_SCHEMA_VERSION = "airpiano-wilor-canonical-v1"
TARGET_FPS = 30.0
def read_skypiano_pose(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one marker CSV as wrist-relative OpenPose-order hand joints."""

    markers = read_skypiano_markers(path)

    presence = np.isfinite(markers).all(axis=(2, 3))
    with np.errstate(invalid="ignore"):
        pose = (markers - markers[:, :, :1]) / np.float32(1000.0)
        pose *= AXIS_SIGNS
    pose[~presence] = 0.0
    return pose.astype(np.float32, copy=False), presence


def convert_motion_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert one SKY-Piano imputed CSV or reuse its canonical output."""

    output = Path(output_path)
    if output.is_file() and not overwrite:
        return output

    pose, presence = read_skypiano_pose(input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pose=pose,
        presence=presence,
        fps=np.asarray(TARGET_FPS, dtype=np.float64),
        schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
    )
    return output


def convert_motion_root(
    motion_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert every direct ``*_imputed.csv`` file in deterministic order."""

    source_root = Path(motion_root)
    destination_root = Path(output_root)
    inputs = sorted(source_root.glob(f"*{INPUT_SUFFIX}"))
    if not inputs:
        raise FileNotFoundError(
            f"no SKY-Piano imputed motion CSVs found under {source_root}"
        )

    converted = 0
    reused = 0
    for input_path in inputs:
        record_id = input_path.name[: -len(INPUT_SUFFIX)]
        output_path = destination_root / f"{record_id}.npz"
        if output_path.is_file() and not overwrite:
            reused += 1
        else:
            converted += 1
        convert_motion_csv(input_path, output_path, overwrite=overwrite)
        print(f"SKY-Piano motion: {record_id} -> {output_path}", flush=True)
    return {"records": len(inputs), "converted": converted, "reused": reused}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=DEFAULT_MOTION_ROOT,
        help="directory containing direct *_imputed.csv files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination directory for canonical NPZ files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing canonical NPZ files",
    )
    args = parser.parse_args()

    summary = convert_motion_root(
        args.motion_root,
        args.output_root,
        overwrite=args.overwrite,
    )
    print(
        "SKY-Piano conversion complete: "
        f"records={summary['records']} converted={summary['converted']} "
        f"reused={summary['reused']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
