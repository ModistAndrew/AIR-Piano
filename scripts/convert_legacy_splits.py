"""Convert PianoMotion ``<subject> <video>`` lists to relative stems.

This is a one-time migration utility. AirPiano itself reads only the resulting
path-like record IDs and has no compatibility parser for the legacy format.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "PianoMotion10M"
DEFAULT_SOURCE_ROOT = Path(
    "/storage/jinxuan/data/AirPianoFull/Metadata/PianoMotion10M"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "assets" / "splits" / "pianomotion10m"
SPLIT_NAMES = ("train", "valid", "test")
OUTPUT_NAMES = {"train": "train", "valid": "valid", "test": "test_full"}


def convert_split(source: Path, destination: Path) -> int:
    """Convert one list while preserving first-seen record order."""

    records: list[str] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            subject, video_id = line.split()
            record_id = (Path(DATASET_NAME) / subject / video_id).as_posix()
            if record_id not in seen:
                seen.add(record_id)
                records.append(record_id)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{record_id}\n" for record_id in records),
        encoding="utf-8",
    )
    return len(records)


def convert_splits(source_root: Path, output_root: Path) -> dict[str, int]:
    """Convert the train, validation, and test lists beneath two roots."""

    return {
        name: convert_split(
            source_root / f"{name}.txt",
            output_root / f"{OUTPUT_NAMES[name]}.txt",
        )
        for name in SPLIT_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="directory containing legacy train.txt, valid.txt, and test.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination directory for generalized split files",
    )
    args = parser.parse_args()

    counts = convert_splits(args.source_root, args.output_root)
    for name in SPLIT_NAMES:
        print(f"{name}: {counts[name]} records")


if __name__ == "__main__":
    main()
