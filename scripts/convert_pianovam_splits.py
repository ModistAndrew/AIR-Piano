"""Create AirPiano split lists from PianoVAM's canonical metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/storage/jinxuan/data/PianoVAM_v1")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "assets" / "splits" / "pianovam"
SPLIT_NAMES = ("train", "valid", "test")
ALLOWED_SOURCE_SPLITS = {
    "train",
    "ext-train",
    "valid",
    "test",
    "special(blurry)",
    "special(4hands)",
}


def _metadata_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    records = list(payload.values())
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: every metadata entry must be an object")
    return records


def convert_splits(data_root: Path, output_root: Path) -> dict[str, int]:
    """Validate PianoVAM metadata and write deterministic AirPiano splits."""

    metadata_path = data_root / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        records = _metadata_records(json.load(handle), metadata_path)

    split_records = {name: [] for name in SPLIT_NAMES}
    seen: set[str] = set()
    for record in records:
        record_id = record.get("record_time")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{metadata_path}: record_time must be a nonempty string")
        if record_id in seen:
            raise ValueError(f"{metadata_path}: duplicate record_time {record_id!r}")
        seen.add(record_id)

        source_split = record.get("split")
        if source_split not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(
                f"{metadata_path}: unexpected split {source_split!r} "
                f"for {record_id!r}"
            )
        if source_split.startswith("special("):
            continue

        target_split = "train" if source_split == "ext-train" else source_split
        for required in (
            data_root / "Video" / f"{record_id}.mp4",
            data_root / "MIDI" / f"{record_id}.mid",
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"required PianoVAM modality not found: {required}"
                )
        split_records[target_split].append(record_id)

    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, record_ids in split_records.items():
        record_ids.sort()
        (output_root / f"{split_name}.txt").write_text(
            "".join(f"{record_id}\n" for record_id in record_ids),
            encoding="utf-8",
        )
    return {name: len(split_records[name]) for name in SPLIT_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="PianoVAM root containing metadata.json, Video/, and MIDI/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination directory for AirPiano split files",
    )
    args = parser.parse_args()

    counts = convert_splits(args.data_root, args.output_root)
    for name in SPLIT_NAMES:
        print(f"{name}: {counts[name]} records")


if __name__ == "__main__":
    main()
