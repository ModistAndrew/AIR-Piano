"""Create a symlink-only training view of PianoMotion10M and PianoVAM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "valid", "test")
EXPECTED_FULL_SPLIT_COUNTS = {"train": 963, "valid": 101, "test": 13}
EXPECTED_EXTENDED_SPLIT_COUNTS = {"train": 1951, "valid": 101, "test": 13}
DEFAULT_FULL_SPLIT_ROOT = PROJECT_ROOT / "assets/splits/full"
DEFAULT_EXTENDED_SPLIT_ROOT = PROJECT_ROOT / "assets/splits/extended"
DEFAULT_PIANOMOTION_SPLIT_ROOT = PROJECT_ROOT / "assets/splits/pianomotion10m"
DEFAULT_PIANOVAM_SPLIT_ROOT = PROJECT_ROOT / "assets/splits/pianovam"
DEFAULT_VIEW_ROOT = Path("/storage2/jinxuan/data/AirPianoFull")
DEFAULT_PIANOMOTION_MIDI_ROOT = Path("/storage/jinxuan/data/PianoMotion10M/midi")
DEFAULT_PIANOMOTION_HAND_ROOT = Path(
    "/storage2/jinxuan/data/PianoMotion10M/airpiano_wilor_2p5d_30fps"
)
DEFAULT_PIANOVAM_MIDI_ROOT = Path("/storage/jinxuan/data/PianoVAM_v1/MIDI")
DEFAULT_PIANOVAM_HAND_ROOT = Path(
    "/storage2/jinxuan/data/PianoVAM_v1/airpiano_wilor_2p5d_30fps"
)


def read_records(path: Path) -> list[str]:
    """Read path-like record stems without hiding duplicate entries."""

    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                records.append(Path(line).as_posix())
    return records


def read_splits(split_root: Path) -> dict[str, list[str]]:
    """Read each standard split under a root."""

    result: dict[str, list[str]] = {}
    for split_name in SPLIT_NAMES:
        path = split_root / f"{split_name}.txt"
        result[split_name] = read_records(path)
    return result


def expected_split_counts(split_root: Path) -> Mapping[str, int] | None:
    """Return count validation for a known combined split root."""

    resolved = split_root.resolve()
    if resolved == DEFAULT_FULL_SPLIT_ROOT.resolve():
        return EXPECTED_FULL_SPLIT_COUNTS
    if resolved == DEFAULT_EXTENDED_SPLIT_ROOT.resolve():
        return EXPECTED_EXTENDED_SPLIT_COUNTS
    return None


def unique_records(
    splits: Mapping[str, list[str]], split_root: Path
) -> list[str]:
    """Return split-order records while rejecting within/cross-split overlap."""

    records: list[str] = []
    seen: set[str] = set()
    for split_name in SPLIT_NAMES:
        for record_id in splits[split_name]:
            if record_id in seen:
                raise ValueError(
                    f"{split_root}: duplicate record {record_id!r} "
                    "within or across splits"
                )
            seen.add(record_id)
            records.append(record_id)
    return records


def resolved_link_target(path: Path) -> Path:
    """Resolve an existing or dangling link target to an absolute path."""

    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    return target.resolve(strict=False)


def validate_destination(destination: Path, source: Path) -> None:
    """Accept an identical link and reject every conflicting destination."""

    if destination.is_symlink():
        if resolved_link_target(destination) == source:
            return
        raise FileExistsError(
            f"refusing to replace symlink {destination} -> "
            f"{os.readlink(destination)}"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path {destination}")


def create_data_view(
    *,
    full_split_root: Path,
    pianomotion_split_root: Path,
    pianovam_split_root: Path,
    view_root: Path,
    pianomotion_midi_root: Path,
    pianomotion_hand_root: Path,
    pianovam_midi_root: Path,
    pianovam_hand_root: Path,
    pianomotion_extra_splits: tuple[Path, ...] = (),
    view_hand_root: Path | None = None,
    expected_split_counts: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Create safe, idempotent MIDI and canonical-hand symlink trees."""

    full_splits = read_splits(full_split_root)
    if expected_split_counts is not None:
        actual_counts = {
            name: len(full_splits[name]) for name in SPLIT_NAMES
        }
        if actual_counts != dict(expected_split_counts):
            raise ValueError(
                f"unexpected full split counts: {actual_counts}; "
                f"expected {dict(expected_split_counts)}"
            )
    records = unique_records(full_splits, full_split_root)

    if view_root.is_symlink() or (view_root.exists() and not view_root.is_dir()):
        raise FileExistsError(
            f"view root must be a real directory or absent: {view_root}"
        )

    pianomotion_records = set(
        unique_records(read_splits(pianomotion_split_root), pianomotion_split_root)
    )
    for split_path in pianomotion_extra_splits:
        pianomotion_records.update(read_records(split_path))
    pianovam_records = set(
        unique_records(read_splits(pianovam_split_root), pianovam_split_root)
    )
    ambiguous = pianomotion_records & pianovam_records
    if ambiguous:
        raise ValueError(
            "records belong to both source datasets: "
            + ", ".join(sorted(ambiguous))
        )

    midi_view = view_root / "midi"
    hand_view = view_hand_root or view_root / "airpiano_wilor_2p5d_30fps"
    links: list[tuple[Path, Path, str]] = []
    counts = {"pianomotion": 0, "pianovam": 0}
    for record_id in records:
        if record_id in pianomotion_records:
            midi_root = pianomotion_midi_root
            hand_root = pianomotion_hand_root
            dataset = "pianomotion"
        elif record_id in pianovam_records:
            midi_root = pianovam_midi_root
            hand_root = pianovam_hand_root
            dataset = "pianovam"
        else:
            raise ValueError(
                f"full-split record {record_id!r} belongs to neither dataset"
            )
        counts[dataset] += 1
        links.extend(
            (
                (
                    (midi_root / f"{record_id}.mid").resolve(strict=False),
                    midi_view / f"{record_id}.mid",
                    "midi",
                ),
                (
                    (hand_root / f"{record_id}.npz").resolve(strict=False),
                    hand_view / f"{record_id}.npz",
                    "hand",
                ),
            )
        )

    missing_midi = [
        source
        for source, _, kind in links
        if kind == "midi" and not source.is_file()
    ]
    if missing_midi:
        raise FileNotFoundError(f"required MIDI source not found: {missing_midi[0]}")
    for source, destination, _ in links:
        validate_destination(destination, source)

    for source, destination, _ in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_symlink():
            destination.symlink_to(source)

    missing_hands = sum(
        not source.is_file() for source, _, kind in links if kind == "hand"
    )
    return {
        "records": len(records),
        **counts,
        "midi_links": len(records),
        "hand_links": len(records),
        "missing_hand_sources": missing_hands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-split-root", type=Path, default=DEFAULT_FULL_SPLIT_ROOT)
    parser.add_argument(
        "--pianomotion-split-root",
        type=Path,
        default=DEFAULT_PIANOMOTION_SPLIT_ROOT,
    )
    parser.add_argument(
        "--pianovam-split-root", type=Path, default=DEFAULT_PIANOVAM_SPLIT_ROOT
    )
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW_ROOT)
    parser.add_argument(
        "--view-hand-root",
        type=Path,
        default=None,
        help="destination hand tree (defaults inside --view-root)",
    )
    parser.add_argument(
        "--pianomotion-extra-split",
        type=Path,
        action="append",
        default=[],
        help="additional PianoMotion record manifest; may be repeated",
    )
    parser.add_argument(
        "--pianomotion-midi-root",
        type=Path,
        default=DEFAULT_PIANOMOTION_MIDI_ROOT,
    )
    parser.add_argument(
        "--pianomotion-hand-root",
        type=Path,
        default=DEFAULT_PIANOMOTION_HAND_ROOT,
    )
    parser.add_argument(
        "--pianovam-midi-root", type=Path, default=DEFAULT_PIANOVAM_MIDI_ROOT
    )
    parser.add_argument(
        "--pianovam-hand-root", type=Path, default=DEFAULT_PIANOVAM_HAND_ROOT
    )
    args = parser.parse_args()

    summary = create_data_view(
        full_split_root=args.full_split_root,
        pianomotion_split_root=args.pianomotion_split_root,
        pianovam_split_root=args.pianovam_split_root,
        view_root=args.view_root,
        pianomotion_midi_root=args.pianomotion_midi_root,
        pianomotion_hand_root=args.pianomotion_hand_root,
        pianovam_midi_root=args.pianovam_midi_root,
        pianovam_hand_root=args.pianovam_hand_root,
        pianomotion_extra_splits=tuple(args.pianomotion_extra_split),
        view_hand_root=args.view_hand_root,
        expected_split_counts=expected_split_counts(args.full_split_root),
    )
    print("full data view ready:")
    for name, value in summary.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
