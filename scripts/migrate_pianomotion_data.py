"""Move PianoMotion postprocessed data to verified storage2 staging.

The utility is deliberately separate from AirPiano's runtime loaders.  It is
the only code that understands the legacy ``pianomotion-wilor-v1`` manifest
signature; published caches use the normal AirPiano schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/storage/jinxuan/data/PianoMotion10M")
DEFAULT_DESTINATION_ROOT = Path("/storage2/jinxuan/data/PianoMotion10M")
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "assets" / "splits" / "pianomotion10m"

LEGACY_RAW_SCHEMA = "pianomotion-wilor-v1"
CURRENT_RAW_SCHEMA = "airpiano-wilor-ragged-v1"
CATALOG_SCHEMA = "airpiano-data-migration-v1"
CATALOG_NAME = ".airpiano-migration-v1.json"
TOMBSTONE_PREFIX = ".airpiano-migration-delete."
CATALOG_RESERVE_BYTES = 1024 * 1024

RETAINED_TREES = (
    "processed_pianomotion_image_xy_wrist_z_30fps",
    "processed_pianomotion_mano_30fps",
    "processed_pianomotionnew_wilor_2p5d_30fps",
    "wilor",
)
CANONICAL_TREE = "airpiano_wilor_2p5d_30fps"
SPLIT_NAMES = ("train", "valid", "test")


def derive_staging_root(destination: str | Path) -> Path:
    """Return the hidden sibling used until an atomic final publication."""

    target = Path(destination).expanduser().resolve()
    return target.parent / f".{target.name}.airpiano-migration"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, value: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        if mode is not None:
            os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes before source data can be removed."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make every verified staged file and directory durable."""

    directories = [root]
    for path, is_directory in _iter_tree_entries(root):
        if is_directory:
            directories.append(path)
        else:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        directories,
        key=lambda item: len(item.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def converted_manifest(payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Return the deterministic current-schema form of one legacy manifest."""

    schema = payload["schema_version"]
    if schema not in {LEGACY_RAW_SCHEMA, CURRENT_RAW_SCHEMA}:
        raise ValueError(f"unsupported raw WiLoR schema at {path}: {schema!r}")
    converted = dict(payload)
    converted["schema_version"] = CURRENT_RAW_SCHEMA
    for field in ("subject_id", "video_id", "source_url"):
        converted.pop(field, None)
    return converted


def convert_manifest(path: str | Path) -> bool:
    """Atomically normalize one manifest; return whether its bytes changed."""

    manifest_path = Path(path)
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    converted = _json_bytes(converted_manifest(payload, manifest_path))
    if converted == original:
        return False
    mode = stat.S_IMODE(manifest_path.stat().st_mode)
    _atomic_bytes(manifest_path, converted, mode=mode)
    return True


def _read_split(path: Path) -> tuple[str, ...]:
    records: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            record = raw_line.strip()
            if not record or record.startswith("#") or record in seen:
                continue
            seen.add(record)
            records.append(record)
    return tuple(records)


def _stems(root: Path, suffix: str) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()[: -len(suffix)]
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    }


def _manifest_is_complete(path: Path, payload: dict[str, Any]) -> bool:
    if not (path.parent / "_SUCCESS").is_file():
        return False
    if payload.get("status") != "complete":
        return False
    decoded = int(payload.get("decoded_frame_count", 0))
    chunks = payload.get("chunks", [])
    inventory_end = int(chunks[-1]["frame_end"]) if chunks else 0
    video = payload.get("video", {})
    expected = round(
        float(video.get("duration_seconds", 0)) * float(video.get("fps", 0))
    )
    tolerance = max(2, math.ceil(expected * 0.002))
    return (
        decoded > 0
        and inventory_end == decoded
        and (expected <= 0 or decoded >= expected - tolerance)
    )


def _raw_inventory(root: Path) -> tuple[set[str], set[str], dict[str, str]]:
    manifests: set[str] = set()
    complete: set[str] = set()
    states: dict[str, str] = {}
    if not root.is_dir():
        return manifests, complete, states
    for path in root.rglob("manifest.json"):
        record = path.parent.relative_to(root).as_posix()
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.add(record)
        states[record] = str(payload.get("status"))
        if _manifest_is_complete(path, payload):
            complete.add(record)
    return manifests, complete, states


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate == candidate.parent:
            return candidate
        candidate = candidate.parent
    return candidate


def collect_report(
    source: str | Path,
    destination: str | Path,
    split_root: str | Path,
) -> dict[str, Any]:
    """Collect split/modality coverage without changing either filesystem."""

    source_root = Path(source).expanduser().resolve()
    target_root = Path(destination).expanduser().resolve()
    splits_root = Path(split_root).expanduser().resolve()
    splits = {name: _read_split(splits_root / f"{name}.txt") for name in SPLIT_NAMES}

    midi = _stems(source_root / "midi", ".mid")
    video = _stems(source_root / "Video", ".mp4")
    raw_root = source_root / "wilor"
    if not raw_root.is_dir():
        raw_root = target_root / "wilor"
    raw, complete_raw, raw_states = _raw_inventory(raw_root)
    canonical_root = source_root / CANONICAL_TREE
    if not canonical_root.is_dir():
        canonical_root = target_root / CANONICAL_TREE
    canonical = _stems(canonical_root, ".npz")

    rows: dict[str, dict[str, int]] = {}
    full_order: list[str] = []
    full_seen: set[str] = set()
    for name in SPLIT_NAMES:
        records = set(splits[name])
        rows[name] = {
            "full": len(records),
            "midi": len(records & midi),
            "video": len(records & video),
            "raw": len(records & raw),
            "complete_raw": len(records & complete_raw),
            "canonical": len(records & canonical),
        }
        for record in splits[name]:
            if record not in full_seen:
                full_seen.add(record)
                full_order.append(record)

    full = set(full_order)
    rows["total"] = {
        "full": len(full),
        "midi": len(full & midi),
        "video": len(full & video),
        "raw": len(full & raw),
        "complete_raw": len(full & complete_raw),
        "canonical": len(full & canonical),
    }
    incomplete = sorted((full & raw) - complete_raw)
    missing = {
        "midi": sorted(full - midi),
        "video": sorted(full - video),
        "raw": sorted(full - raw),
        "complete_raw": sorted(full - complete_raw),
        "canonical": sorted(full - canonical),
    }

    retained_bytes = sum(
        _tree_bytes(
            source_root / tree
            if (source_root / tree).exists()
            else target_root / tree
        )
        for tree in RETAINED_TREES
    )
    capacity_root = _nearest_existing(target_root.parent)
    capacity = shutil.disk_usage(capacity_root)
    return {
        "source": str(source_root),
        "destination": str(target_root),
        "staging": str(derive_staging_root(target_root)),
        "rows": rows,
        "raw_manifest_count": len(full & raw),
        "raw_states": {record: raw_states[record] for record in incomplete},
        "incomplete": incomplete,
        "missing": missing,
        "retained_bytes": retained_bytes,
        "destination_capacity_bytes": capacity.total,
        "destination_free_bytes": capacity.free,
        "destination_probe": str(capacity_root),
        "destination_writable": os.access(capacity_root, os.W_OK | os.X_OK),
    }


def _size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError


def _ratio(value: int, full: int) -> str:
    return f"{100.0 * value / full:.2f}%" if full else "0.00%"


def format_report(report: dict[str, Any], show_missing: bool = False) -> str:
    """Format the live inventory as a compact operational report."""

    lines = [
        f"source:      {report['source']}",
        f"staging:     {report['staging']}",
        f"destination: {report['destination']}",
        "",
        "Split | Full | MIDI | Video | Complete raw | Canonical",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for name in (*SPLIT_NAMES, "total"):
        row = report["rows"][name]
        label = name.capitalize()
        lines.append(
            f"{label} | {row['full']} | {row['midi']} | {row['video']} | "
            f"{row['complete_raw']} | {row['canonical']}"
        )
    total = report["rows"]["total"]
    lines.extend(
        [
            "",
            f"video coverage:       {total['video']}/{total['full']} "
            f"({_ratio(total['video'], total['full'])})",
            f"MIDI coverage:        {total['midi']}/{total['full']} "
            f"({_ratio(total['midi'], total['full'])})",
            f"raw manifest coverage: {total['raw']}/{total['full']} "
            f"({_ratio(total['raw'], total['full'])})",
            f"complete raw coverage: {total['complete_raw']}/{total['full']} "
            f"({_ratio(total['complete_raw'], total['full'])})",
            f"canonical coverage:   {total['canonical']}/{total['full']} "
            f"({_ratio(total['canonical'], total['full'])})",
            f"retained data:        {_size(report['retained_bytes'])} "
            f"({report['retained_bytes']} bytes)",
            f"destination free:     {_size(report['destination_free_bytes'])} "
            f"of {_size(report['destination_capacity_bytes'])} available at "
            f"{report['destination_probe']}",
            f"destination writable: {report['destination_writable']}",
        ]
    )
    if report["incomplete"]:
        lines.append("incomplete raw records:")
        lines.extend(
            f"  {record} ({report['raw_states'][record]})"
            for record in report["incomplete"]
        )
    if show_missing:
        for modality, records in report["missing"].items():
            lines.append(f"missing {modality} ({len(records)}):")
            lines.extend(f"  {record}" for record in records)
    return "\n".join(lines) + "\n"


def _iter_tree_entries(root: Path) -> Iterable[tuple[Path, bool]]:
    """Walk ordinary entries without descending through links or mountpoints."""

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        directory_names.sort()
        file_names.sort()
        parent = Path(directory)
        for name in directory_names:
            path = parent / name
            if path.is_symlink() or path.is_mount() or not path.is_dir():
                raise RuntimeError(
                    f"migration does not accept links or mountpoints: {path}"
                )
            yield path, True
        for name in file_names:
            path = parent / name
            if path.is_symlink() or path.is_mount() or not path.is_file():
                raise RuntimeError(
                    f"migration accepts only ordinary files and directories: {path}"
                )
            yield path, False


def _iter_files(root: Path) -> Iterable[Path]:
    for path, is_directory in _iter_tree_entries(root):
        if not is_directory:
            yield path


def _update_digest_header(digest: Any, relative: str, size: int) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")


def _update_digest_bytes(digest: Any, relative: str, data: bytes) -> None:
    _update_digest_header(digest, relative, len(data))
    digest.update(data)


def _update_digest_file(digest: Any, relative: str, path: Path) -> int:
    size = path.stat().st_size
    _update_digest_header(digest, relative, size)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return size


def _digest_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"retained migration tree is missing: {root}")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        size = _update_digest_file(digest, relative, path)
        files += 1
        total_bytes += size
    return {"sha256": digest.hexdigest(), "files": files, "bytes": total_bytes}


def _expected_tree_digest(source: Path, tree: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in _iter_files(source):
        relative = path.relative_to(source).as_posix()
        if tree == "wilor" and path.name == "manifest.json":
            payload = json.loads(path.read_bytes())
            data = _json_bytes(converted_manifest(payload, path))
            _update_digest_bytes(digest, relative, data)
            size = len(data)
        else:
            size = _update_digest_file(digest, relative, path)
        files += 1
        total_bytes += size
    return {"sha256": digest.hexdigest(), "files": files, "bytes": total_bytes}


def _required_free_bytes(source: Path, staging: Path) -> int:
    """Estimate final growth plus one worst-case rsync replacement file."""

    growth = 0
    replacement_peak = 0
    for path in _iter_files(source):
        destination = staging / path.relative_to(source)
        source_size = path.stat().st_size
        if destination.is_file():
            destination_size = destination.stat().st_size
            growth += max(0, source_size - destination_size)
            replacement_peak = max(replacement_peak, source_size)
        else:
            growth += source_size
    return growth + replacement_peak


def _run_rsync(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "--archive",
        "--partial",
        "--checksum",
        "--delete",
        "--fsync",
        "--human-readable",
    ]
    if os.isatty(1):
        command.append("--info=progress2")
    command.extend((f"{source}/", f"{destination}/"))
    subprocess.run(command, check=True)


def _validate_roots(source: Path, destination: Path) -> None:
    staging = derive_staging_root(destination)
    for left, right in ((source, destination), (source, staging)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError(
                "source, staging, and destination dataset roots must be disjoint"
            )
    if any(path == path.parent for path in (source, staging, destination)):
        raise ValueError("refusing a filesystem root as a dataset root")


def _remove_stale_catalog_temps(root: Path) -> None:
    """Remove only temp files left by an interrupted catalog publication."""

    if not root.exists():
        return
    if root.is_symlink() or root.is_mount() or not root.is_dir():
        raise RuntimeError(f"migration root must be an ordinary directory: {root}")
    pattern = f".{CATALOG_NAME}.*.tmp"
    for path in root.glob(pattern):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unexpected migration catalog temporary: {path}")
        path.unlink()
    _fsync_directory(root)


def _check_migration_root_entries(root: Path) -> None:
    if not root.exists():
        return
    if root.is_symlink() or root.is_mount() or not root.is_dir():
        raise RuntimeError(f"migration root must be an ordinary directory: {root}")
    allowed = set(RETAINED_TREES) | {CATALOG_NAME, CANONICAL_TREE}
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed)
    if unexpected:
        raise RuntimeError(f"unexpected migration-root entries: {unexpected}")
    for tree in RETAINED_TREES:
        path = root / tree
        if (
            path.is_symlink()
            or path.is_mount()
            or (path.exists() and not path.is_dir())
        ):
            raise RuntimeError(f"retained migration tree must be a directory: {path}")
        if path.exists():
            for _ in _iter_files(path):
                pass
    catalog = root / CATALOG_NAME
    if catalog.is_symlink() or (catalog.exists() and not catalog.is_file()):
        raise RuntimeError(f"migration catalog must be a regular file: {catalog}")
    canonical = root / CANONICAL_TREE
    if (
        canonical.is_symlink()
        or canonical.is_mount()
        or (canonical.exists() and not canonical.is_dir())
    ):
        raise RuntimeError(f"canonical migration path must be a directory: {canonical}")
    if canonical.exists() and any(canonical.iterdir()):
        raise RuntimeError(f"unexpected data in disposable canonical directory: {canonical}")


def _check_existing_catalog_owner(
    source: Path,
    destination: Path,
    staging: Path,
) -> None:
    """Do not repurpose another migration's resumable staging directory."""

    catalog = staging / CATALOG_NAME
    if not catalog.exists():
        return
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    if (
        payload["schema_version"] != CATALOG_SCHEMA
        or payload["source_root"] != str(source)
        or payload["destination_root"] != str(destination)
    ):
        raise RuntimeError(f"migration staging belongs to different roots: {catalog}")


def _catalog_payload(
    source: Path,
    destination: Path,
    tree_digests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA,
        "source_root": str(source),
        "destination_root": str(destination),
        "trees": tree_digests,
    }


def _write_catalog(
    source: Path,
    destination: Path,
    migrated_root: Path,
    tree_digests: dict[str, dict[str, Any]],
) -> None:
    payload = _catalog_payload(source, destination, tree_digests)
    _atomic_bytes(migrated_root / CATALOG_NAME, _json_bytes(payload), mode=0o644)


def _verify_catalog(
    source: Path, destination: Path, migrated_root: Path
) -> dict[str, dict[str, Any]]:
    catalog_path = migrated_root / CATALOG_NAME
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != CATALOG_SCHEMA:
        raise RuntimeError(f"unexpected migration catalog schema: {catalog_path}")
    if payload["source_root"] != str(source) or payload["destination_root"] != str(destination):
        raise RuntimeError(f"migration catalog roots do not match: {catalog_path}")
    actual = {tree: _digest_tree(migrated_root / tree) for tree in RETAINED_TREES}
    if actual != payload["trees"]:
        raise RuntimeError(f"migration catalog checksum verification failed: {migrated_root}")
    return actual


def _verify_source_digests(
    source: Path,
    migrated_digests: dict[str, dict[str, Any]],
    require_all: bool,
) -> None:
    differences: list[str] = []
    for tree in RETAINED_TREES:
        source_tree = source / tree
        if not source_tree.is_dir():
            if require_all:
                differences.append(f"missing source tree {source_tree}")
            continue
        if _expected_tree_digest(source_tree, tree) != migrated_digests[tree]:
            differences.append(f"checksum differs for {tree}")
    if differences:
        raise RuntimeError("migration verification failed: " + "; ".join(differences))


def copy_data(source: str | Path, destination: str | Path) -> Path:
    """Copy, checksum, and schema-convert retained trees without source removal."""

    source_root = Path(source).expanduser().resolve()
    target_root = Path(destination).expanduser().resolve()
    _validate_roots(source_root, target_root)
    if target_root.exists():
        raise RuntimeError(f"destination already exists: {target_root}")
    staging = derive_staging_root(target_root)
    _remove_stale_catalog_temps(staging)
    _check_migration_root_entries(staging)
    _check_existing_catalog_owner(source_root, target_root, staging)

    for tree in RETAINED_TREES:
        source_tree = source_root / tree
        if (
            not source_tree.is_dir()
            or source_tree.is_symlink()
            or source_tree.is_mount()
        ):
            raise FileNotFoundError(source_tree)

    capacity_root = _nearest_existing(target_root.parent)
    if not os.access(capacity_root, os.W_OK | os.X_OK):
        raise PermissionError(f"destination is not writable: {capacity_root}")
    required = sum(
        _required_free_bytes(source_root / tree, staging / tree)
        for tree in RETAINED_TREES
    ) + CATALOG_RESERVE_BYTES
    free = shutil.disk_usage(capacity_root).free
    if required > free:
        raise RuntimeError(
            f"insufficient destination capacity: need {_size(required)}, "
            f"have {_size(free)} at {capacity_root}"
        )

    staging.mkdir(parents=True, exist_ok=True)
    for tree in RETAINED_TREES:
        source_tree = source_root / tree
        staged_tree = staging / tree
        _run_rsync(source_tree, staged_tree)

    for manifest in (staging / "wilor").rglob("manifest.json"):
        convert_manifest(manifest)
    tree_digests = {
        tree: _digest_tree(staging / tree) for tree in RETAINED_TREES
    }
    _verify_source_digests(source_root, tree_digests, require_all=True)
    _write_catalog(source_root, target_root, staging, tree_digests)
    _fsync_tree(staging)
    return staging


def _safe_source_tree(source: Path, tree: str) -> Path:
    requested = source / tree
    resolved = requested.resolve()
    if (
        tree not in {*RETAINED_TREES, CANONICAL_TREE}
        or requested.is_symlink()
        or requested.is_mount()
        or (requested.exists() and not requested.is_dir())
        or resolved != requested
        or resolved.parent != source
    ):
        raise RuntimeError(f"refusing unexpected deletion target: {requested}")
    return requested


def _safe_tombstone(source: Path, tree: str) -> Path:
    path = source / f"{TOMBSTONE_PREFIX}{tree}"
    resolved = path.resolve()
    if (
        tree not in {*RETAINED_TREES, CANONICAL_TREE}
        or path.is_symlink()
        or path.is_mount()
        or (path.exists() and not path.is_dir())
        or resolved != path
        or resolved.parent != source
    ):
        raise RuntimeError(f"refusing unexpected cleanup tombstone: {path}")
    return path


def finalize_data(
    source: str | Path,
    destination: str | Path,
    confirm_delete_source: bool,
) -> Path:
    """Publish verified staging, then remove only explicitly approved sources."""

    if not confirm_delete_source:
        raise ValueError("finalize requires --confirm-delete-source")
    source_root = Path(source).expanduser().resolve()
    target_root = Path(destination).expanduser().resolve()
    _validate_roots(source_root, target_root)
    staging = derive_staging_root(target_root)
    cleanup_paths = [
        (
            _safe_source_tree(source_root, tree),
            _safe_tombstone(source_root, tree),
        )
        for tree in (*RETAINED_TREES, CANONICAL_TREE)
    ]
    for source_path, tombstone in cleanup_paths:
        if source_path.exists() and tombstone.exists():
            raise RuntimeError(
                f"both source and cleanup tombstone exist: "
                f"{source_path}, {tombstone}"
            )
    canonical_source = source_root / CANONICAL_TREE
    if canonical_source.exists():
        for _ in _iter_files(canonical_source):
            pass

    if target_root.exists():
        if staging.exists():
            raise RuntimeError(
                f"both staging and destination exist; refusing to merge: {staging}, {target_root}"
            )
        migrated_root = target_root
        _remove_stale_catalog_temps(migrated_root)
        _check_migration_root_entries(migrated_root)
        tree_digests = _verify_catalog(source_root, target_root, migrated_root)
        _verify_source_digests(source_root, tree_digests, require_all=False)
        _fsync_tree(migrated_root)
        _fsync_directory(target_root.parent)
    else:
        existing_tombstones = [
            str(tombstone) for _, tombstone in cleanup_paths if tombstone.exists()
        ]
        if existing_tombstones:
            raise RuntimeError(
                "cleanup tombstones exist before destination publication: "
                + ", ".join(existing_tombstones)
            )
        if not staging.is_dir():
            raise FileNotFoundError(staging)
        _remove_stale_catalog_temps(staging)
        _check_migration_root_entries(staging)
        tree_digests = _verify_catalog(source_root, target_root, staging)
        _verify_source_digests(source_root, tree_digests, require_all=True)
        if not (source_root / CANONICAL_TREE).is_dir():
            raise FileNotFoundError(source_root / CANONICAL_TREE)
        (staging / CANONICAL_TREE).mkdir(exist_ok=True)
        _fsync_tree(staging)
        os.replace(staging, target_root)
        _fsync_directory(target_root.parent)
        migrated_root = target_root

    # Every still-present source was checked before any deletion begins.
    for source_path, tombstone in cleanup_paths:
        if source_path.exists():
            os.replace(source_path, tombstone)
            _fsync_directory(source_root)
        if tombstone.exists():
            shutil.rmtree(tombstone)
            _fsync_directory(source_root)
    (migrated_root / CANONICAL_TREE).mkdir(exist_ok=True)
    _fsync_directory(migrated_root)
    return migrated_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report_parser = subparsers.add_parser("report", help="print a read-only inventory")
    report_parser.add_argument("--show-missing", action="store_true")
    subparsers.add_parser("copy", help="copy and verify data into hidden staging")
    finalize_parser = subparsers.add_parser(
        "finalize",
        help="publish staging and remove verified source trees",
        description=(
            "Publish verified staging and remove the five allowlisted source "
            "trees. Stop all readers and writers before running this command."
        ),
    )
    finalize_parser.add_argument(
        "--confirm-delete-source",
        action="store_true",
        help="acknowledge permanent cleanup of the verified source trees",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        report = collect_report(args.source_root, args.destination_root, args.split_root)
        print(format_report(report, show_missing=args.show_missing), end="")
    elif args.command == "copy":
        staging = copy_data(args.source_root, args.destination_root)
        print(f"migration copy verified in staging: {staging}")
    elif args.command == "finalize":
        destination = finalize_data(
            args.source_root,
            args.destination_root,
            args.confirm_delete_source,
        )
        print(f"migration finalized: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
