"""Safety and compatibility tests for the PianoMotion storage migration."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from airpiano.preprocessing.canonical import WiLoRToCanonical
from airpiano.preprocessing import video as video_module
from airpiano.preprocessing.wilor import cache as cache_module
from airpiano.preprocessing.wilor.cache import completed_output_is_valid
from scripts import migrate_pianomotion_data as migration


LEGACY_SCHEMA = "pianomotion-wilor-v1"
CURRENT_SCHEMA = "airpiano-wilor-ragged-v1"

MIGRATED_TREES = (
    "processed_pianomotion_image_xy_wrist_z_30fps",
    "processed_pianomotion_mano_30fps",
    "processed_pianomotionnew_wilor_2p5d_30fps",
    "wilor",
)
DISPOSABLE_TREE = "airpiano_wilor_2p5d_30fps"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_chunk(path: Path, frame_index: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_indices=np.asarray([frame_index], dtype=np.int64),
        frame_timestamps_seconds=np.asarray(
            [frame_index / 30.0], dtype=np.float64
        ),
        frame_detection_offsets=np.asarray([0, 0], dtype=np.int64),
        detector_confidence=np.empty((0,), dtype=np.float32),
        is_right=np.empty((0,), dtype=np.uint8),
        pred_keypoints_3d=np.empty((0, 21, 3), dtype=np.float32),
        pred_keypoints_2d_full=np.empty((0, 21, 2), dtype=np.float32),
        image_size_wh=np.empty((0, 2), dtype=np.float32),
    )


def _write_raw_record(
    path: Path,
    *,
    schema: str = LEGACY_SCHEMA,
    complete: bool = True,
    chunk_count: int = 1,
) -> dict[str, object]:
    chunks = []
    for index in range(chunk_count):
        relative = f"chunks/{index:06d}.npz"
        _write_chunk(path / relative, index)
        chunks.append(
            {
                "file": relative,
                "frame_start": index,
                "frame_end": index + 1,
                "detections": 0,
            }
        )

    manifest: dict[str, object] = {
        "schema_version": schema,
        "status": "complete" if complete else "processing",
        "subject_id": "fixture subject",
        "video_id": "take.03",
        "source_url": "https://example.invalid/take.03",
        "video_path": "/old/storage/Video/fixture subject/take.03.mp4",
        "video": {
            "fps": 30.0,
            "duration_seconds": chunk_count / 30.0,
            "width": 1920,
            "height": 1080,
        },
        "model": {"checkpoint_sha256": "model-checksum"},
        "inference": {"chunk_frames": 1, "device": "cuda:0"},
        "chunks": chunks if complete else [],
        "created_at": "2026-09-01T00:00:00+00:00",
        "updated_at": "2026-09-01T00:01:00+00:00",
    }
    if complete:
        manifest["decoded_frame_count"] = chunk_count
        manifest["detection_count"] = 0
        manifest["completed_at"] = "2026-09-01T00:01:00+00:00"
    _write_text(path / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    if complete:
        (path / "_SUCCESS").touch()
    return manifest


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class _Corpus:
    """Small source corpus whose coverage differs in every useful category."""

    def __init__(self, root: Path) -> None:
        self.workspace = root
        self.source = root / "source" / "PianoMotion10M"
        self.destination = root / "destination" / "PianoMotion10M"
        self.destination.parent.mkdir(parents=True)
        self.staging = (
            self.destination.parent
            / f".{self.destination.name}.airpiano-migration"
        )
        self.splits = root / "splits" / "pianomotion10m"

        self.train = (
            "alpha/take.01",
            "performer one/session.v1/take.03",
            "missing/all",
        )
        self.valid = ("beta/incomplete",)
        self.test = ("gamma/complete", "delta/video-only")
        for split, records in (
            ("train", self.train),
            ("valid", self.valid),
            ("test", self.test),
        ):
            _write_text(self.splits / f"{split}.txt", "\n".join(records) + "\n")

        for record in (*self.train[:2], *self.valid, *self.test):
            _write_text(self.source / "Video" / f"{record}.mp4", f"video:{record}")
        for record in (*self.train[:2], *self.valid):
            _write_text(self.source / "midi" / f"{record}.mid", f"midi:{record}")

        self.complete_records = (
            self.train[0],
            self.train[1],
            self.test[0],
        )
        for record in self.complete_records:
            _write_raw_record(self.source / "wilor" / record)
        _write_raw_record(
            self.source / "wilor" / self.valid[0],
            complete=False,
            chunk_count=3,
        )
        # Out-of-split data and historical root logs are copied, but do not
        # inflate report intersections.
        _write_raw_record(self.source / "wilor" / "outside" / "ignored")
        _write_text(
            self.source / "wilor" / "inference_status.premerge.jsonl",
            '{"status":"historical"}\n',
        )

        for record in (self.train[0], self.test[0]):
            _write_text(
                self.source / DISPOSABLE_TREE / f"{record}.npz",
                f"canonical:{record}",
            )

        for index, tree in enumerate(MIGRATED_TREES[:-1]):
            _write_text(
                self.source / tree / "nested" / f"legacy.{index}.bin",
                f"legacy payload {index}",
            )
        _write_text(self.source / "annotation" / "keep.txt", "must remain")
        _write_text(self.source / "unrelated.bin", "must also remain")

    def arguments(self, command: str, *extra: str) -> list[str]:
        return [
            "--source-root",
            str(self.source),
            "--destination-root",
            str(self.destination),
            "--split-root",
            str(self.splits),
            command,
            *extra,
        ]

    def run(self, command: str, *extra: str) -> tuple[object, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = migration.main(self.arguments(command, *extra))
        return result, stdout.getvalue(), stderr.getvalue()


class DataMigrationReportTest(unittest.TestCase):
    def test_report_counts_splits_missing_records_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            before = _snapshot(corpus.source)

            _, output, _ = corpus.run("report", "--show-missing")

            self.assertEqual(_snapshot(corpus.source), before)
            self.assertFalse(corpus.staging.exists())
            self.assertFalse(corpus.destination.exists())
            # Split rows are deliberately asymmetric so these assertions also
            # catch incorrect suffix stripping for nested and dotted stems.
            normalized = " ".join(output.lower().replace("|", " ").split())
            self.assertRegex(normalized, r"train\s+3\s+2\s+2\s+2\s+1")
            self.assertRegex(normalized, r"valid\s+1\s+1\s+1\s+0\s+0")
            self.assertRegex(normalized, r"test\s+2\s+0\s+2\s+1\s+1")
            self.assertRegex(normalized, r"total\s+6\s+3\s+5\s+3\s+2")
            self.assertRegex(
                output,
                r"(?i)midi coverage:\s+3/6\s+\(50(?:\.0+)?%\)",
            )
            self.assertGreaterEqual(output.count("%"), 4)
            self.assertRegex(output, r"83(?:\.3+)?%")
            self.assertRegex(output, r"50(?:\.0+)?%")
            self.assertIn("missing/all", output)
            self.assertIn("delta/video-only", output)
            self.assertIn("beta/incomplete", output)
            self.assertRegex(output, r"(?i)(bytes|KiB|MiB|GiB|\d+ B)")
            self.assertRegex(output, r"(?i)(available|capacity|free)")


class ManifestMigrationTest(unittest.TestCase):
    def test_conversion_is_deterministic_atomic_and_removes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "record" / "manifest.json"
            original = _write_raw_record(manifest_path.parent)

            before = manifest_path.read_bytes()
            with patch.object(
                migration.os,
                "replace",
                side_effect=OSError("simulated publication interruption"),
            ):
                with self.assertRaisesRegex(OSError, "publication interruption"):
                    migration.convert_manifest(manifest_path)
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertEqual(list(manifest_path.parent.rglob("*.tmp")), [])

            migration.convert_manifest(manifest_path)
            converted_bytes = manifest_path.read_bytes()
            converted = json.loads(converted_bytes)

            self.assertEqual(converted["schema_version"], CURRENT_SCHEMA)
            self.assertTrue(
                {"subject_id", "video_id", "source_url"}.isdisjoint(converted)
            )
            for key in (
                "status",
                "video_path",
                "video",
                "model",
                "inference",
                "chunks",
                "decoded_frame_count",
                "detection_count",
                "created_at",
                "updated_at",
                "completed_at",
            ):
                self.assertEqual(converted[key], original[key])
            self.assertEqual(list(manifest_path.parent.glob("*.tmp")), [])

            migration.convert_manifest(manifest_path)
            self.assertEqual(manifest_path.read_bytes(), converted_bytes)

    def test_unknown_manifest_schema_is_rejected_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            _write_text(
                manifest_path,
                json.dumps({"schema_version": "unknown-v9", "status": "complete"}),
            )
            before = manifest_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "unknown-v9"):
                migration.convert_manifest(manifest_path)

            self.assertEqual(manifest_path.read_bytes(), before)


class DataMigrationCopyTest(unittest.TestCase):
    def test_copy_rejects_staging_catalog_owned_by_another_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            catalog = {
                "schema_version": migration.CATALOG_SCHEMA,
                "source_root": str(
                    (Path(directory) / "different-source").resolve()
                ),
                "destination_root": str(corpus.destination.resolve()),
                "trees": {
                    tree: {"sha256": "0" * 64, "files": 0, "bytes": 0}
                    for tree in MIGRATED_TREES
                },
            }
            _write_text(
                corpus.staging / migration.CATALOG_NAME,
                json.dumps(catalog, indent=2, sort_keys=True) + "\n",
            )
            source_before = _snapshot(corpus.source)
            staging_before = _snapshot(corpus.staging)

            with patch.object(migration, "_run_rsync") as run_rsync:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "(?i)(catalog|source|root|owned)",
                ):
                    corpus.run("copy")

            run_rsync.assert_not_called()
            self.assertEqual(_snapshot(corpus.source), source_before)
            self.assertEqual(_snapshot(corpus.staging), staging_before)
            self.assertFalse(corpus.destination.exists())

    def test_source_equal_to_derived_staging_is_rejected_before_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "volume" / "PianoMotion10M"
            source = migration.derive_staging_root(destination)
            _write_text(source / "sentinel.txt", "unchanged")
            before = _snapshot(source)

            with patch.object(migration, "_run_rsync") as run_rsync:
                with self.assertRaisesRegex(ValueError, "(?i)disjoint"):
                    migration.copy_data(source, destination)

            run_rsync.assert_not_called()
            self.assertEqual(_snapshot(source), before)
            self.assertFalse(destination.exists())

    def test_staged_retained_tree_symlink_is_rejected_before_rsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            external = Path(directory) / "external-victim"
            _write_text(external / "keep.txt", "must remain")
            corpus.staging.mkdir()
            (corpus.staging / MIGRATED_TREES[0]).symlink_to(
                external,
                target_is_directory=True,
            )
            before = _snapshot(corpus.source)

            with patch.object(migration, "_run_rsync") as run_rsync:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "(?i)retained.*directory",
                ):
                    corpus.run("copy")

            run_rsync.assert_not_called()
            self.assertEqual(
                (external / "keep.txt").read_text(encoding="utf-8"),
                "must remain",
            )
            self.assertEqual(_snapshot(corpus.source), before)
            self.assertTrue((corpus.staging / MIGRATED_TREES[0]).is_symlink())

    def test_rsync_requests_durable_destination_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()

            with patch.object(migration.subprocess, "run") as run:
                migration._run_rsync(source, destination)

            command = run.call_args.args[0]
            self.assertIn("--fsync", command)
            self.assertIn("--checksum", command)
            self.assertIn("--partial", command)

    def test_copy_recovers_only_tightly_named_catalog_temporaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            stale = (
                corpus.staging
                / f".{migration.CATALOG_NAME}.abcdefgh.tmp"
            )
            _write_text(stale, "incomplete catalog")

            corpus.run("copy")

            self.assertFalse(stale.exists())
            self.assertTrue((corpus.staging / migration.CATALOG_NAME).is_file())

        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            unrelated = corpus.staging / "unrelated.tmp"
            _write_text(unrelated, "must not be silently removed")
            before = _snapshot(corpus.source)

            with self.assertRaisesRegex(RuntimeError, "(?i)unexpected"):
                corpus.run("copy")

            self.assertEqual(
                unrelated.read_text(encoding="utf-8"),
                "must not be silently removed",
            )
            self.assertEqual(_snapshot(corpus.source), before)
            self.assertFalse(corpus.destination.exists())

    def test_copy_rejects_unwritable_destination_before_creating_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            before = _snapshot(corpus.source)

            with patch.object(migration.os, "access", return_value=False):
                with self.assertRaisesRegex(PermissionError, "(?i)not writable"):
                    corpus.run("copy")

            self.assertFalse(corpus.staging.exists())
            self.assertFalse(corpus.destination.exists())
            self.assertEqual(_snapshot(corpus.source), before)

    def test_copy_rejects_insufficient_capacity_before_creating_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            before = _snapshot(corpus.source)

            with patch.object(
                migration.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(free=0),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "(?i)insufficient.*capacity",
                ):
                    corpus.run("copy")

            self.assertFalse(corpus.staging.exists())
            self.assertFalse(corpus.destination.exists())
            self.assertEqual(_snapshot(corpus.source), before)

    def test_copy_resumes_repairs_partial_files_and_never_changes_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            before = _snapshot(corpus.source)
            copied_tree = MIGRATED_TREES[0]
            partial = corpus.staging / copied_tree / "nested" / "legacy.0.bin"
            source_payload = corpus.source / copied_tree / "nested" / "legacy.0.bin"
            source_stat = source_payload.stat()
            _write_text(
                partial,
                "x" * len(source_payload.read_bytes()),
            )
            # Size and mtime match exactly, so only a content checksum can
            # distinguish this interrupted/corrupt staged payload.
            os.utime(
                partial,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )

            corpus.run("copy")

            self.assertEqual(_snapshot(corpus.source), before)
            self.assertEqual(partial.read_text(encoding="utf-8"), "legacy payload 0")
            for tree in MIGRATED_TREES:
                self.assertTrue((corpus.staging / tree).is_dir())
            self.assertFalse((corpus.staging / DISPOSABLE_TREE).exists())
            self.assertEqual(
                (
                    corpus.staging
                    / "wilor"
                    / "inference_status.premerge.jsonl"
                ).read_text(encoding="utf-8"),
                '{"status":"historical"}\n',
            )

            incomplete = corpus.staging / "wilor" / corpus.valid[0]
            self.assertFalse((incomplete / "_SUCCESS").exists())
            self.assertEqual(len(list((incomplete / "chunks").glob("*.npz"))), 3)
            for manifest_path in (corpus.staging / "wilor").rglob(
                "manifest.json"
            ):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema_version"], CURRENT_SCHEMA)
                self.assertTrue(
                    {"subject_id", "video_id", "source_url"}.isdisjoint(manifest)
                )

            # A second pass must accept its own converted staging manifests and
            # leave the source byte-for-byte untouched.
            first_staging = _snapshot(corpus.staging)
            corpus.run("copy")
            self.assertEqual(_snapshot(corpus.staging), first_staging)
            self.assertEqual(_snapshot(corpus.source), before)

    def test_finalize_rejects_checksum_difference_then_copy_repairs_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")
            staged_payload = (
                corpus.staging
                / MIGRATED_TREES[1]
                / "nested"
                / "legacy.1.bin"
            )
            staged_payload.write_text("corrupt", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "(?i)(checksum|differ|verify)"):
                corpus.run("finalize", "--confirm-delete-source")

            self.assertFalse(corpus.destination.exists())
            self.assertTrue((corpus.source / MIGRATED_TREES[1]).exists())

            corpus.run("copy")
            corpus.run("finalize", "--confirm-delete-source")
            self.assertEqual(
                (
                    corpus.destination
                    / MIGRATED_TREES[1]
                    / "nested"
                    / "legacy.1.bin"
                ).read_text(encoding="utf-8"),
                "legacy payload 1",
            )


class DataMigrationFinalizeTest(unittest.TestCase):
    def test_nested_canonical_mountpoint_is_rejected_before_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")
            mountpoint = corpus.source / DISPOSABLE_TREE / "mounted-volume"
            _write_text(mountpoint / "keep.npz", "mounted data")
            source_before = _snapshot(corpus.source)
            staging_before = _snapshot(corpus.staging)
            path_type = type(mountpoint)
            real_is_mount = path_type.is_mount

            def narrow_is_mount(path: Path) -> bool:
                return path == mountpoint or real_is_mount(path)

            with (
                patch.object(path_type, "is_mount", new=narrow_is_mount),
                patch.object(migration.os, "replace") as replace,
            ):
                with self.assertRaisesRegex(RuntimeError, "(?i)mountpoint"):
                    corpus.run("finalize", "--confirm-delete-source")

            replace.assert_not_called()
            self.assertFalse(corpus.destination.exists())
            self.assertEqual(_snapshot(corpus.staging), staging_before)
            self.assertEqual(_snapshot(corpus.source), source_before)
            self.assertEqual(
                (mountpoint / "keep.npz").read_text(encoding="utf-8"),
                "mounted data",
            )

    def test_cleanup_resumes_after_partial_tombstone_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")
            real_rmtree = shutil.rmtree
            interrupted: list[Path] = []

            def partial_rmtree(path: str | Path) -> None:
                tombstone = Path(path)
                victim = next(
                    item for item in tombstone.rglob("*") if item.is_file()
                )
                victim.unlink()
                interrupted.append(tombstone)
                raise OSError("simulated cleanup interruption")

            with patch.object(
                migration.shutil,
                "rmtree",
                side_effect=partial_rmtree,
            ):
                with self.assertRaisesRegex(OSError, "cleanup interruption"):
                    corpus.run("finalize", "--confirm-delete-source")

            first_tree = MIGRATED_TREES[0]
            first_tombstone = (
                corpus.source / f"{migration.TOMBSTONE_PREFIX}{first_tree}"
            )
            self.assertEqual(interrupted, [first_tombstone])
            self.assertTrue(corpus.destination.is_dir())
            self.assertFalse((corpus.source / first_tree).exists())
            self.assertTrue(first_tombstone.is_dir())
            for tree in MIGRATED_TREES[1:]:
                self.assertTrue((corpus.source / tree).is_dir())

            self.assertIs(migration.shutil.rmtree, real_rmtree)
            corpus.run("finalize", "--confirm-delete-source")

            for tree in (*MIGRATED_TREES, DISPOSABLE_TREE):
                self.assertFalse((corpus.source / tree).exists())
                self.assertFalse(
                    (
                        corpus.source
                        / f"{migration.TOMBSTONE_PREFIX}{tree}"
                    ).exists()
                )
            self.assertTrue((corpus.destination / "wilor").is_dir())
            self.assertTrue((corpus.source / "annotation" / "keep.txt").is_file())

    def test_publication_directory_is_synced_before_source_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")
            events: list[tuple[str, Path]] = []
            real_replace = os.replace
            real_fsync_directory = migration._fsync_directory

            def recording_replace(source: str | Path, target: str | Path) -> None:
                source_path = Path(source)
                target_path = Path(target)
                if source_path == corpus.staging and target_path == corpus.destination:
                    events.append(("publish", target_path))
                elif (
                    source_path.parent == corpus.source
                    and target_path.parent == corpus.source
                    and target_path.name.startswith(migration.TOMBSTONE_PREFIX)
                ):
                    events.append(("source-rename", source_path))
                real_replace(source, target)

            def recording_fsync_directory(path: str | Path) -> None:
                events.append(("directory-fsync", Path(path)))
                real_fsync_directory(Path(path))

            with (
                patch.object(
                    migration.os,
                    "replace",
                    side_effect=recording_replace,
                ),
                patch.object(
                    migration,
                    "_fsync_directory",
                    side_effect=recording_fsync_directory,
                ),
            ):
                corpus.run("finalize", "--confirm-delete-source")

            publication = events.index(("publish", corpus.destination))
            first_source_rename = next(
                index
                for index, event in enumerate(events)
                if event[0] == "source-rename"
            )
            publication_sync = next(
                index
                for index, event in enumerate(events)
                if index > publication
                and event == ("directory-fsync", corpus.destination.parent)
            )
            self.assertLess(publication, publication_sync)
            self.assertLess(publication_sync, first_source_rename)

    def test_confirmation_is_required_and_only_allowlisted_sources_are_deleted(
        self,
    ) -> None:
        self.assertEqual(set(migration.RETAINED_TREES), set(MIGRATED_TREES))
        self.assertEqual(migration.CANONICAL_TREE, DISPOSABLE_TREE)
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")

            with self.assertRaises((SystemExit, RuntimeError, ValueError)):
                corpus.run("finalize")

            self.assertFalse(corpus.destination.exists())
            for tree in (*MIGRATED_TREES, DISPOSABLE_TREE):
                self.assertTrue((corpus.source / tree).exists())

            # A crash can occur after this empty directory is created but
            # before staging is renamed. Finalize must be safe to rerun.
            (corpus.staging / DISPOSABLE_TREE).mkdir()
            corpus.run("finalize", "--confirm-delete-source")

            self.assertFalse(corpus.staging.exists())
            for tree in MIGRATED_TREES:
                self.assertTrue((corpus.destination / tree).is_dir())
                self.assertFalse((corpus.source / tree).exists())
            self.assertTrue((corpus.destination / DISPOSABLE_TREE).is_dir())
            self.assertEqual(list((corpus.destination / DISPOSABLE_TREE).iterdir()), [])
            self.assertFalse((corpus.source / DISPOSABLE_TREE).exists())
            self.assertEqual(
                (corpus.source / "annotation" / "keep.txt").read_text(
                    encoding="utf-8"
                ),
                "must remain",
            )
            self.assertEqual(
                (corpus.source / "unrelated.bin").read_text(encoding="utf-8"),
                "must also remain",
            )
            self.assertTrue((corpus.source / "Video").is_dir())
            self.assertTrue((corpus.source / "midi").is_dir())

    def test_finalize_resumes_after_publication_and_partial_source_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")

            # Reproduce an interruption after atomic publication and one source
            # deletion, before the remaining allowlist cleanup completed.
            corpus.staging.rename(corpus.destination)
            (corpus.destination / DISPOSABLE_TREE).mkdir()
            shutil.rmtree(corpus.source / MIGRATED_TREES[0])

            corpus.run("finalize", "--confirm-delete-source")

            for tree in (*MIGRATED_TREES, DISPOSABLE_TREE):
                self.assertFalse((corpus.source / tree).exists())
            self.assertTrue((corpus.destination / "wilor").is_dir())
            self.assertTrue((corpus.source / "annotation" / "keep.txt").is_file())

    def test_published_destination_contamination_stops_source_cleanup(self) -> None:
        for contamination in ("top-level", "canonical"):
            with self.subTest(contamination=contamination):
                with tempfile.TemporaryDirectory() as directory:
                    corpus = _Corpus(Path(directory))
                    corpus.run("copy")
                    corpus.staging.rename(corpus.destination)
                    canonical = corpus.destination / DISPOSABLE_TREE
                    canonical.mkdir()
                    if contamination == "top-level":
                        _write_text(
                            corpus.destination / "unexpected.txt",
                            "must not be blessed",
                        )
                    else:
                        _write_text(canonical / "old.npz", "must be regenerated")
                    before = _snapshot(corpus.source)

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "(?i)(unexpected|canonical|verification)",
                    ):
                        corpus.run("finalize", "--confirm-delete-source")

                    self.assertEqual(_snapshot(corpus.source), before)

    def test_all_deletion_targets_are_validated_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            corpus.run("copy")
            shutil.rmtree(corpus.source / DISPOSABLE_TREE)
            # A symlink to a sibling under the same source root is especially
            # dangerous: a parent-only path check would accept it and delete
            # the unrelated sibling after resolving the link.
            (corpus.source / DISPOSABLE_TREE).symlink_to(
                corpus.source / "annotation",
                target_is_directory=True,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "(?i)(deletion|symlink|refusing)",
            ):
                corpus.run("finalize", "--confirm-delete-source")

            self.assertFalse(corpus.destination.exists())
            for tree in MIGRATED_TREES:
                self.assertTrue((corpus.source / tree).is_dir())
            self.assertTrue((corpus.source / "annotation" / "keep.txt").is_file())

    def test_catalog_rejects_a_missing_retained_directory_even_when_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            tree = MIGRATED_TREES[0]
            (corpus.source / tree / "nested" / "legacy.0.bin").unlink()
            corpus.run("copy")
            corpus.staging.rename(corpus.destination)
            (corpus.destination / DISPOSABLE_TREE).mkdir()
            shutil.rmtree(corpus.destination / tree)

            with self.assertRaisesRegex(
                (RuntimeError, FileNotFoundError),
                "(?i)(catalog|directory|missing|verification)",
            ):
                corpus.run("finalize", "--confirm-delete-source")

            for source_tree in MIGRATED_TREES:
                self.assertTrue((corpus.source / source_tree).is_dir())

    def test_unexpected_existing_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _Corpus(Path(directory))
            _write_text(corpus.destination / "unexpected.txt", "do not overwrite")

            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "(?i)(destination|target|exist)"
            ):
                corpus.run("copy")

            self.assertEqual(
                (corpus.destination / "unexpected.txt").read_text(encoding="utf-8"),
                "do not overwrite",
            )
            self.assertFalse(corpus.staging.exists())


class MigratedRuntimeCompatibilityTest(unittest.TestCase):
    def test_converted_incomplete_cache_resumes_orphan_chunks_in_producer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            _write_raw_record(raw, complete=False, chunk_count=3)
            orphan_chunks = sorted((raw / "chunks").glob("*.npz"))
            orphan_bytes = {path.name: path.read_bytes() for path in orphan_chunks}
            migration.convert_manifest(raw / "manifest.json")

            class Reader:
                def __init__(self) -> None:
                    self.position = 0
                    self.grabbed = 0
                    self.released = False

                def grab(self) -> bool:
                    if self.position >= 4:
                        return False
                    self.position += 1
                    self.grabbed += 1
                    return True

                def read(self) -> tuple[bool, np.ndarray | None]:
                    if self.position >= 4:
                        return False, None
                    frame = np.zeros((2, 3, 3), dtype=np.uint8)
                    self.position += 1
                    return True, frame

                def timestamp_seconds(self) -> float:
                    return (self.position - 1) / 30.0

                def release(self) -> None:
                    self.released = True

            class Backend:
                def __init__(self) -> None:
                    self.calls: list[list[int]] = []

                def model_metadata(self) -> dict[str, str]:
                    return {"model": "synthetic"}

                def inference_metadata(self, chunk_frames: int) -> dict[str, int]:
                    return {"chunk_frames": chunk_frames}

                def build_chunk(self, indices, timestamps, frames):
                    del frames
                    self.calls.append(list(indices))
                    return {
                        "frame_indices": np.asarray(indices, dtype=np.int64),
                        "frame_timestamps_seconds": np.asarray(
                            timestamps,
                            dtype=np.float64,
                        ),
                        "frame_detection_offsets": np.zeros(
                            len(indices) + 1,
                            dtype=np.int64,
                        ),
                        "detector_confidence": np.empty((0,), dtype=np.float32),
                        "is_right": np.empty((0,), dtype=np.uint8),
                        "pred_keypoints_3d": np.empty(
                            (0, 21, 3),
                            dtype=np.float32,
                        ),
                        "pred_keypoints_2d_full": np.empty(
                            (0, 21, 2),
                            dtype=np.float32,
                        ),
                        "image_size_wh": np.empty((0, 2), dtype=np.float32),
                    }

            reader = Reader()
            backend = Backend()
            video = root / "video.mp4"
            video.touch()
            metadata = {
                "codec_name": "h264",
                "width": 3,
                "height": 2,
                "fps": 30.0,
                "duration_seconds": 4 / 30.0,
            }
            with (
                patch.object(video_module, "WiLoRBackend", return_value=backend),
                patch.object(cache_module, "probe_video", return_value=metadata),
                patch.object(
                    cache_module,
                    "open_video_reader",
                    return_value=reader,
                ),
            ):
                producer = video_module.VideoToWiLoR(
                    wilor_repo=root / "unused-repository",
                    device="cpu",
                    chunk_frames=1,
                )
                result = producer.run(video, raw)

            self.assertEqual(result, raw.resolve())
            self.assertEqual(reader.grabbed, 3)
            self.assertTrue(reader.released)
            self.assertEqual(backend.calls, [[3]])
            for chunk in orphan_chunks:
                self.assertEqual(chunk.read_bytes(), orphan_bytes[chunk.name])
            manifest = cache_module.read_manifest(raw)
            self.assertEqual(manifest["schema_version"], CURRENT_SCHEMA)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["chunks"]), 4)
            self.assertTrue((raw / "_SUCCESS").is_file())

    def test_migrated_cache_is_accepted_while_unmigrated_cache_stays_strict(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            manifest_path = raw / "manifest.json"
            _write_raw_record(raw)

            with self.assertRaisesRegex(ValueError, LEGACY_SCHEMA):
                completed_output_is_valid(raw)
            with self.assertRaisesRegex(ValueError, LEGACY_SCHEMA):
                WiLoRToCanonical().run(raw, root / "legacy.npz")

            migration.convert_manifest(manifest_path)

            self.assertTrue(completed_output_is_valid(raw))
            with (
                patch.object(video_module, "WiLoRBackend", return_value=object()),
                patch.object(video_module, "process_video") as process_video,
            ):
                producer = video_module.VideoToWiLoR(
                    wilor_repo=root / "unused-repository",
                    device="cpu",
                )
                self.assertEqual(
                    producer.run(root / "video-no-longer-needed.mp4", raw),
                    raw.resolve(),
                )
            process_video.assert_not_called()

            output = WiLoRToCanonical().run(raw, root / "migrated.npz")
            self.assertTrue(output.is_file())
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["pose"].shape, (1, 2, 21, 3))
                self.assertEqual(payload["presence"].tolist(), [[False, False]])


if __name__ == "__main__":
    unittest.main()
