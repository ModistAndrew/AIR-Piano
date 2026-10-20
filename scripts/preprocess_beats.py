#!/usr/bin/env python3
"""Detect beats in audio with Beat This! and DBN postprocessing."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path


INPUT_SUFFIXES = frozenset({
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mka", ".mp3",
    ".mp4", ".ogg", ".opus", ".wav", ".webm", ".wma",
})
OUTPUT_SUFFIX = ".txt"


def load_processor(device: str):
    from beat_this.inference import File2Beats

    return File2Beats(
        checkpoint_path="final0", device=device, float16=False, dbn=True,
    )


def valid_output(destination: Path) -> bool:
    """Accept a nonempty beat file with increasing timestamps."""
    previous_timestamp = None
    try:
        with destination.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                fields = raw_line.split()
                if len(fields) != 2:
                    return False
                timestamp = float(fields[0])
                int(fields[1])
                if not math.isfinite(timestamp) or timestamp < 0:
                    return False
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    return False
                previous_timestamp = timestamp
    except (OSError, UnicodeError, ValueError):
        return False
    return previous_timestamp is not None


def process_file(processor, source: Path, destination: Path) -> None:
    from beat_this.utils import save_beat_tsv

    beats, downbeats = processor(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.txt", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_beat_tsv(beats, downbeats, temporary)
        # Beat This catches Ctrl-C during writing and deletes the temporary file.
        if not temporary.exists():
            raise KeyboardInterrupt("Beat serialization interrupted")
        if not valid_output(temporary):
            raise ValueError(f"Beat This! produced invalid or empty beats for {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

# Each runner keeps this small loop local so it can be used independently.
def discover_inputs(root):
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in INPUT_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def output_path(source, input_root, output_root):
    return output_root / source.relative_to(input_root).with_suffix(OUTPUT_SUFFIX)


def write_report(output_root, report):
    import json
    import os
    import tempfile

    fd, name = tempfile.mkstemp(prefix=".report.", suffix=".json", dir=output_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(name, output_root / "report.json")
    finally:
        Path(name).unlink(missing_ok=True)


def record_failure(report, source, destination, error):
    report["failed"] += 1
    report["failures"].append({
        "input_path": str(source),
        "output_path": str(destination),
        "error_type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    })


def print_progress(report):
    print(
        f"[progress] discovered={report['discovered']} processed={report['processed']} "
        f"skipped={report['skipped']} failed={report['failed']}",
        flush=True,
    )


def main(argv=None):
    import argparse
    import sys
    from collections import Counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    report = {
        "status": "running",
        "discovered": 0, "processed": 0, "skipped": 0, "failed": 0,
        "failures": [],
    }
    current = None
    try:
        sources = discover_inputs(input_root)
        output_root.mkdir(parents=True, exist_ok=True)
        jobs = [(source, output_path(source, input_root, output_root)) for source in sources]
        report["discovered"] = len(jobs)
        destinations = Counter(destination for _, destination in jobs)
        write_report(output_root, report)
        processor = None
        for source, destination in jobs:
            current = (source, destination)
            if destinations[destination] > 1:
                record_failure(report, source, destination, ValueError(
                    f"Multiple inputs map to {destination}"
                ))
            elif valid_output(destination):
                report["skipped"] += 1
            else:
                if processor is None:
                    # Initialization errors are setup failures, not per-file errors.
                    processor = load_processor(args.device)
                print(f"[process] {source.relative_to(input_root)}", flush=True)
                try:
                    process_file(processor, source, destination)
                    report["processed"] += 1
                except Exception as error:
                    record_failure(report, source, destination, error)
                    print(f"[failed] {source}: {error}", file=sys.stderr, flush=True)
            current = None
            write_report(output_root, report)
            print_progress(report)
        report["status"] = "completed_with_failures" if report["failed"] else "complete"
        return 1 if report["failed"] else 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        if current is not None:
            record_failure(report, *current, KeyboardInterrupt("Processing interrupted"))
        return 130
    except Exception as error:
        report["status"] = "setup_failed"
        report["setup_error"] = {
            "error_type": type(error).__name__, "message": str(error),
        }
        print(f"[setup failed] {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            write_report(output_root, report)
        except OSError as error:
            print(f"[report failed] {error}", file=sys.stderr, flush=True)
        print_progress(report)


if __name__ == "__main__":
    raise SystemExit(main())
