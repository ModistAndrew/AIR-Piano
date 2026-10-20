#!/usr/bin/env python3
"""Extract raw WiLoR caches from an MP4 directory, preserving relative paths.

Usage: python scripts/preprocess_wilor.py INPUT_ROOT OUTPUT_ROOT [--device cuda:1]
Complete AirPiano caches are skipped; partial caches resume saved chunks.
Failures are listed in OUTPUT_ROOT/report.json. Unsupported cache schemas are rejected.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.preprocessing.video import VideoToWiLoR
from airpiano.preprocessing.wilor.cache import atomic_json, completed_output_is_valid


def discover_inputs(root):
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".mp4"
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


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
        jobs = [
            (source, output_root / source.relative_to(input_root).with_suffix(""))
            for source in sources
        ]
        report["discovered"] = len(jobs)
        destinations = Counter(destination for _, destination in jobs)
        atomic_json(output_root / "report.json", report)
        processor = None
        for source, destination in jobs:
            current = (source, destination)
            try:
                if destinations[destination] > 1:
                    raise ValueError(f"Multiple inputs map to {destination}")
                if completed_output_is_valid(destination):
                    report["skipped"] += 1
                else:
                    if processor is None:
                        processor = VideoToWiLoR(device=args.device)
                    print(f"[process] {source.relative_to(input_root)}", flush=True)
                    processor.run(source, destination)
                    report["processed"] += 1
            except Exception as error:
                record_failure(report, source, destination, error)
                print(f"[failed] {source}: {error}", file=sys.stderr, flush=True)
            current = None
            atomic_json(output_root / "report.json", report)
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
            atomic_json(output_root / "report.json", report)
        except OSError as error:
            print(f"[report failed] {error}", file=sys.stderr, flush=True)
        print_progress(report)


if __name__ == "__main__":
    raise SystemExit(main())
