#!/usr/bin/env python3
"""Run complete tabletop videos with and without the streaming deadline.

The reference uses the same live clock and fast WiLoR, but disables dropping and
backpressures video capture. Physical release accuracy is recomputed from the
aligned MIDI; skipped frames never shrink the target-chord denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import cv2
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airpiano import data
from airpiano.evaluation import aggregate_alignment_metrics, calculate_alignment_metrics
from airpiano.render.artifact import InferenceArtifact
from airpiano.utils import resolve_path, write_json

TOLERANCES = (4, 8, 16)


def source_hashes(config_name: str) -> dict[str, str]:
    config_file = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
    paths = {
        "airpiano/streaming/__init__.py", "airpiano/streaming/runtime.py",
        "airpiano/streaming/core.py", "airpiano/streaming/io.py", "airpiano/streaming/srt.py", "airpiano/streaming/recording.py", "airpiano/streaming/midi_output.py",
        "airpiano/full_song_real_time.py",
        "configs/default.yaml", f"configs/{config_file}",
        "run.py", "scripts/validate_streaming.py",
    }
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(paths)
    }


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open validation video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isclose(fps, 30.0, atol=0.01) or frame_count <= 0:
            raise ValueError("Validation requires a complete native 30 fps video")
        return {
            "video_path": str(video_path), "input_fps": fps,
            "source_frame_count": frame_count,
            "source_duration_seconds": frame_count / fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def release_metrics(
    chords: Sequence[data.Chord],
    releases: Sequence[dict[str, Any]],
    *,
    frame_count: int,
    fps: float = 30.0,
    target_radius_seconds: float = 0.2,
    record_id: str = "streaming_validation",
) -> dict[str, Any]:
    """Evaluate actual physical releases using the production C-AR definition.

    Controller IDs are one-based score occurrence IDs, not MIDI Chord.id values.
    Training's target delay is deliberately absent: ground truth stays physical.
    """
    if frame_count <= 0 or fps <= 0:
        raise ValueError("Validation requires positive frame count and fps")
    target_generator = data.TargetGenerator(
        data.ChordOnsetEvents(delay_seconds=0.0),
        data.TriangularSoftener(radius_seconds=target_radius_seconds),
    )
    physical_target = target_generator.generate(
        tuple(chords), frame_start=0, frame_count=frame_count, fps=fps,
    ).numpy()
    release_times = np.full(len(chords), np.nan, dtype=np.float64)
    for event in releases:
        if event.get("kind") != "release":
            continue
        chord_id = int(event["chord_id"])
        if not 1 <= chord_id <= len(chords):
            raise ValueError(f"Release references unknown chord occurrence {chord_id}")
        if np.isfinite(release_times[chord_id - 1]):
            raise ValueError(f"Duplicate release for chord occurrence {chord_id}")
        release_time = float(event["release_time"])
        if not np.isfinite(release_time) or release_time < 0:
            raise ValueError(f"Invalid physical release time: {release_time}")
        release_times[chord_id - 1] = release_time
    chord_pitches = np.zeros((len(chords), 128), dtype=np.bool_)
    for slot, chord in enumerate(chords):
        for note in chord.onset_notes:
            chord_pitches[slot, note.pitch] = True
    artifact = InferenceArtifact(
        probability=np.zeros_like(physical_target), target=physical_target,
        physical_target=physical_target,
        frame_mask=np.ones(frame_count, dtype=np.bool_),
        chord_mask=np.ones(len(chords), dtype=np.bool_),
        chord_pitches=chord_pitches,
        chord_ids=np.asarray([chord.id for chord in chords], dtype=np.int64),
        fps=fps, record_id=record_id, frame_start=0, release_time=release_times,
    )
    return calculate_alignment_metrics(artifact, tolerances_frames=TOLERANCES)


def validate_complete(summary: dict[str, Any], expected_frames: int) -> None:
    if summary.get("status") != "completed":
        raise ValueError(f"Incomplete streaming session: {summary.get('status')!r}")
    captured = int(summary["source_frame_count"])
    if captured != expected_frames:
        raise ValueError(
            f"Incomplete source: captured {captured} of {expected_frames} frames"
        )
    processed = int(summary["processed_frames"])
    dropped = int(summary["dropped_frames"])
    if processed + dropped != captured:
        raise ValueError("Processed plus dropped frames must equal captured frames")


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "average": None, "p95": None, "max": None}
    return {
        "count": len(array), "average": float(array.mean()),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def evaluate_session(
    session_dir: Path, record_id: str, metadata: dict[str, Any], cfg: DictConfig,
) -> dict[str, Any]:
    summary = json.loads((session_dir / "summary.json").read_text())
    validate_complete(summary, metadata["source_frame_count"])
    with (session_dir / "events.jsonl").open() as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    midi_path = resolve_path(cfg.data.midi_root) / f"{record_id}.mid"
    beat_path = resolve_path(cfg.data.beat_root) / f"{record_id}.txt"
    if not beat_path.is_file():
        beat_path = session_dir / "score_beats.txt"
    sequence = instantiate(cfg.dataset.midi_loader).load(midi_path, beat_path)
    metrics = release_metrics(
        sequence.chords, events,
        frame_count=metadata["source_frame_count"], fps=metadata["input_fps"],
        target_radius_seconds=float(cfg.dataset.target_generator.softener.radius_seconds),
        record_id=record_id,
    )
    with (session_dir / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    processed = [row for row in rows if row["status"] == "processed"]
    if len(processed) != int(summary["processed_frames"]):
        raise ValueError("Frame log and session summary disagree on processed count")
    latency = {
        key: _stats([float(row[key]) for row in processed])
        for key in ("wilor_canonical_ms", "airpiano_ms", "overhead_ms", "latency_ms")
    }
    source_manifest = (session_dir.parent.parent / "logs" /
                       f"{session_dir.parent.name}__{session_dir.name}.sources.json")
    source_sha256 = (
        json.loads(source_manifest.read_text())["source_sha256"]
        if source_manifest.is_file() else None
    )
    return {
        "record_id": record_id, "session_dir": str(session_dir),
        "source_sha256": source_sha256,
        "input": metadata, "summary": summary, "latency_ms": latency,
        "physical_release_metrics": metrics,
    }


def _override(key: str, value: Any) -> str:
    # Hydra parses the scalar; subprocess receives literal argv (never a shell).
    return f"{key}={json.dumps(value, separators=(',', ':'))}"


def run_session(
    cfg: DictConfig, config_name: str, overrides: Sequence[str], record_id: str,
    mode: str, session_dir: Path, log_path: Path,
) -> None:
    if session_dir.exists():
        raise FileExistsError(f"Refusing to overwrite streaming session: {session_dir}")
    midi_path = resolve_path(cfg.data.midi_root) / f"{record_id}.mid"
    beat_path = resolve_path(cfg.data.beat_root) / f"{record_id}.txt"
    settings = {
        "stage": "streaming", "streaming.source.kind": "video",
        "streaming.source.video_path": str(resolve_path(cfg.data.video_root) / f"{record_id}.mp4"),
        "streaming.score_midi": str(midi_path),
        "streaming.beat_path": str(beat_path) if beat_path.is_file() else None,
        "streaming.preview": False, "streaming.output_dir": str(session_dir),
        "streaming.midi_output.enabled": False,
        "streaming.wilor_devices": ["cuda:1", "cuda:2"],
        "streaming.airpiano_device": "cuda:1", "streaming.fast": True,
        "streaming.deadline_ms": 100 if mode == "bounded" else None,
        "streaming.drop_stale": mode == "bounded", "streaming.max_inflight": 4,
        "hydra.run.dir": str(session_dir.parent / f"{session_dir.name}_hydra"),
    }
    command = [sys.executable, str(ROOT / "run.py"), "--config-name", config_name]
    command.extend(overrides)
    command.extend(_override(key, value) for key, value in settings.items())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(log_path.with_suffix(".sources.json"), {
        "source_sha256": source_hashes(config_name), "command": command,
    })
    print(f"[validate] {mode}: {record_id}; log={log_path}", flush=True)
    with log_path.open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--split-file", type=Path, default=Path("splits/air_tabletop.txt"))
    parser.add_argument("--records", nargs="+", help="Default: first two split recordings")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override")
    parser.add_argument("--evaluate-only", action="store_true", help="Read existing complete sessions")
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    records = args.records or [
        line.strip() for line in resolve_path(args.split_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ][:2]
    if not records or len(set(records)) != len(records):
        parser.error("Choose at least one unique recording")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        record: video_metadata(resolve_path(cfg.data.video_root) / f"{record}.mp4")
        for record in records
    }
    manifest = {
        "config_name": args.config_name, "overrides": args.override,
        "records": metadata, "tolerances_frames": TOLERANCES,
        "ground_truth": "aligned MIDI physical onset peaks on all original video frames",
        "source_sha256": source_hashes(args.config_name),
    }
    manifest_path = output_dir / "inputs.json"
    if args.evaluate_only:
        original = json.loads(manifest_path.read_text())
        for key in ("config_name", "overrides", "records"):
            if original[key] != manifest[key]:
                raise ValueError(f"Re-evaluation {key} differs from original validation manifest")
        manifest = original
    else:
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite validation manifest: {manifest_path}")
        write_json(manifest_path, manifest)
    results: dict[str, list[dict[str, Any]]] = {"bounded": [], "uncapped": []}
    for mode in results:
        for record in records:
            slug = record.replace("/", "__")
            session_dir = output_dir / mode / slug
            if not args.evaluate_only:
                run_session(cfg, args.config_name, args.override, record, mode,
                            session_dir, output_dir / "logs" / f"{mode}__{slug}.log")
            result = evaluate_session(session_dir, record, metadata[record], cfg)
            if mode == "uncapped" and result["summary"]["dropped_frames"]:
                raise ValueError("Uncapped reference unexpectedly dropped frames")
            results[mode].append(result)
            write_json(output_dir / "partial_results.json", results)
    aggregate = {}
    for mode, runs in results.items():
        micro, _ = aggregate_alignment_metrics(
            [run["physical_release_metrics"] for run in runs], TOLERANCES,
        )
        elapsed = sum(float(run["summary"]["elapsed_seconds"]) for run in runs)
        source_seconds = sum(run["input"]["source_duration_seconds"] for run in runs)
        processed = sum(int(run["summary"]["processed_frames"]) for run in runs)
        captured = sum(int(run["summary"]["source_frame_count"]) for run in runs)
        pooled_samples = {key: [] for key in (
            "wilor_canonical_ms", "airpiano_ms", "overhead_ms", "latency_ms"
        )}
        for run in runs:
            with (Path(run["session_dir"]) / "frames.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["status"] == "processed":
                        for key, samples in pooled_samples.items():
                            samples.append(float(row[key]))
        aggregate[mode] = {
            "recording_count": len(runs), "source_frames": captured,
            "processed_frames": processed, "dropped_frames": captured - processed,
            "elapsed_seconds": elapsed, "source_duration_seconds": source_seconds,
            "throughput_captured_fps": captured / elapsed,
            "throughput_processed_fps": processed / elapsed,
            "processed_over_source_fps": processed / source_seconds,
            "over_100ms_frames": sum(value > 100 for value in pooled_samples["latency_ms"]),
            "latency_ms": {key: _stats(values) for key, values in pooled_samples.items()},
            "physical_release_metrics": micro,
        }
    bounded, reference = (aggregate[mode]["physical_release_metrics"] for mode in results)
    if bounded["ground_truth_chord_count"] != reference["ground_truth_chord_count"]:
        raise ValueError("Bounded and uncapped comparisons have different target denominators")
    differences = {
        str(tolerance): (
            100 * (bounded["c_ar"][str(tolerance)] - reference["c_ar"][str(tolerance)])
            if bounded["c_ar"][str(tolerance)] is not None else None
        ) for tolerance in TOLERANCES
    }
    report = {
        "runs": results, "aggregate": aggregate,
        "evaluation_source_sha256": source_hashes(args.config_name),
        "all_sessions_match_initial_source": all(
            run["source_sha256"] == manifest["source_sha256"]
            for runs in results.values() for run in runs
        ),
        "c_ar_bounded_minus_uncapped_percentage_points": differences,
        "latency_target_ms": 100, "processed_throughput_target_fps": 30,
        "notes": [
            "Actual release_time is evaluated against physical aligned MIDI onsets.",
            "C-AR is micro-averaged: summed correct chords / summed target chords.",
            "Uncapped reference uses the same source clock, disables drops, and may accumulate delay.",
            "Finite-session elapsed throughput includes final inference and scheduled-event drain.",
            "Dropped frames reduce processed throughput and never reduce target denominators.",
        ],
    }
    write_json(output_dir / "comparison.json", report)
    for mode in results:
        values = aggregate[mode]
        print(f"{mode}: {values['processed_frames']}/{values['source_frames']} frames, "
              f"{values['throughput_processed_fps']:.3f} processed fps; "
              f"C-AR@8={values['physical_release_metrics']['c_ar']['8']:.4%}")
    print(f"Report: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
