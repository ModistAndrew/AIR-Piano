#!/usr/bin/env python3
"""Run fresh full-song inference for DP skip costs using cached inputs only.

Example:
    python scripts/evaluate_dp_skip_penalties.py --config configs/default.yaml \
        --checkpoint outputs/airpiano/checkpoints/last.pt \
        --split-file splits/air_tabletop.txt --output-dir outputs/tmp/dp_skip_trial

Each penalty has independent controller states and model forward passes. Existing
NPZ + MIDI + atomic metrics triples are resumed; incomplete runs are recomputed.
The application source and configuration files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano import data
import airpiano.full_song_real_time as rt
from airpiano.data.files import read_record_stems
from airpiano.evaluation import aggregate_alignment_metrics, calculate_alignment_metrics
from airpiano.inference import load_model_checkpoint
from airpiano.render.artifact import load_artifact
from airpiano.train import seed_everything
from airpiano.utils import resolve_path

TOLERANCES = (4, 8, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def penalty_controller(original: type, method: object, penalty: float) -> type:
    return type(
        "SkipPenaltyController", (original,),
        {"score_skip_penalty": penalty, "process_event": method},
    )


def baseline_parity(song, artifact_path, original, patched, cfg) -> dict:
    """Check original versus cost-one DP/window/history on saved events."""
    kwargs = {
        "window_size": int(cfg.dataset.slicer.segment_frames),
        "fps": float(cfg.dataset.fps),
        "target_radius_seconds": float(cfg.dataset.target_generator.softener.radius_seconds),
    }
    expected = original(song.chords, **kwargs)
    actual = patched(song.chords, **kwargs)
    with np.load(artifact_path, allow_pickle=False) as payload:
        for chord_id, event_time, score, frame in zip(
            payload["event_chord_id"], payload["event_time"],
            payload["event_score"], payload["event_frame_index"], strict=True,
        ):
            event = rt.Event(int(chord_id), float(event_time), float(score))
            assert expected.process_event(event, int(frame)) == actual.process_event(event, int(frame))
        np.testing.assert_array_equal(expected.prev, actual.prev)
        np.testing.assert_array_equal(expected.cur, actual.cur)
        assert expected.event_records == actual.event_records
        for key in ("success", "match_score", "best_score", "chord_window_start", "video_history_start"):
            np.testing.assert_array_equal(
                [row[key] for row in expected.event_records], payload[f"event_{key}"]
            )
    return {"passed": True, "event_count": len(expected.event_records),
            "scope": "original versus patched cost-one DP, windows, and history on saved events"}


def write_reports(output_dir: Path, penalties: list[float], records: list[str], runs: list[dict]) -> None:
    summaries = []
    for penalty in penalties:
        selected = [run for run in runs if run["skip_penalty"] == penalty]
        micro, macro = aggregate_alignment_metrics([run["metrics"] for run in selected], TOLERANCES)
        summaries.append({"skip_penalty": penalty, "skip_score": -penalty,
                          "completed_records": len(selected), "micro": micro, "macro": macro})
    atomic_json(output_dir / "summary.json", {
        "complete": len(runs) == len(records) * len(penalties),
        "requested_records": len(records), "tolerances_frames": list(TOLERANCES),
        "onset_source": "release_time", "variants": summaries,
    })
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append({
            "record_id": run["record_id"], "skip_penalty": run["skip_penalty"],
            "chord_acc": metrics["chord_acc"],
            **{f"c_ar_{value}": metrics["c_ar"][str(value)] for value in TOLERANCES},
            "chord_coverage": metrics["chord_coverage"],
            "onset_mae_frames": metrics["onset_mae_frames"],
            "onset_mae_seconds": metrics["onset_mae_seconds"],
            "event_count": run["event_count"],
            "accepted_event_count": run["accepted_event_count"],
            "released_chord_count": run["released_chord_count"],
            "elapsed_seconds": run["elapsed_seconds"],
            "artifact_path": run["artifact_path"],
        })
    atomic_json(output_dir / "comparison.json", {"records": records, "runs": rows})
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(output_dir / "comparison.csv", stream.getvalue())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--penalties", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    args = parser.parse_args(argv)
    if any(not math.isfinite(cost) or cost < 0 for cost in args.penalties):
        parser.error("--penalties must be finite nonnegative costs")
    if len(set(args.penalties)) != len(args.penalties):
        parser.error("--penalties must be distinct")
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config)
    checkpoint = resolve_path(args.checkpoint)
    split_path = resolve_path(args.split_file)
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem)
    cfg.device = args.device
    cfg.stage = "full_song_real_time"
    cfg.full_song_real_time.checkpoint = str(checkpoint)
    cfg.full_song_real_time.split_file = str(split_path)
    cfg.full_song_real_time.output_dir = str(output_dir)

    records = read_record_stems([split_path])
    if not records:
        raise ValueError(f"No records in {split_path}")
    pairs = []
    inputs = []
    missing = []
    for record in records:
        if Path(record).is_absolute() or ".." in Path(record).parts:
            raise ValueError(f"Record must be a relative stem: {record}")
        pair = data.FilePair(
            record_id=record,
            midi_path=resolve_path(cfg.data.midi_root) / f"{record}.mid",
            hand_path=resolve_path(cfg.data.hand_root) / f"{record}.npz",
            beat_path=resolve_path(cfg.data.beat_root) / f"{record}.txt",
        )
        pairs.append(pair)
        for path in (pair.midi_path, pair.hand_path, pair.beat_path):
            if not path.is_file():
                missing.append(str(path))
            else:
                stat = path.stat()
                inputs.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if missing:
        raise FileNotFoundError("Required cached inputs are missing:\n" + "\n".join(missing))

    original = rt.RealTimeController
    original_source = textwrap.dedent(inspect.getsource(original.process_event))
    old_term = "self.cur[index - 1] - 1"
    assert original_source.count(old_term) == 1, "Controller skip expression changed; review the patch"
    patched_source = original_source.replace(old_term, "self.cur[index - 1] - self.score_skip_penalty")
    namespace = dict(vars(rt))
    exec(compile(patched_source, str(output_dir / "process_event_penalty.txt"), "exec"), namespace)
    method = namespace["process_event"]
    identity = {
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "split_path": str(split_path), "split_sha256": sha256(split_path),
        "records": records, "cached_inputs": inputs, "penalties": args.penalties,
        "script_sha256": sha256(Path(__file__)),
        "inference_source_sha256": sha256(Path(rt.__file__)),
        "runtime_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in sorted((PROJECT_ROOT / "airpiano").rglob("*.py"))
        },
        "patched_method_sha256": hashlib.sha256(patched_source.encode()).hexdigest(),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "torch_num_threads": 1, "tolerances_frames": list(TOLERANCES),
    }
    manifest_path = output_dir / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Experiment identity differs; choose a new --output-dir")
    else:
        atomic_json(manifest_path, {"created_at": datetime.now(timezone.utc).isoformat(), "identity": identity})
    atomic_text(output_dir / "process_event_original.txt", original_source)
    atomic_text(output_dir / "process_event_penalty.txt", patched_source)
    atomic_text(output_dir / "experiment_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))

    torch.set_num_threads(1)
    seed_everything(int(cfg.seed))
    components = data.build_data_components(cfg)
    reader = components["file_loader"].reader
    device = torch.device(args.device)
    model = None
    runs = []
    for index, pair in enumerate(pairs, 1):
        song = None
        for penalty in args.penalties:
            label = format(penalty, "g").replace(".", "p")
            artifact_path = output_dir / f"penalty_{label}" / split_path.stem / f"{pair.record_id}.npz"
            midi_path = artifact_path.with_suffix(".mid")
            metrics_path = artifact_path.with_suffix(".metrics.json")
            if all(path.is_file() for path in (artifact_path, midi_path, metrics_path)):
                run = json.loads(metrics_path.read_text(encoding="utf-8"))
                assert run["record_id"] == pair.record_id and run["skip_penalty"] == penalty
                print(f"resume {index}/{len(pairs)} cost={penalty:g} {pair.record_id}", flush=True)
            else:
                metrics_path.unlink(missing_ok=True)
                if song is None:
                    song = reader.read(pair, components["midi_loader"], components["hand_loader"])
                if model is None:
                    model = load_model_checkpoint(checkpoint, device)
                print(f"infer {index}/{len(pairs)} cost={penalty:g} {pair.record_id} "
                      f"({len(song.hands)} frames, {len(song.chords)} chords)", flush=True)
                trial_controller = penalty_controller(original, method, penalty)
                started = time.perf_counter()
                rt.RealTimeController = trial_controller
                try:
                    controller = rt.infer_full_song_real_time(
                        model, song, components["slicer"], components["target_generator"],
                        device, artifact_path, tokenizer=components["tokenizer"],
                        fps=float(cfg.dataset.fps), bf16=bool(cfg.train.bf16),
                    )
                finally:
                    rt.RealTimeController = original
                elapsed = time.perf_counter() - started
                metrics = calculate_alignment_metrics(load_artifact(artifact_path), tolerances_frames=TOLERANCES)
                parity = baseline_parity(song, artifact_path, original, trial_controller, cfg) if penalty == 1 else None
                run = {
                    "record_id": pair.record_id, "skip_penalty": penalty,
                    "artifact_path": str(artifact_path.relative_to(output_dir)),
                    "metrics": metrics, "elapsed_seconds": elapsed,
                    "frame_count": len(song.hands), "chord_count": len(song.chords),
                    "event_count": len(controller.event_records),
                    "accepted_event_count": sum(row["success"] for row in controller.event_records),
                    "released_chord_count": len(controller.release_records),
                    "baseline_parity": parity,
                }
                atomic_json(metrics_path, run)
                print(f"done cost={penalty:g} {pair.record_id} seconds={elapsed:.2f} "
                      f"C-AR={metrics['c_ar']} coverage={metrics['chord_coverage']} "
                      f"MAE_seconds={metrics['onset_mae_seconds']}", flush=True)
                del controller
            runs.append(run)
            write_reports(output_dir, args.penalties, records, runs)
    print(f"Complete: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
