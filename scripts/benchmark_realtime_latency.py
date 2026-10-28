#!/usr/bin/env python3
"""Measure the current full_song_real_time loop on complete cached songs.

Timing statements are inserted into an in-memory copy of the production
function, so its frame processing stays identical and runtime files are untouched.
The existing CPU score readback waits for CUDA before each frame timer ends.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from airpiano import data
import airpiano.full_song_real_time as rt
from airpiano.inference import load_model_checkpoint
from airpiano.utils import resolve_path


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": len(values), "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem)
    checkpoint = resolve_path(args.checkpoint or cfg.full_song_real_time.checkpoint)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    components = data.build_data_components(cfg)
    torch.manual_seed(int(cfg.seed))
    model = load_model_checkpoint(checkpoint, device)
    source = inspect.getsource(rt.infer_full_song_real_time)
    loop = "    for frame_index in range(frame_count):\n"
    end = "        controller.release_due(frame_index)\n"
    assert source.count(loop) == source.count(end) == 1
    instrumented = source.replace(loop, loop + "        _started = _clock()\n")
    instrumented = instrumented.replace(end, end + (
        "        _elapsed_ms = (_clock() - _started) * 1000\n"
        "        _rows.append((frame_index, _elapsed_ms, frame_index + 1 - history_start))\n"
        "        if (frame_index + 1) % 500 == 0:\n"
        "            print(f'  frame {frame_index + 1}/{frame_count}, last {_elapsed_ms:.2f} ms', flush=True)\n"
    ))
    namespace = dict(vars(rt), _clock=time.perf_counter, _rows=[])
    exec(compile(instrumented, str(Path(rt.__file__)), "exec"), namespace)
    infer = namespace["infer_full_song_real_time"]
    report = {
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(),
        "checkpoint": str(checkpoint), "bf16": bool(cfg.train.bf16),
        "warmup_frames": args.warmup,
        "runtime_source_sha256": hashlib.sha256(Path(rt.__file__).read_bytes()).hexdigest(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timing_scope": "Frame input assembly through CPU score readback, controller DP and release_due; excludes loading, warmup, final artifact/MIDI export.",
        "songs": [],
    }
    all_rows = []
    for song_index, record in enumerate(args.records):
        pair = data.FilePair(
            record_id=record,
            midi_path=resolve_path(cfg.data.midi_root) / f"{record}.mid",
            hand_path=resolve_path(cfg.data.hand_root) / f"{record}.npz",
            beat_path=resolve_path(cfg.data.beat_root) / f"{record}.txt",
        )
        song = components["file_loader"].reader.read(
            pair, components["midi_loader"], components["hand_loader"]
        )
        kwargs = dict(tokenizer=components["tokenizer"], fps=float(cfg.dataset.fps),
                      bf16=bool(cfg.train.bf16))
        if song_index == 0:
            warmup = replace(song, hands=song.hands[:args.warmup],
                             presence=song.presence[:args.warmup])
            infer(model, warmup, components["slicer"], components["target_generator"],
                  device, output_dir / "warmup.npz", **kwargs)
        namespace["_rows"] = []
        torch.cuda.synchronize(device)
        print(f"Benchmark {record}: {len(song.hands)} frames, {len(song.chords)} chords", flush=True)
        started = time.perf_counter()
        controller = infer(
            model, song, components["slicer"], components["target_generator"],
            device, output_dir / "artifacts" / f"{record}.npz", **kwargs,
        )
        torch.cuda.synchronize(device)
        total = time.perf_counter() - started
        rows = namespace["_rows"]
        summary = {
            "record_id": record, "frames": len(rows), "chords": len(song.chords),
            "frame_ms": statistics([row[1] for row in rows]),
            "history_frames": statistics([row[2] for row in rows]),
            "loop_seconds": sum(row[1] for row in rows) / 1000,
            "function_seconds_including_export": total,
            "events": len(controller.event_records),
        }
        report["songs"].append(summary)
        all_rows.extend((record, *row) for row in rows)
        report["all_frame_ms"] = statistics([row[2] for row in all_rows])
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output_dir / "frames.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "frame_index", "frame_ms", "history_frames"])
            writer.writerows(all_rows)
        print(json.dumps(summary), flush=True)
    print(json.dumps(report["all_frame_ms"]), flush=True)


if __name__ == "__main__":
    main()
