#!/usr/bin/env python3
"""Replay native 30 Hz videos through parallel WiLoR and ordered AirPiano.

Example: conda run --no-capture-output -n v2n python \
    scripts/benchmark_two_gpu_streaming.py AIR/jinxuan_tabletop_aligned/beethoven_8_3 \
    --devices cuda:1 cuda:2 --output-dir outputs/two_gpu_streaming
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airpiano import data
from airpiano.inference import load_model_checkpoint
from airpiano.preprocessing.video import VideoToWiLoR
from airpiano.utils import resolve_path
from scripts.streaming_latency_components import AirPianoStepper, CanonicalFrameAdapter


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def wilor_worker(device, name, shape, jobs, results, available, warmup, threads,
                 fast=False, warmup_hand_counts=(1, 2, 3)):
    if threads is not None:
        torch.set_num_threads(threads)
    torch.cuda.set_device(torch.device(device))
    memory = shared_memory.SharedMemory(name=name)
    frames = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
    try:
        startup_started = time.perf_counter()
        backend = VideoToWiLoR(device=device, frame_batch_size=1, fast=fast)._backend
        models = backend.models
        compile_started = time.perf_counter()
        if fast:
            image_size = int(models.model_cfg.MODEL.IMAGE_SIZE)
            for count in warmup_hand_counts:
                crops = torch.zeros(
                    (count, 3, image_size, image_size), device=models.device,
                    dtype=next(models.model.parameters()).dtype,
                )
                with torch.inference_mode():
                    # Keep the real crop view strides, including its 32-pixel margins.
                    models.model.backbone(crops[:, :, :, 32:-32])
            torch.cuda.synchronize(models.device)
        compile_seconds = time.perf_counter() - compile_started if fast else 0.0
        for _ in range(warmup):
            example = backend.build_chunk([0], [0.0], [frames[0]])
        torch.cuda.synchronize(models.device)
        results.put(("ready", device, {
            "device_name": torch.cuda.get_device_name(models.device),
            "torch_threads": torch.get_num_threads(),
            "pose_dtype": str(next(models.model.parameters()).dtype),
            "fast": fast,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "warmup_hand_counts": list(warmup_hand_counts) if fast else [],
            "shape_compile_warmup_seconds_excluded": compile_seconds,
            "startup_seconds_excluded": time.perf_counter() - startup_started,
        }, example))
        while True:
            job = jobs.get()
            if job is None:
                break
            index, slot, timing = job
            started = time.perf_counter()
            arrays = backend.build_chunk([index], [index / 30.0], [frames[slot]])
            torch.cuda.synchronize(models.device)
            finished = time.perf_counter()
            available.put(slot)
            results.put(("frame", index, device, timing, started, finished, arrays))
    finally:
        memory.close()


def receive(results, workers, producer=None):
    while True:
        try:
            return results.get(timeout=0.5)
        except queue.Empty:
            for worker in workers:
                if worker.exitcode is not None and worker.exitcode != 0:
                    raise RuntimeError(f"WiLoR worker {worker.name} exited {worker.exitcode}")
            if producer is not None and producer.done():
                producer.result()


def produce(video, capture, first_frame, count, epoch, frames, jobs, available, stop, progress):
    try:
        for index in range(count):
            scheduled = epoch + index / 30.0
            if stop.wait(max(0.0, scheduled - time.perf_counter())):
                return
            decode_started = time.perf_counter()
            ok, frame = (True, first_frame) if index == 0 else capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode {video}, frame {index}")
            decoded = time.perf_counter()
            while not stop.is_set():
                try:
                    slot = available.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            else:
                return
            frames[slot] = frame
            dispatched = time.perf_counter()
            timing = (scheduled, decode_started, decoded, dispatched,
                      index + 1 - progress["completed"])
            jobs[index % len(jobs)].put((index, slot, timing))
    finally:
        capture.release()


def run_record(record, cfg, components, model, device, jobs, results, workers,
               available, frames, count_limit, output_dir):
    video = resolve_path(cfg.data.video_root) / f"{record}.mp4"
    decoder_started = time.perf_counter()
    probe = cv2.VideoCapture(str(video))
    if not probe.isOpened():
        raise FileNotFoundError(video)
    fps = probe.get(cv2.CAP_PROP_FPS)
    count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = (int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
    if abs(fps - 30.0) > 1e-3 or shape != frames.shape[1:]:
        probe.release()
        raise ValueError(f"Expected native 30 Hz frames of {frames.shape[1:]}, got {fps}, {shape}")
    ok, first_frame = probe.read()
    if not ok:
        probe.release()
        raise RuntimeError(f"Cannot prime decoder: {video}")
    decoder_startup = time.perf_counter() - decoder_started
    if count_limit:
        count = min(count, count_limit)
    midi = components["midi_loader"].load(
        resolve_path(cfg.data.midi_root) / f"{record}.mid",
        resolve_path(cfg.data.beat_root) / f"{record}.txt",
    )
    canonical = CanonicalFrameAdapter()
    stepper = AirPianoStepper(model, midi.chords, components, device,
                             fps=30.0, bf16=bool(cfg.train.bf16))
    pending = {}
    rows = []
    release_rows = []
    progress = {"completed": 0}
    stop = threading.Event()
    epoch = time.perf_counter() + 0.25
    print(f"REPLAY {record}: {count} frames at 30 Hz", flush=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        producer = pool.submit(produce, video, probe, first_frame, count, epoch, frames, jobs,
                               available, stop, progress)
        try:
            while len(rows) < count:
                message = receive(results, workers, producer)
                _, index, worker_device, timing, started, wilor_done, arrays = message
                pending[index] = (worker_device, timing, started, wilor_done, arrays)
                while len(rows) in pending:
                    index = len(rows)
                    worker_device, timing, started, wilor_done, arrays = pending.pop(index)
                    scheduled, decode_started, decoded, dispatched, inflight = timing
                    ordered = time.perf_counter()
                    pose, presence = canonical.step(arrays)
                    canonical_done = time.perf_counter()
                    step = stepper.step(pose, presence)
                    completed = time.perf_counter()
                    progress["completed"] = index + 1
                    row = {
                        "record_id": record, "frame_index": index, "wilor_device": worker_device,
                        "scheduled_seconds": scheduled - epoch,
                        "completed_seconds": completed - epoch,
                        "latency_ms": (completed - scheduled) * 1000,
                        "dispatch_to_done_ms": (completed - dispatched) * 1000,
                        "arrival_lateness_ms": (decode_started - scheduled) * 1000,
                        "decode_ms": (decoded - decode_started) * 1000,
                        "dispatch_ms": (dispatched - decoded) * 1000,
                        "worker_queue_ms": (started - dispatched) * 1000,
                        "wilor_ms": (wilor_done - started) * 1000,
                        "ordered_wait_ms": (ordered - wilor_done) * 1000,
                        "canonical_ms": (canonical_done - ordered) * 1000,
                        "airpiano_ms": (completed - canonical_done) * 1000,
                        "inflight_at_dispatch": inflight,
                        "history_frames": step["history_frames"],
                        "hands_detected": int(arrays["frame_detection_offsets"][-1]),
                    }
                    rows.append(row)
                    for release in stepper.controller.release_records[len(release_rows):]:
                        trigger_time = release["predicted_onset"] - stepper.controller.onset_prediction_offset
                        release_rows.append({
                            "record_id": record, "chord_id": release["chord_id"],
                            "frame_index": index,
                            "trigger_to_release_ready_ms": (completed - epoch - trigger_time) * 1000,
                            "predicted_onset_to_release_ready_ms": (completed - epoch - release["predicted_onset"]) * 1000,
                        })
                    if (index + 1) % 150 == 0:
                        print(f"  {index + 1}/{count}: delay {row['latency_ms']:.1f} ms, "
                              f"WiLoR {row['wilor_ms']:.1f} ms, AirPiano {row['airpiano_ms']:.1f} ms", flush=True)
            producer.result()
        finally:
            stop.set()
    latency = [r["latency_ms"] for r in rows]
    summary = {
        "record_id": record, "frames": count, "source_shape": shape, "input_fps": fps,
        "decoder_startup_seconds_excluded": decoder_startup,
        "latency_ms": stats(latency),
        "over_100ms_frames": sum(t > 100 for t in latency), "dropped_frames": 0,
        "output_cadence_fps": (count - 1) / (rows[-1]["completed_seconds"] - rows[0]["completed_seconds"]),
        "throughput_including_start_and_drain_fps": count / rows[-1]["completed_seconds"],
        "first_second_latency_ms": float(np.mean(latency[:30])),
        "last_second_latency_ms": float(np.mean(latency[-30:])),
        "latency_slope_ms_per_second": float(np.polyfit(np.arange(count) / 30.0, latency, 1)[0]),
        "max_inflight": max(r["inflight_at_dispatch"] for r in rows),
        "stages_ms": {key: stats([r[key] for r in rows]) for key in (
            "arrival_lateness_ms", "decode_ms", "dispatch_ms", "worker_queue_ms",
            "wilor_ms", "ordered_wait_ms", "canonical_ms", "airpiano_ms")},
        "controller_release_count": len(release_rows),
    }
    if release_rows:
        summary["controller_trigger_to_release_ready_ms"] = stats(
            [r["trigger_to_release_ready_ms"] for r in release_rows])
        summary["controller_predicted_onset_to_release_ready_ms"] = stats(
            [r["predicted_onset_to_release_ready_ms"] for r in release_rows])
    output = output_dir / record
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, records in ((".frames.csv", rows), (".releases.csv", release_rows)):
        if records:
            with Path(str(output) + suffix).open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    Path(str(output) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("record_id", "frames", "latency_ms", "output_cadence_fps", "over_100ms_frames")}), flush=True)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2"])
    parser.add_argument("--airpiano-device", default="cuda:1")
    parser.add_argument("--fast", action="store_true", help="Use official WiLoR fast mode in the spawned workers")
    parser.add_argument("--frames", type=int, default=0, help="0 replays the complete recording")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--warmup-hand-counts", nargs="+", type=int, default=[1, 2, 3],
                        help="Pose batch sizes compiled before replay in fast mode")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.slots < 1 or args.frames == 1 or args.frames < 0:
        parser.error("Warmup/slots must be positive and frames must be 0 or at least 2")
    if any(count < 1 or count > 16 for count in args.warmup_hand_counts):
        parser.error("Warmup hand counts must be between 1 and the hand batch size 16")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="default")
    components = data.build_data_components(cfg)
    device = torch.device(args.airpiano_device)
    torch.cuda.set_device(device)
    model = load_model_checkpoint(resolve_path(cfg.full_song_real_time.checkpoint), device)
    first_video = resolve_path(cfg.data.video_root) / f"{args.records[0]}.mp4"
    capture = cv2.VideoCapture(str(first_video))
    ok, example = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {first_video}")
    shape = (args.slots, *example.shape)
    memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
    frames = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
    frames[0] = example
    context = mp.get_context("spawn")
    available = context.Queue()
    results = context.Queue()
    jobs = [context.Queue() for _ in args.devices]
    workers = [context.Process(
        target=wilor_worker, name=f"wilor-{gpu}",
        args=(gpu, memory.name, shape, inbox, results, available, args.warmup,
              args.torch_threads, args.fast, args.warmup_hand_counts),
    ) for gpu, inbox in zip(args.devices, jobs)]
    try:
        for worker in workers:
            worker.start()
        metadata = {}
        for _ in workers:
            kind, gpu, info, arrays = receive(results, workers)
            assert kind == "ready"
            metadata[gpu] = info
        midi = components["midi_loader"].load(
            resolve_path(cfg.data.midi_root) / f"{args.records[0]}.mid",
            resolve_path(cfg.data.beat_root) / f"{args.records[0]}.txt",
        )
        warm = AirPianoStepper(model, midi.chords, components, device,
                              fps=30.0, bf16=bool(cfg.train.bf16))
        canonical = CanonicalFrameAdapter()
        for _ in range(30):
            warm.step(*canonical.step(arrays))
        torch.cuda.synchronize(device)
        for slot in range(args.slots):
            available.put(slot)
        report = {
            "status": "running", "wilor_devices": args.devices,
            "airpiano_device": str(device), "workers": metadata,
            "wilor_fast": args.fast,
            "airpiano_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_version": torch.__version__, "main_torch_threads": torch.get_num_threads(),
            "checkpoint": str(resolve_path(cfg.full_song_real_time.checkpoint)),
            "airpiano_bf16": bool(cfg.train.bf16), "frame_batch_size": 1,
            "warmup_wilor_frames_per_worker": args.warmup, "warmup_airpiano_frames": 30,
            "warmup_wilor_hand_counts_per_worker": args.warmup_hand_counts if args.fast else [],
            "method": "Real 30Hz paced native-resolution file decode, shared-memory transport, round-robin WiLoR workers, in-order results, causal canonical conversion and stateful AirPiano on actual inferred hands. No frames dropped. Delay starts at scheduled frame availability and includes decoding, queueing/IPC/reordering and all computation. Decoder is primed with frame0 before the clock starts; subsequent decoding is timed. Excludes model/decoder initialization, camera exposure/driver buffering and physical MIDI/audio playback. Controller release metrics retain production simulated scheduling.",
            "songs": [],
        }
        all_rows = []
        for record in args.records:
            summary, rows = run_record(record, cfg, components, model, device, jobs, results,
                                       workers, available, frames, args.frames, args.output_dir)
            report["songs"].append(summary)
            all_rows.extend(rows)
            report["aggregate_latency_ms"] = stats([r["latency_ms"] for r in all_rows])
            report["over_100ms_frames"] = sum(r["latency_ms"] > 100 for r in all_rows)
            (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "complete"
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        for inbox in jobs:
            inbox.put(None)
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
        memory.close()
        memory.unlink()


if __name__ == "__main__":
    main()
