#!/usr/bin/env python3
"""Measure warmed WiLoR latency on preloaded native-resolution video frames.

Example: conda run -n v2n python scripts/benchmark_wilor_latency.py VIDEO.mp4 \
    --output-dir outputs/latency/wilor --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.preprocessing.video import VideoToWiLoR


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def preload_video(path, sample_count):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        indices = np.unique(
            np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int)
        )
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            frames.append(frame)
    finally:
        capture.release()
    return {
        "path": str(path),
        "source_frame_count": frame_count,
        "fps": fps,
        "source_frame_indices": indices.tolist(),
        "timestamps_seconds": (indices / fps).tolist(),
        "dimensions_wh": sorted({(frame.shape[1], frame.shape[0]) for frame in frames}),
        "frames": frames,
    }


class ComponentTimers:
    """Observe detector calls and pose forwards without synchronizing inside them."""

    def __init__(self, models):
        self.models = models
        self.records = []
        self.pose_start = None

    def start(self):
        event = None
        if self.models.device.type == "cuda":
            event = self.models.torch.cuda.Event(enable_timing=True)
            event.record(self.models.torch.cuda.current_stream(self.models.device))
        return time.perf_counter(), event

    def finish(self, component, start):
        wall_start, cuda_start = start
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cuda_end = None
        if cuda_start is not None:
            cuda_end = self.models.torch.cuda.Event(enable_timing=True)
            cuda_end.record(self.models.torch.cuda.current_stream(self.models.device))
        self.records.append((component, wall_ms, cuda_start, cuda_end))

    def pose_pre_hook(self, module, args):
        self.pose_start = self.start()

    def pose_post_hook(self, module, args, output):
        self.finish("pose", self.pose_start)

    def totals(self):
        result = {}
        for component in ("detector", "pose"):
            records = [item for item in self.records if item[0] == component]
            result[f"{component}_calls"] = len(records)
            result[f"{component}_call_wall_ms"] = sum(item[1] for item in records)
            result[f"{component}_cuda_ms"] = (
                sum(item[2].elapsed_time(item[3]) for item in records)
                if self.models.device.type == "cuda"
                else None
            )
        return result


class TimedDetector:
    def __init__(self, detector, timers):
        self.detector = detector
        self.timers = timers

    def __call__(self, *args, **kwargs):
        start = self.timers.start()
        result = self.detector(*args, **kwargs)
        self.timers.finish("detector", start)
        return result


def measure_batch(backend, video, positions, timers):
    models = backend.models
    indices = [video["source_frame_indices"][i] for i in positions]
    timestamps = [video["timestamps_seconds"][i] for i in positions]
    frames = [video["frames"][i] for i in positions]
    timers.records.clear()
    if models.device.type == "cuda":
        models.torch.cuda.synchronize(models.device)
    start = time.perf_counter()
    chunk = backend.build_chunk(indices, timestamps, frames)
    if models.device.type == "cuda":
        models.torch.cuda.synchronize(models.device)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if any(not np.isfinite(value).all() for value in chunk.values()):
        raise FloatingPointError("WiLoR produced nonfinite predictions")
    return {
        "batch_wall_ms": wall_ms,
        "amortized_wall_ms": wall_ms / len(frames),
        "hand_counts": np.diff(chunk["frame_detection_offsets"]).tolist(),
        **timers.totals(),
    }


def summarize_batches(batches):
    frame_count = sum(batch["batch_size"] for batch in batches)
    total_ms = sum(batch["batch_wall_ms"] for batch in batches)
    summary = {
        "sample_count": frame_count,
        "batch_count": len(batches),
        "total_wall_ms": total_ms,
        "mean_amortized_frame_ms": total_ms / frame_count,
        "throughput_fps": frame_count * 1000.0 / total_ms,
        "batch_completion_latency": statistics([batch["batch_wall_ms"] for batch in batches]),
        "mean_hands_per_frame": sum(sum(batch["hand_counts"]) for batch in batches) / frame_count,
        "zero_hand_frames": sum(count == 0 for batch in batches for count in batch["hand_counts"]),
    }
    for component in ("detector", "pose"):
        for timing in ("call_wall_ms", "cuda_ms"):
            key = f"{component}_{timing}"
            values = [batch[key] for batch in batches]
            summary[f"mean_{key}_per_frame"] = (
                sum(values) / frame_count if values[0] is not None else None
            )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--fast", action="store_true", help="Use official FP16, compilation, and backbone layer skipping")
    parser.add_argument("--samples", default=32, type=int, help="Evenly spread frames per video")
    parser.add_argument("--warmup", default=5, type=int, help="Excluded batches for each video and mode")
    parser.add_argument("--frame-batch-size", default=8, type=int)
    parser.add_argument("--hand-batch-size", default=16, type=int)
    args = parser.parse_args(argv)
    if min(args.samples, args.frame_batch_size, args.hand_batch_size) < 1 or args.warmup < 0:
        parser.error("Samples and batch sizes must be positive; warmup must be nonnegative")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT / "outputs"):
        parser.error("--output-dir must be inside the repository outputs/ directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "frames.csv"
    if summary_path.exists() or csv_path.exists():
        raise FileExistsError(f"Benchmark output already exists in {output_dir}")

    videos = [preload_video(path.resolve(), args.samples) for path in args.videos]
    processor = VideoToWiLoR(
        device=args.device,
        frame_batch_size=args.frame_batch_size,
        hand_batch_size=args.hand_batch_size,
        fast=args.fast,
    )
    backend = processor._backend
    loading_start = time.perf_counter()
    models = backend.models
    if models.device.type == "cuda":
        models.torch.cuda.synchronize(models.device)
    loading_seconds = time.perf_counter() - loading_start
    timers = ComponentTimers(models)
    detector = models.detector
    models.detector = TimedDetector(detector, timers)
    hooks = [
        models.model.register_forward_pre_hook(timers.pose_pre_hook),
        models.model.register_forward_hook(timers.pose_post_hook),
    ]
    summary = {
        "status": "running",
        "device": str(models.device),
        "device_name": models.torch.cuda.get_device_name(models.device) if models.device.type == "cuda" else "CPU",
        "torch_version": models.torch.__version__,
        "torch_num_threads": models.torch.get_num_threads(),
        "torch_num_interop_threads": models.torch.get_num_interop_threads(),
        "opencv_num_threads": cv2.getNumThreads(),
        "pose_parameter_dtype": str(next(models.model.parameters()).dtype),
        "fast": args.fast,
        "float32_matmul_precision": models.torch.get_float32_matmul_precision(),
        "predictions_checked_finite": True,
        "model_loading_seconds_excluded": loading_seconds,
        "requested_samples_per_video": args.samples,
        "warmup_batches_per_video_and_mode": args.warmup,
        "frame_batch_size": args.frame_batch_size,
        "hand_batch_size": args.hand_batch_size,
        "method": {
            "total": "CUDA-synchronized build_chunk wall time; preloaded native-resolution frames; no video decode, model loading, cache writing, or frame acquisition wait",
            "components": "Detector wrapper and pose forward hooks; wall intervals are unsynchronized; CUDA events measure elapsed stream intervals, including any CPU launch gaps; components exclude crop preparation, host transfers, and output assembly outside those calls",
            "batch_latency": "Time to complete an already available batch; every frame in that batch shares completion latency; amortized time is throughput cost, not live per-frame response latency",
            "sample_indices": "Evenly spaced across each complete source video",
        },
        "videos": [],
    }
    all_batches = {"single_frame": [], "batched": []}
    fields = [
        "video", "mode", "sample_position", "source_frame_index", "timestamp_seconds",
        "width", "height", "hand_count", "batch_id", "batch_size", "batch_wall_ms",
        "amortized_wall_ms", "detector_calls", "detector_call_wall_ms", "detector_cuda_ms",
        "pose_calls", "pose_call_wall_ms", "pose_cuda_ms",
    ]
    try:
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for video in videos:
                video_summary = {key: value for key, value in video.items() if key != "frames"}
                video_summary["modes"] = {}
                for mode, batch_size in (("single_frame", 1), ("batched", args.frame_batch_size)):
                    positions = list(range(len(video["frames"])))
                    groups = [positions[start:start + batch_size] for start in range(0, len(positions), batch_size)]
                    for index in range(args.warmup):
                        measure_batch(backend, video, groups[index % len(groups)], timers)
                    batches = []
                    for batch_id, group in enumerate(groups):
                        result = measure_batch(backend, video, group, timers)
                        result["batch_size"] = len(group)
                        batches.append(result)
                        for position, hand_count in zip(group, result["hand_counts"]):
                            frame = video["frames"][position]
                            writer.writerow({
                                "video": video["path"], "mode": mode, "sample_position": position,
                                "source_frame_index": video["source_frame_indices"][position],
                                "timestamp_seconds": video["timestamps_seconds"][position],
                                "width": frame.shape[1], "height": frame.shape[0], "hand_count": hand_count,
                                "batch_id": batch_id,
                                **{key: value for key, value in result.items() if key != "hand_counts"},
                            })
                        handle.flush()
                    all_batches[mode].extend(batches)
                    video_summary["modes"][mode] = summarize_batches(batches)
                    print(json.dumps({"video": video["path"], "mode": mode, **video_summary["modes"][mode]}), flush=True)
                summary["videos"].append(video_summary)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["aggregate"] = {mode: summarize_batches(batches) for mode, batches in all_batches.items()}
        summary["status"] = "complete"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        models.detector = detector
        for hook in hooks:
            hook.remove()
    print(f"Saved {summary_path} and {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
