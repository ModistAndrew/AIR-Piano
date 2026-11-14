#!/usr/bin/env python3
"""Headless webcam timing probe with a synchronized browser clock target.

Run in v2n, forward port 8766 over SSH, and open http://localhost:8766 on
the PC whose screen the webcam can see. See docs/camera_latency.md.
No inference, audio, MIDI, or preprocessing caches are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Bound a stalled V4L2 read. OpenCV reads this setting when opening the backend.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "3")

import cv2
import numpy as np


FRAME_FIELDS = (
    "frame_id", "arrival_s", "arrival_monotonic_s", "read_call_ms",
    "driver_monotonic_s", "driver_to_read_ms", "sample_path",
)


def statistics(values):
    if not values:
        return None
    return {
        "count": len(values), "average_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)), "min_ms": float(np.min(values)),
    }


def driver_delay_ms(driver_s, arrival_s, clock_name):
    """Only subtract timestamps when the driver's monotonic clock is known."""
    if (clock_name != "CLOCK_MONOTONIC" or not math.isfinite(driver_s)
            or driver_s <= 0 or driver_s > arrival_s):
        return None
    return (arrival_s - driver_s) * 1000


class Session:
    def __init__(self, args, output_dir):
        self.args = args
        self.output_dir = output_dir
        self.epoch = time.monotonic()
        self.lock = threading.Lock()
        self.start = threading.Event()
        self.latest_sample = None
        self.sync_samples = []
        self.status = {
            "state": "waiting", "captured_frames": 0, "elapsed_s": 0,
            "duration_s": args.duration, "output_dir": str(output_dir),
        }

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.status)


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, payload, status=200, content_type="application/json"):
            body = json.dumps(payload).encode() if content_type == "application/json" else payload
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A browser refresh can close an in-flight time request.

        def do_GET(self):
            received_s = time.monotonic() - session.epoch
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.respond(Path(__file__).with_name("camera_latency_clock.html").read_bytes(),
                             content_type="text/html; charset=utf-8")
            elif path == "/time":
                self.respond({"received_s": received_s, "sent_s": time.monotonic() - session.epoch})
            elif path == "/status":
                self.respond(session.snapshot())
            elif path == "/sample/latest.jpg":
                with session.lock:
                    sample = session.latest_sample
                if sample is None:
                    self.respond({"error": "No sample saved yet"}, 404)
                else:
                    self.respond(sample.read_bytes(), content_type="image/jpeg")
            else:
                self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path == "/start":
                with session.lock:
                    if session.status["state"] != "waiting":
                        result, status = {"error": "Capture already started"}, 409
                    elif not session.sync_samples:
                        result, status = {"error": "Synchronize the clock first"}, 409
                    else:
                        session.status["state"] = "warming_up"
                        session.start.set()
                        result, status = {"started": True}, 200
                self.respond(result, status)
            elif self.path == "/sync":
                try:
                    sample = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    sample = {key: float(sample[key]) for key in (
                        "offset_s", "round_trip_s", "uncertainty_s")}
                    if not all(math.isfinite(value) for value in sample.values()):
                        raise ValueError("Non-finite synchronization sample")
                    if sample["round_trip_s"] < 0 or sample["uncertainty_s"] < 0:
                        raise ValueError("Negative synchronization uncertainty")
                except (ValueError, TypeError, KeyError):
                    self.respond({"error": "Invalid synchronization sample"}, 400)
                    return
                sample["recorded_s"] = time.monotonic() - session.epoch
                with session.lock:
                    session.sync_samples.append(sample)
                    with (session.output_dir / "clock_sync.jsonl").open("a") as handle:
                        handle.write(json.dumps(sample) + "\n")
                self.respond({"saved": True})
            else:
                self.respond({"error": "Not found"}, 404)

    return Handler


class SampleWriter:
    """Encode occasional evidence images off the capture thread, with bounded work."""

    def __init__(self, session):
        self.session = session
        self.pending = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.error = None
        self.saved = 0
        self.skipped = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def offer(self, frame, row):
        name = f"samples/frame_{row['frame_id']:06d}.jpg"
        try:
            self.pending.put_nowait((frame, dict(row), name))
        except queue.Full:
            self.skipped += 1
            return ""
        return name

    def run(self):
        try:
            while not self.stop.is_set() or not self.pending.empty():
                try:
                    frame, row, name = self.pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                # Padding preserves every original pixel, including the filmed clock.
                image = cv2.copyMakeBorder(frame, 100, 0, 0, 0, cv2.BORDER_CONSTANT)
                scale = min(1.0, frame.shape[1] / 1200)
                lines = (
                    f"READ ARRIVAL = {row['arrival_s']:.3f} s   Frame {row['frame_id']}",
                    "Subtract the filmed clock below. Includes reference screen delay.",
                )
                for index, line in enumerate(lines):
                    cv2.putText(image, line, (12, 35 + index * 40), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 255, 255), 2, cv2.LINE_AA)
                path = self.session.output_dir / name
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError(f"Failed to write {path}")
                with self.session.lock:
                    self.session.latest_sample = path
                self.saved += 1
        except (OSError, cv2.error) as error:
            self.error = error

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Image writer did not stop within five seconds")
        if self.error is not None:
            raise RuntimeError("Image writer failed") from self.error


def capture(session):
    args = session.args
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    writer = None
    arrivals, driver_delays, read_times = [], [], []
    metadata = {}
    started = None
    failure = None
    interrupted = False
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{args.camera}; check permissions and other camera users")
        settings = (
            ("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, args.width),
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, args.height),
            ("fps", cv2.CAP_PROP_FPS, args.fps),
            ("buffers", cv2.CAP_PROP_BUFFERSIZE, 2),
        )
        metadata = {name: {"requested": value, "set_ok": bool(camera.set(prop, value))}
                    for name, prop, value in settings}
        for name, prop, _ in settings:
            metadata[name]["actual"] = camera.get(prop)
        metadata["backend"] = camera.getBackendName()
        # This is the known Linux UVC driver setting, not a claim of exposure timing.
        for name in ("clock", "hwtimestamps"):
            try:
                metadata[f"uvc_{name}"] = Path(f"/sys/module/uvcvideo/parameters/{name}").read_text().strip()
            except FileNotFoundError:
                metadata[f"uvc_{name}"] = None
        driver_link = Path(f"/sys/class/video4linux/video{args.camera}/device/driver")
        metadata["driver"] = driver_link.resolve().name if driver_link.exists() else None
        clock_name = metadata.get("uvc_clock") if metadata["driver"] == "uvcvideo" else None
        session.update(state="warming_up")
        print(f"[capture] Camera opened. Warming up for {args.warmup:g} s.", flush=True)
        warmup_end = time.monotonic() + args.warmup
        while time.monotonic() < warmup_end:
            ok, _frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed during warmup")
        writer = SampleWriter(session)
        started = time.monotonic()
        next_sample = next_log = started
        session.update(state="capturing")
        print(f"[capture] Measuring for {args.duration:g} s; keep the clock visible.", flush=True)
        with (session.output_dir / "frames.csv").open("w", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            csv_writer.writeheader()
            while time.monotonic() - started < args.duration:
                before_read = time.monotonic()
                ok, frame = camera.read()
                arrival = time.monotonic()  # Before property lookup, annotation, or I/O.
                if not ok:
                    raise RuntimeError("Camera read failed during measurement")
                driver_s = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000
                delay = driver_delay_ms(driver_s, arrival, clock_name)
                row = {
                    "frame_id": len(arrivals), "arrival_s": arrival - session.epoch,
                    "arrival_monotonic_s": arrival, "read_call_ms": (arrival - before_read) * 1000,
                    "driver_monotonic_s": driver_s if math.isfinite(driver_s) and driver_s > 0 else "",
                    "driver_to_read_ms": delay if delay is not None else "", "sample_path": "",
                }
                arrivals.append(arrival)
                read_times.append(row["read_call_ms"])
                if delay is not None:
                    driver_delays.append(delay)
                if arrival >= next_sample:
                    row["sample_path"] = writer.offer(frame, row)
                    next_sample = arrival + args.sample_interval
                csv_writer.writerow(row)
                elapsed = arrival - started
                session.update(captured_frames=len(arrivals), elapsed_s=elapsed)
                if writer.error is not None:
                    raise RuntimeError("Image writer failed") from writer.error
                if arrival >= next_log:
                    print(f"[capture] {elapsed:.1f}s | {len(arrivals)} frames | "
                          f"{len(arrivals) / elapsed:.2f} fps", flush=True)
                    next_log = arrival + 5
                    handle.flush()
    except KeyboardInterrupt:
        interrupted = True
    except (OSError, RuntimeError, cv2.error) as error:
        failure = error
    finally:
        capture_finished = time.monotonic()
        camera.release()
        if writer is not None:
            try:
                writer.close()
            except RuntimeError as error:
                failure = failure or error
        state = "failed" if failure else "interrupted" if interrupted else "complete"
        elapsed = capture_finished - started if started is not None else 0
        summary = {
            "state": state, "camera": metadata, "captured_frames": len(arrivals),
            "elapsed_s": elapsed, "captured_fps": len(arrivals) / elapsed if elapsed else 0,
            "arrival_interval_ms": statistics((np.diff(arrivals) * 1000).tolist()),
            "driver_to_read_ms": statistics(driver_delays),
            "driver_delay_unavailable_frames": len(arrivals) - len(driver_delays),
            "read_call_ms": statistics(read_times),
            "samples_saved": writer.saved if writer else 0,
            "samples_skipped_writer_busy": writer.skipped if writer else 0,
            "error": str(failure) if failure else None,
            "clock_sync_file": "clock_sync.jsonl",
            "optical_latency_ms": None,
            "measurement_notes": [
                "Driver-to-read is not sensor exposure-to-read latency.",
                "Read-call duration includes waiting for the next frame; it is not frame age.",
                "Optical estimate: image arrival overlay minus filmed synchronized clock.",
                "Optical estimate includes reference display latency and clock synchronization uncertainty.",
                "No OCR or automatic full camera-latency estimate is performed.",
            ],
        }
        (session.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        session.update(state=state, elapsed_s=elapsed, error=summary["error"])
        print(json.dumps(summary, indent=2), flush=True)
    return 1 if failure else 130 if interrupted else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--warmup", type=float, default=5, help="Seconds of discarded warmup reads")
    parser.add_argument("--sample-interval", type=float, default=1, help="Seconds between saved evidence images")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capture-now", action="store_true", help="Capture immediately without browser synchronization, then exit")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps, args.duration, args.sample_interval) <= 0 or args.warmup < 0:
        parser.error("Dimensions, fps, duration and sample interval must be positive; warmup must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or Path("outputs/camera_latency") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "samples").mkdir()
    session = Session(args, output_dir)
    config = {**vars(args), "output_dir": str(output_dir), "server_epoch_monotonic_s": session.epoch}
    (output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    print(f"[capture] Results: {output_dir}", flush=True)
    if args.capture_now:
        print("[capture] No synchronized browser target: driver timing and evidence images only.", flush=True)
        return capture(session)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[capture] Open http://localhost:{args.port} on your PC through an SSH port forward.", flush=True)
    print("[capture] Point the webcam at the page, then click Start capture. Ctrl+C exits.", flush=True)
    result = 0
    try:
        session.start.wait()
        result = capture(session)
        print("[capture] Capture stopped; page remains available for reviewing the last sample. Ctrl+C exits.", flush=True)
        if result == 130:
            return result
        threading.Event().wait()
    except KeyboardInterrupt:
        return result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
