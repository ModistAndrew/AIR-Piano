"""Live 30 Hz capture, bounded WiLoR inference, and wall-clock chord releases."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from .. import data
from ..inference import load_model_checkpoint
from ..render.sonify import _seconds_to_tick, _write_note_events
from ..song_input import _midi_beats
from .core import AirPianoStepper, CanonicalFrameAdapter
from ..utils import resolve_path


FRAME_FIELDS = (
    "frame_index", "status", "reason", "source_time", "scheduled_at", "acquired_at",
    "dispatched_at", "wilor_started_at", "wilor_finished_at", "completed_at",
    "wilor_device", "latency_ms", "capture_to_completion_ms", "wilor_ms",
    "canonical_ms", "wilor_canonical_ms", "airpiano_ms", "overhead_ms",
    "history_frames", "hands_detected", "source_pts_seconds", "decoded_at",
    "arrival_pts_drift_ms",
)


def statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "average": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "average": float(array.mean()),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


class SessionLog:
    """Write diagnostics as they happen; keep frame/image queues out of logs."""

    def __init__(self, directory: Path, *, log_midi_events: bool = True) -> None:
        self.directory = directory
        self.log_midi_events = log_midi_events
        self.frames = (directory / "frames.csv").open("x", newline="")
        self.events = (directory / "events.jsonl").open("x")
        self.writer = csv.DictWriter(self.frames, fieldnames=FRAME_FIELDS)
        self.writer.writeheader()
        self.lock = threading.Lock()

    def frame(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.frames.flush()

    def event(self, kind: str, record: dict[str, Any]) -> None:
        with self.lock:
            self.events.write(json.dumps({"kind": kind, **record}) + "\n")
            self.events.flush()
            if self.log_midi_events and kind in {"trigger", "release"}:
                message = (f"[midi {kind}] chord={record['chord_id']} "
                           f"pitches={list(record['pitches'])}")
                if kind == "trigger":
                    message += (f" onset={record['time']:.3f}s"
                                f" confidence={record['score']:.3f}")
                else:
                    message += (f" onset={record['predicted_onset']:.3f}s"
                                f" released={record['release_time']:.3f}s"
                                f" duration={record['duration']:.3f}s")
                print(message, flush=True)

    def close(self) -> None:
        try:
            self.frames.close()
        finally:
            self.events.close()


class ReleaseScheduler:
    """Emit due events even when capture or model inference is stalled."""

    def __init__(self, stepper: AirPianoStepper, epoch: float,
                 emit: Callable[[dict[str, Any]], None],
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.stepper = stepper
        self.epoch = epoch
        self.emit = emit
        self.clock = clock
        self.wake = threading.Event()
        self.stopped = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="chord-releases", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def release_once(self) -> list[dict[str, Any]]:
        controller = self.stepper.controller
        with controller.lock:
            releases = controller.release_due(
                self.stepper.last_frame_index,
                now=max(0.0, self.clock() - self.epoch),
                frame_time=self.stepper.last_source_time,
            )
        for release in releases:
            self.emit(release)
        return releases

    def _run(self) -> None:
        try:
            while not self.stopped.is_set():
                self.wake.clear()
                self.release_once()
                with self.stepper.controller.lock:
                    pending = list(self.stepper.controller.pending_releases.values())
                wait = max(0.0, min(pending) - (self.clock() - self.epoch)) if pending else 0.05
                self.wake.wait(min(wait, 0.05))
        except BaseException as error:
            self.error = error
            self.stopped.set()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("Chord release scheduler failed") from self.error

    def close(self, *, drain: bool = False) -> None:
        if drain:
            while not self.stopped.is_set():
                self.check()
                with self.stepper.controller.lock:
                    pending = bool(self.stepper.controller.pending_releases)
                if not pending:
                    break
                self.wake.set()
                self.stopped.wait(0.005)
        self.stopped.set()
        self.wake.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
        self.check()


def prepare_start(capture: Any, pool: Any, countdown_seconds: int) -> float:
    """Drain live sources before a shared start epoch, checking worker health."""
    print("[streaming] models ready", flush=True)
    # Camera reads before this epoch drain old driver frames. Video retains frame
    # zero; SRT keeps decoding and discards pre-epoch frames. Keep the original
    # brief camera drain interval when the user disables the visible countdown.
    epoch = time.perf_counter() + max(0.2, countdown_seconds)
    capture.begin(epoch)
    last_remaining = None
    while True:
        capture.healthcheck()
        pool.healthcheck()
        remaining = epoch - time.perf_counter()
        if remaining <= 0:
            return epoch
        if countdown_seconds:
            display = math.ceil(remaining)
            if display != last_remaining:
                print(f"[streaming] starting in {display}...", flush=True)
                last_remaining = display
        time.sleep(min(0.05, remaining))


def validate_config(cfg: DictConfig) -> None:
    settings = cfg.streaming
    countdown = float(settings.get("countdown_seconds", 5))
    if not math.isfinite(countdown) or countdown < 0 or not countdown.is_integer():
        raise ValueError("streaming.countdown_seconds must be a nonnegative integer")
    midi_output = settings.get("midi_output", {})
    if midi_output.get("enabled", False):
        try:
            url = urlsplit(str(midi_output.get("url", "")))
            valid = url.scheme in {"ws", "wss"} and bool(url.hostname) and not url.fragment
            valid = valid and (url.port is None or 1 <= url.port <= 65535)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("streaming.midi_output.url must be a valid ws:// or wss:// URL")
        capacity = float(midi_output.get("capacity", 32))
        if not math.isfinite(capacity) or not capacity.is_integer() or not 1 <= capacity <= 1024:
            raise ValueError("streaming.midi_output.capacity must be an integer from 1 to 1024")
        for name, default in (("max_age_ms", 100), ("open_timeout_seconds", 1),
                              ("close_timeout_seconds", 0.5), ("reconnect_seconds", 1)):
            value = float(midi_output.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"streaming.midi_output.{name} must be finite and positive")
    recording = settings.get("recording", {})
    if recording.get("enabled", True):
        for name, default, minimum, maximum in (
            ("max_dimension", 640, 16, 4096), ("fps", 30, 1, 30), ("crf", 30, 0, 51),
        ):
            value = float(recording.get(name, default))
            if not math.isfinite(value) or not value.is_integer() or not minimum <= value <= maximum:
                raise ValueError(f"streaming.recording.{name} must be an integer from {minimum} to {maximum}")
        if int(recording.get("max_dimension", 640)) % 2:
            raise ValueError("streaming.recording.max_dimension must be even")
    if not math.isclose(float(cfg.dataset.fps), 30.0, abs_tol=1e-6):
        raise ValueError("Streaming v1 requires dataset.fps=30")
    if settings.source.kind not in ("camera", "video", "srt"):
        raise ValueError("streaming.source.kind must be camera, video, or srt")
    if settings.source.kind == "srt":
        try:
            url = urlsplit(str(settings.source.srt_url))
            valid = url.scheme == "srt" and bool(url.hostname) and url.port is not None and 1 <= url.port <= 65535
            modes = parse_qs(url.query).get("mode", ["caller"])
            valid = valid and len(modes) == 1 and modes[0] in {"caller", "listener", "rendezvous"}
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("streaming.source.srt_url must be srt://host:port with a valid SRT mode")
        for name in ("srt_open_timeout_seconds", "srt_read_timeout_seconds"):
            value = float(settings.source[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"streaming.source.{name} must be finite and positive")
    if settings.source.kind == "video" and not settings.source.video_path:
        raise ValueError("streaming.source.video_path is required for video input")
    if not settings.score_midi:
        raise ValueError("streaming.score_midi must identify the score to follow")
    if not settings.wilor_devices or len(set(settings.wilor_devices)) != len(settings.wilor_devices):
        raise ValueError("Specify one or more distinct streaming.wilor_devices")
    if int(settings.max_inflight) < len(settings.wilor_devices):
        raise ValueError("streaming.max_inflight must accommodate every WiLoR worker")
    if settings.deadline_ms is not None and (
        not math.isfinite(float(settings.deadline_ms)) or float(settings.deadline_ms) <= 0
    ):
        raise ValueError("streaming.deadline_ms must be positive or null")
    if not settings.drop_stale and settings.deadline_ms is not None:
        raise ValueError("No-drop diagnostic replay requires streaming.deadline_ms=null")
    if not settings.drop_stale and settings.source.kind != "video":
        raise ValueError("No-drop diagnostic replay is supported for videos only")
    if int(settings.warmup_frames) < 1 or int(settings.warmup_airpiano_frames) < 1:
        raise ValueError("Streaming warmup counts must be positive")
    if settings.preview and os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("No display is available; use streaming.preview=false for headless execution")


def check_checkpoint_timing(cfg: DictConfig, checkpoint: Path) -> dict[str, Any]:
    """Never silently apply a shifted checkpoint using an unshifted controller."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = OmegaConf.create(payload.get("config", {}))
    checked, unavailable, assumed = {}, [], {}
    paths = (
        "dataset.fps",
        "dataset.target_generator.softener.radius_seconds",
        "dataset.target_generator.event_generator.delay_seconds",
    )
    for path in paths:
        value = OmegaConf.select(saved, path)
        runtime = OmegaConf.select(cfg, path)
        if value is None:
            unavailable.append(path)
            if path.endswith("event_generator.delay_seconds"):
                # Checkpoints predating delayed targets used unshifted events.
                value = 0.0
                assumed[path] = value
            else:
                continue
        if not math.isclose(float(value), float(runtime), abs_tol=1e-8):
            raise ValueError(f"Checkpoint timing mismatch for {path}: saved={value}, configured={runtime}; use its training config")
        else:
            checked[path] = float(value)
    return {"verified": checked, "unavailable": unavailable, "legacy_assumptions": assumed}


def write_midi(releases: list[dict[str, Any]], path: Path) -> None:
    events = []
    for release in releases:
        start = _seconds_to_tick(release["release_time"], ticks_per_beat=480, tempo=500_000)
        end = _seconds_to_tick(release["release_time"] + release["duration"], ticks_per_beat=480, tempo=500_000)
        for pitch in release["pitches"]:
            events.extend(((start, 1, int(pitch), 64), (end, 0, int(pitch), 0)))
    _write_note_events(events, path)


def summarize(directory: Path, *, source_frame_count: int, elapsed_seconds: float,
              captured_frames: int | None = None, **metadata: Any) -> dict[str, Any]:
    with (directory / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    processed = [row for row in rows if row["status"] == "processed"]
    dropped = [row for row in rows if row["status"] == "dropped"]
    stages = {key: statistics([float(row[key]) for row in processed]) for key in (
        "wilor_canonical_ms", "airpiano_ms", "overhead_ms", "latency_ms", "capture_to_completion_ms",
    )}
    reasons: dict[str, int] = {}
    for row in dropped:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    source_duration = source_frame_count / 30.0
    completed = [float(row["completed_at"]) for row in processed]
    cadence = ((len(completed) - 1) / (completed[-1] - completed[0])
               if len(completed) > 1 and completed[-1] > completed[0] else None)
    srt_drift = [float(row["arrival_pts_drift_ms"]) for row in processed
                 if row.get("arrival_pts_drift_ms") not in (None, "")]
    timing_scope = (
        "Monotonic source availability to completed AirPiano; video includes decode/IPC/queueing. "
        "Camera begins at read completion and excludes exposure/driver buffering. "
        "SRT begins when a decoded frame is available, before BGR conversion; it excludes phone "
        "capture/encoding, network transport, SRT buffering, and decoding. SRT source_time follows "
        "PTS anchored to the first post-warmup decoded frame; arrival_pts_drift_ms is relative "
        "to that anchor, not absolute network latency. Startup excluded. Event releases use "
        "actual wall clock; MIDI export is not physical audio playback."
    )
    return {
        **metadata,
        "source_frame_count": source_frame_count,
        "captured_frames": source_frame_count if captured_frames is None else captured_frames,
        "processed_frames": len(processed), "dropped_frames": len(dropped),
        "drop_reasons": reasons,
        "drop_fraction": len(dropped) / source_frame_count if source_frame_count else None,
        "elapsed_seconds": elapsed_seconds, "source_duration_seconds": source_duration,
        "input_fps": 30.0,
        "throughput_captured_fps": (captured_frames if captured_frames is not None else source_frame_count) / elapsed_seconds if elapsed_seconds else None,
        "throughput_processed_fps": len(processed) / elapsed_seconds if elapsed_seconds else None,
        "processed_over_source_fps": len(processed) / source_duration if source_duration else None,
        "output_cadence_fps": cadence,
        "over_100ms_frames": sum(float(row["latency_ms"]) > 100 for row in processed),
        "deadline_dropped_frames": reasons.get("inference_deadline", 0) + reasons.get("capture_deadline", 0),
        "deadline_misses": (
            sum(float(row["latency_ms"]) > float(metadata["deadline_ms"]) for row in processed)
            + reasons.get("inference_deadline", 0) + reasons.get("capture_deadline", 0)
        ) if metadata.get("deadline_ms") is not None else None,
        "stages_ms": stages,
        "target_p95_below_100ms": bool(processed) and stages["latency_ms"]["p95"] < 100,
        "all_source_frames_processed": bool(source_frame_count) and len(processed) == source_frame_count,
        "srt_arrival_pts_drift_ms": statistics(srt_drift),
        "timing_scope": timing_scope,
    }


def _show_preview(image: np.ndarray, *, processed: int, dropped: int,
                  elapsed: float, latency: float | None, chord: int) -> bool:
    height, width = image.shape[:2]
    display = cv2.resize(image, (min(width, 960), max(1, int(height * min(width, 960) / width))))
    lines = [f"Chord: {chord} | processed {processed} | dropped {dropped}",
             f"Inference: {processed / max(elapsed, 1e-6):.1f} fps | delay: {latency:.1f} ms" if latency is not None else "Waiting for inference",
             "Q / Esc: finish session"]
    for index, line in enumerate(lines):
        cv2.putText(display, line, (12, 28 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(display, line, (12, 28 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.imshow("AirPiano live", display)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27) and cv2.getWindowProperty("AirPiano live", cv2.WND_PROP_VISIBLE) >= 1


def streaming(cfg: DictConfig) -> dict[str, Any]:
    """Run one score-following session without reading/writing hand caches."""
    from .io import CaptureSource, WiLoRWorkerPool
    from .recording import VideoRecorder
    from .midi_output import WebSocketMidiOutput

    validate_config(cfg)
    settings = cfg.streaming
    checkpoint = resolve_path(settings.checkpoint)
    score_path = resolve_path(settings.score_midi)
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    verification = check_checkpoint_timing(cfg, checkpoint)
    output = (resolve_path(settings.output_dir) if settings.output_dir else
              resolve_path(cfg.output.dir) / "streaming" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    log = SessionLog(output, log_midi_events=bool(settings.get("log_midi_events", True)))
    capture = pool = scheduler = stepper = recorder = midi_output = None
    midi_settings = settings.get("midi_output", {})
    midi_report = {"status": "not_started" if midi_settings.get("enabled", False) else "disabled"}
    midi_console_status = None
    recording_warning = False
    recording_settings = settings.get("recording", {})
    recording_report = {"status": "disabled" if not recording_settings.get("enabled", True) else "not_started"}
    epoch = None
    metadata: dict[str, Any] = {"checkpoint": str(checkpoint), "checkpoint_timing": verification,
                                "score_midi": str(score_path), "wilor_devices": list(settings.wilor_devices),
                                "airpiano_device": str(settings.airpiano_device), "fast": bool(settings.fast),
                                "deadline_ms": settings.deadline_ms, "drop_stale": bool(settings.drop_stale),
                                "source_kind": str(settings.source.kind),
                                "countdown_seconds": int(settings.get("countdown_seconds", 5)),
                                "log_midi_events": bool(settings.get("log_midi_events", True))}
    status, failure = "starting", None
    cleanup_errors: list[str] = []
    seen = -1
    processed = dropped = 0
    captured_frames = None
    source_frames = 0
    waiting: dict[str, Any] | None = None
    inflight: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    retired: set[int] = set()
    eof = False
    peak_inflight = 0
    last_latency = None
    current_chord = 1
    latest_image = None
    deadline_seconds = None if settings.deadline_ms is None else float(settings.deadline_ms) / 1000.0

    def drop(frame: dict[str, Any], reason: str) -> None:
        nonlocal dropped
        dropped += 1
        log.frame({"frame_index": frame["sequence"], "status": "dropped", "reason": reason,
                   "source_time": frame.get("source_time", frame["sequence"] / 30.0),
                   "scheduled_at": frame["scheduled_at"] - epoch if epoch is not None and "scheduled_at" in frame else "",
                   "acquired_at": frame["acquired_at"] - epoch if epoch is not None and "acquired_at" in frame else "",
                   "source_pts_seconds": frame.get("source_pts_seconds"),
                   "decoded_at": frame["decoded_at"] - epoch if epoch is not None and frame.get("decoded_at") is not None else "",
                   "arrival_pts_drift_ms": frame.get("arrival_pts_drift_ms")})

    def midi_transport_event(record: dict[str, Any]) -> None:
        nonlocal midi_console_status
        stamp = record.get("timestamp")
        session_time = float(stamp) - epoch if stamp is not None and epoch is not None else None
        log.event("midi_output", {**record, "session_time": session_time})
        state = record.get("status")
        if state in {"connected", "disconnected", "send_failed", "disabled", "error"}:
            if state != midi_console_status:
                detail = record.get("error") or str(midi_settings.get("url", ""))
                print(f"[midi websocket] {state}: {detail}", flush=True)
            midi_console_status = state

    def release_event(record: dict[str, Any]) -> None:
        if midi_output is not None:
            try:
                # Queue before file/console I/O and age notes from actual release.
                midi_output.submit(record, released_at=epoch + float(record["release_time"]))
            except Exception as error:
                midi_transport_event({"status": "error", "error": str(error),
                                      "timestamp": time.perf_counter()})
        log.event("release", record)

    def check_recording() -> None:
        nonlocal recording_warning
        if recorder is not None:
            report = recorder.healthcheck()
            if report.get("error") and not recording_warning:
                print(f"[streaming] video recording stopped: {report['error']}; inference continues", flush=True)
                recording_warning = True

    def expired(frame: dict[str, Any], now: float) -> bool:
        return bool(settings.drop_stale) and deadline_seconds is not None and now - frame["scheduled_at"] >= deadline_seconds

    try:
        if settings.torch_threads is not None:
            torch.set_num_threads(int(settings.torch_threads))
        torch.manual_seed(int(cfg.seed))
        device = torch.device(str(settings.airpiano_device))
        if device.type == "cuda":
            torch.cuda.set_device(device)
        components = data.build_data_components(cfg)
        beat_path = resolve_path(settings.beat_path) if settings.beat_path else output / "score_beats.txt"
        if not settings.beat_path:
            beat_path.write_text("".join(f"{value:.9f} {number}\n" for value, number in _midi_beats(score_path)))
        score = components["midi_loader"].load(score_path, beat_path)
        metadata["beat_path"] = str(beat_path)
        model = load_model_checkpoint(checkpoint, device).eval()
        capture_options = {}
        capture_timeout = 30.0
        if settings.source.kind == "video":
            source = str(resolve_path(settings.source.video_path))
        elif settings.source.kind == "srt":
            source = str(settings.source.srt_url)
            capture_options["srt_options"] = {
                "open_timeout_seconds": float(settings.source.srt_open_timeout_seconds),
                "read_timeout_seconds": float(settings.source.srt_read_timeout_seconds),
            }
            capture_timeout = sum(capture_options["srt_options"].values()) + 10.0
            print("[streaming] opening SRT receiver; start the phone stream, then wait for model warmup", flush=True)
        else:
            source = int(settings.source.camera_index)
        if recording_settings.get("enabled", True):
            try:
                recorder = VideoRecorder(
                    output / "performance.mp4",
                    max_dimension=int(recording_settings.get("max_dimension", 640)),
                    fps=int(recording_settings.get("fps", 30)),
                    crf=int(recording_settings.get("crf", 30)),
                )
                recorder.start()
                capture_options["recording_sink"] = recorder.sink
                recording_report = {"status": "starting", "path": str(output / "performance.mp4")}
            except Exception as error:
                recording_report = {"status": "error", "error": str(error)}
                print(f"[streaming] video recording unavailable: {error}; inference continues", flush=True)
                recording_warning = True
        capture = CaptureSource(source, kind=str(settings.source.kind), fps=30.0,
                                width=int(settings.source.width), height=int(settings.source.height),
                                capacity=1, drop_stale=bool(settings.drop_stale), **capture_options)
        capture.start()
        source_metadata = capture.wait_ready(timeout=capture_timeout)
        example = source_metadata.pop("example_image", None)
        metadata["source"] = source_metadata
        frame_shape = tuple(source_metadata.get("frame_shape", (settings.source.height, settings.source.width, 3)))
        backend = OmegaConf.to_container(cfg.preprocess.video_to_ragged, resolve=True)
        for key in ("_target_", "device", "fast", "frame_batch_size"):
            backend.pop(key, None)
        pool = WiLoRWorkerPool(list(settings.wilor_devices), fast=bool(settings.fast),
                              warmup=int(settings.warmup_frames), warmup_hand_counts=tuple(settings.warmup_hand_counts),
                              capacity=int(settings.max_inflight), frame_shape=frame_shape,
                              torch_threads=settings.torch_threads, backend_kwargs=backend)
        if midi_settings.get("enabled", False):
            try:
                midi_output = WebSocketMidiOutput(
                    str(midi_settings["url"]), capacity=int(midi_settings.get("capacity", 32)),
                    max_age_ms=float(midi_settings.get("max_age_ms", 100)),
                    open_timeout_seconds=float(midi_settings.get("open_timeout_seconds", 1)),
                    close_timeout_seconds=float(midi_settings.get("close_timeout_seconds", 0.5)),
                    reconnect_seconds=float(midi_settings.get("reconnect_seconds", 1)),
                    emit=midi_transport_event,
                )
                midi_output.start()
            except Exception as error:
                midi_report = {"status": "error", "error": str(error)}
                print(f"[midi websocket] unavailable: {error}; inference continues", flush=True)
        pool.start(example_image=example)
        print("[streaming] warming WiLoR and AirPiano; session clock has not started", flush=True)
        warmup = pool.wait_ready(timeout=float(settings.startup_timeout_seconds))
        metadata["workers"] = warmup["workers"]
        canonical = CanonicalFrameAdapter()
        warm_stepper = AirPianoStepper(model, score.chords, components, device, fps=30.0, bf16=bool(cfg.train.bf16))
        for index in range(int(settings.warmup_airpiano_frames)):
            pose, presence = canonical.step(warmup["arrays"])
            presence_tensor = torch.from_numpy(presence)
            hands = components["hand_loader"].transform(torch.from_numpy(pose), presence_tensor)
            warm_stepper.step(hands, presence_tensor, release=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del warm_stepper
        stepper = AirPianoStepper(model, score.chords, components, device, fps=30.0, bf16=bool(cfg.train.bf16))
        canonical = CanonicalFrameAdapter()
        metadata.update({"torch_version": torch.__version__, "opencv_version": cv2.__version__,
                         "torch_threads": torch.get_num_threads(), "airpiano_bf16": bool(cfg.train.bf16),
                         "onset_prediction_offset": stepper.controller.onset_prediction_offset})
        (output / "session.json").write_text(json.dumps(metadata, indent=2) + "\n")
        epoch = prepare_start(capture, pool, int(settings.get("countdown_seconds", 5)))
        scheduler = ReleaseScheduler(stepper, epoch, release_event)
        scheduler.start()
        status = "running"
        last_progress = epoch
        last_preview = 0.0
        event_count = 0
        print(f"[streaming] ready: {source} -> {output}", flush=True)
        print("[streaming] GO! Start playing.", flush=True)
        while not eof or waiting is not None or inflight:
            pool.healthcheck()
            capture.healthcheck()
            scheduler.check()
            check_recording()
            now = time.perf_counter()
            # In diagnostic mode, downstream backpressure reaches the decoder.
            if not eof and (settings.drop_stale or waiting is None):
                message = capture.poll(timeout=0.0)
                if message is not None:
                    if message["kind"] == "eof":
                        eof = True
                        source_frames = int(message["last_sequence"]) + 1
                        captured_frames = int(message["frames"])
                        for missing in range(seen + 1, source_frames):
                            drop({"sequence": missing}, "capture_overwrite")
                        seen = max(seen, source_frames - 1)
                    elif message["kind"] == "frame":
                        for missing in range(seen + 1, int(message["sequence"])):
                            drop({"sequence": missing}, "source_gap_or_capture_overwrite"
                                 if settings.source.kind == "srt" else "capture_overwrite")
                        seen = int(message["sequence"])
                        source_frames = max(source_frames, seen + 1)
                        latest_image = message["image"]
                        if settings.source.kind in ("camera", "srt"):
                            captured_frames = int(message.get("captured_frames", seen + 1))
                            prefix = str(settings.source.kind)
                            metadata[f"{prefix}_read_count"] = int(message.get("read_count", captured_frames))
                            metadata[f"{prefix}_duplicate_reads_dropped"] = int(message.get("duplicate_reads_dropped", 0))
                        if waiting is not None:
                            drop(waiting, "replaced_waiting_capture")
                        waiting = message
                    else:
                        raise RuntimeError(f"Unexpected capture message: {message['kind']}")
            while True:
                result = pool.poll(timeout=0.0)
                if result is None:
                    break
                index = int(result["sequence"])
                if index in retired:
                    retired.remove(index)
                    inflight.pop(index, None)
                elif index in inflight:
                    results[index] = result
                else:
                    raise RuntimeError(f"Unexpected WiLoR result for frame {index}")
            # Retired work retains its credit until its worker actually finishes.
            while True:
                active = sorted(index for index in inflight if index not in retired)
                if not active:
                    break
                index = active[0]
                frame = inflight[index]
                if expired(frame, time.perf_counter()):
                    drop(frame, "inference_deadline")
                    if index in results:
                        results.pop(index)
                        inflight.pop(index)
                    else:
                        retired.add(index)
                    continue
                if index not in results:
                    break
                result = results.pop(index)
                canonical_start = time.perf_counter()
                pose, presence = canonical.step(result["arrays"], frame_index=index)
                presence_tensor = torch.from_numpy(presence)
                hands = components["hand_loader"].transform(torch.from_numpy(pose), presence_tensor)
                canonical_done = time.perf_counter()
                step = stepper.step(hands, presence_tensor, frame_index=index,
                                    source_time=float(frame["source_time"]), release=False)
                completed = time.perf_counter()
                scheduler.wake.set()
                with stepper.controller.lock:
                    new_events = stepper.controller.event_records[event_count:]
                    event_count += len(new_events)
                for event in new_events:
                    log.event("trigger", {
                        **event, "pitches": stepper.controller.chords[event["chord_id"]].pitches,
                    })
                    if event["success"]:
                        current_chord = event["chord_id"]
                wilor_ms = (result["finished_at"] - result["started_at"]) * 1000
                canonical_ms = (canonical_done - canonical_start) * 1000
                airpiano_ms = (completed - canonical_done) * 1000
                last_latency = (completed - frame["scheduled_at"]) * 1000
                row = {
                    "frame_index": index, "status": "processed", "reason": "",
                    "source_time": frame["source_time"], "scheduled_at": frame["scheduled_at"] - epoch,
                    "acquired_at": frame["acquired_at"] - epoch, "dispatched_at": frame["dispatched_at"] - epoch,
                    "wilor_started_at": result["started_at"] - epoch, "wilor_finished_at": result["finished_at"] - epoch,
                    "completed_at": completed - epoch, "wilor_device": result["device"],
                    "latency_ms": last_latency, "capture_to_completion_ms": (completed - frame["acquired_at"]) * 1000,
                    "wilor_ms": wilor_ms, "canonical_ms": canonical_ms,
                    "wilor_canonical_ms": wilor_ms + canonical_ms, "airpiano_ms": airpiano_ms,
                    "overhead_ms": last_latency - wilor_ms - canonical_ms - airpiano_ms,
                    "history_frames": step["history_frames"],
                    "hands_detected": int(result["arrays"]["frame_detection_offsets"][-1]),
                    "source_pts_seconds": frame.get("source_pts_seconds"),
                    "decoded_at": frame["decoded_at"] - epoch if frame.get("decoded_at") is not None else None,
                    "arrival_pts_drift_ms": frame.get("arrival_pts_drift_ms"),
                }
                log.frame(row)
                processed += 1
                inflight.pop(index)
            if waiting is not None:
                if expired(waiting, time.perf_counter()):
                    drop(waiting, "capture_deadline")
                    waiting = None
                elif len(inflight) < int(settings.max_inflight):
                    dispatched = time.perf_counter()
                    if pool.submit(int(waiting["sequence"]), waiting["image"], float(waiting["source_time"])):
                        # Retain metadata only after copying BGR into worker storage.
                        frame = {key: value for key, value in waiting.items() if key != "image"}
                        frame["dispatched_at"] = dispatched
                        inflight[int(frame["sequence"])] = frame
                        peak_inflight = max(peak_inflight, len(inflight))
                        waiting = None
            now = time.perf_counter()
            if settings.preview and latest_image is not None and now - last_preview >= 1 / 30:
                if not _show_preview(latest_image, processed=processed, dropped=dropped,
                                     elapsed=max(0.0, now - epoch), latency=last_latency,
                                     chord=current_chord):
                    status = "interrupted"
                    break
                last_preview = now
            if now - last_progress >= 5:
                delay = f"{last_latency:.1f} ms" if last_latency is not None else "pending"
                print(f"[streaming] {processed} processed, {dropped} dropped, {processed / (now - epoch):.2f} fps, delay {delay}", flush=True)
                last_progress = now
            time.sleep(0.001)
        if status == "running":
            status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException as error:
        status, failure = "error", error
    finally:
        def cleanup_error(error: BaseException) -> None:
            nonlocal status, failure
            status, failure = "error", failure or error
            cleanup_errors.append(f"{type(error).__name__}: {error}")

        if scheduler is not None:
            try:
                scheduler.close(drain=status == "completed")
            except BaseException as error:
                cleanup_error(error)
        ended = time.perf_counter()
        if recorder is not None:
            recorder.stop_accepting()
        # Mark accepted unfinished work exactly once on interruption/failure.
        unfinished = ([waiting] if waiting is not None else []) + [
            frame for index, frame in inflight.items() if index not in retired
        ]
        for frame in unfinished:
            try:
                drop(frame, "session_stopped")
            except BaseException as error:
                cleanup_error(error)
        if midi_output is not None:
            try:
                midi_report = midi_output.close(drain=status == "completed", timeout=2.0)
            except Exception as error:
                midi_report = {"status": "error", "error": str(error)}
                print(f"[midi websocket] shutdown error: {error}", flush=True)
        metadata["midi_output"] = midi_report
        try:
            (output / "midi_output.json").write_text(json.dumps(midi_report, indent=2) + "\n")
        except Exception as error:
            print(f"[streaming] could not save MIDI transport report: {error}", flush=True)
        for resource in (capture, pool):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    cleanup_error(error)
        if settings.preview:
            try:
                cv2.destroyAllWindows()
            except BaseException as error:
                cleanup_error(error)
        releases = stepper.controller.release_records if stepper is not None else []
        elapsed = max(0.0, ended - epoch) if epoch is not None else 0.0
        if recorder is not None:
            video_end = max(elapsed, max(
                (float(record["release_time"]) + float(record["duration"]) for record in releases),
                default=0.0,
            ))
            try:
                recording_report = recorder.finish(video_end)
            except Exception as error:
                recording_report = {"status": "error", "error": str(error)}
            if recording_report.get("error") and not recording_warning:
                print(f"[streaming] video recording error: {recording_report['error']}", flush=True)
            elif recording_report.get("status") == "completed":
                print(f"[streaming] saved video: {output / 'performance.mp4'}", flush=True)
        metadata["recording"] = recording_report
        try:
            (output / "recording.json").write_text(json.dumps(recording_report, indent=2) + "\n")
        except Exception as error:
            print(f"[streaming] could not save recording report: {error}", flush=True)
        try:
            write_midi(releases, output / "performance.mid")
        except BaseException as error:
            cleanup_error(error)
        try:
            log.close()
        except BaseException as error:
            cleanup_error(error)
        summary = {**metadata, "source_frame_count": source_frames,
                   "processed_frames": processed, "dropped_frames": dropped,
                   "elapsed_seconds": elapsed, "peak_inflight": peak_inflight,
                   "released_chords": len(releases)}
        try:
            summary = summarize(output, source_frame_count=source_frames,
                                captured_frames=captured_frames,
                                elapsed_seconds=elapsed, peak_inflight=peak_inflight,
                                released_chords=len(releases), **metadata)
        except BaseException as error:
            cleanup_error(error)
        summary.update(status=status, error=str(failure) if failure else None,
                       cleanup_errors=cleanup_errors)
        try:
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        except BaseException as error:
            cleanup_error(error)
    print(f"[streaming] {status}: {summary['processed_frames']} processed, {summary['dropped_frames']} dropped; {output / 'summary.json'}", flush=True)
    if failure is not None:
        raise failure
    return summary
