"""Optional PyAV SRT receiver with a presentation-time model timeline.

SRT runs continuously through model warmup. Only decoded BGR images in the
existing shared-memory mailbox reach inference; audio is never decoded. Sender
PTS preserve temporal positions but cannot measure phone-to-host latency without
clock synchronization. Processing deadlines start at local decode readiness.
"""

from __future__ import annotations

import math
from multiprocessing import shared_memory
import queue
import time
import traceback

import numpy as np


class SRTTimeline:
    """Anchor sender PTS once, keeping temporal slots independent of arrival jitter.

    ``source_time`` follows media PTS relative to the session epoch. Local
    processing availability (``scheduled_at``) is decode completion instead.
    Negative arrival drift is valid: later packets can arrive sooner than the
    first anchor's delay would predict. No clock offset is silently updated.
    """

    def __init__(self, fps, epoch, max_clock_lead_seconds=5.0):
        self.fps = float(fps)
        self.epoch = float(epoch)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("SRT frame rate must be positive and finite")
        if not math.isfinite(self.epoch):
            raise ValueError("SRT session epoch must be finite")
        self.max_clock_lead_seconds = float(max_clock_lead_seconds)
        if not math.isfinite(self.max_clock_lead_seconds) or self.max_clock_lead_seconds <= 0:
            raise ValueError("SRT maximum clock lead must be positive and finite")
        self.anchor_pts = None
        self.anchor_decoded = None
        self.anchor_sequence = None
        self.last_pts = None
        self.last_sequence = -1
        self.read_count = 0
        self.duplicate_reads_dropped = 0

    def map(self, pts_seconds, decoded_at):
        if pts_seconds is None or not math.isfinite(float(pts_seconds)):
            raise ValueError("SRT frame has missing or non-finite presentation timestamp (PTS)")
        pts = float(pts_seconds)
        decoded = float(decoded_at)
        if not math.isfinite(decoded):
            raise ValueError("SRT decode timestamp must be finite")
        if decoded < self.epoch:
            return None
        if self.last_pts is not None and pts < self.last_pts:
            raise ValueError(
                f"SRT presentation timestamps moved backwards: {self.last_pts:g} -> {pts:g}; "
                "start a new session after reconnecting the sender"
            )
        self.last_pts = pts
        self.read_count += 1
        if self.anchor_pts is None:
            self.anchor_pts = pts
            self.anchor_decoded = decoded
            self.anchor_sequence = int((decoded - self.epoch) * self.fps)
        relative_pts = pts - self.anchor_pts
        sequence = self.anchor_sequence + int(math.floor(relative_pts * self.fps + 0.5))
        if sequence <= self.last_sequence:
            self.duplicate_reads_dropped += 1
            return None
        source_time = self.anchor_decoded - self.epoch + relative_pts
        if self.epoch + source_time - decoded > self.max_clock_lead_seconds:
            raise ValueError(
                "SRT presentation timestamp jumped too far ahead of the local clock; "
                "start a new session after restarting or reconnecting the sender"
            )
        self.last_sequence = sequence
        return {
            "sequence": sequence,
            "source_time": source_time,
            "source_pts_seconds": pts,
            "scheduled_at": decoded,
            "acquired_at": decoded,
            "decoded_at": decoded,
            "arrival_pts_drift_ms": (decoded - self.epoch - source_time) * 1000.0,
            "read_count": self.read_count,
            "duplicate_reads_dropped": self.duplicate_reads_dropped,
        }


def _open_srt(source, options):
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "SRT input requires the optional PyAV dependency. Install it in the "
            "AirPiano environment with `python -m pip install av`; use a PyAV/FFmpeg "
            "build with libsrt support. Camera and video input do not require PyAV."
        ) from error
    open_timeout = float(options.get("open_timeout_seconds", 120.0))
    read_timeout = float(options.get("read_timeout_seconds", 5.0))
    for name, timeout in (("open_timeout_seconds", open_timeout),
                          ("read_timeout_seconds", read_timeout)):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"SRT {name} must be positive and finite")
    try:
        container = av.open(str(source), format="mpegts", timeout=(open_timeout, read_timeout))
    except Exception as error:
        if "protocol" in str(error).lower():
            raise RuntimeError(
                "Cannot open SRT input: this PyAV/FFmpeg build may lack libsrt support. "
                "Install a build that supports the srt protocol."
            ) from error
        raise RuntimeError(
            f"Cannot open SRT input ({type(error).__name__}: {error}). "
            "Check the sender address, caller/listener modes, UDP reachability, "
            "and open_timeout_seconds."
        ) from error
    return av, container


def _srt_capture_worker(source, fps, options, state, ready, free, lock,
                        control, begin, stop, recording_sink=None):
    """Decode continuously, then publish post-start frames in the latest slot."""
    # Import here to reuse the mailbox without a module-level dependency cycle.
    from .io import _write_capture_frame

    container = memory = None
    ready_sent = False
    try:
        opened_at = time.perf_counter()
        av, container = _open_srt(source, options)
        if not container.streams.video:
            raise ValueError("SRT MPEG-TS input contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        source_fps = float(stream.average_rate) if stream.average_rate else None
        if source_fps is not None and (not math.isfinite(source_fps)
                                      or abs(source_fps - fps) > 0.1):
            raise ValueError(
                f"SRT video must provide approximately {fps:g} Hz, got {source_fps:g}"
            )
        image_buffer = shape = timeline = epoch = None
        dropped = accepted_frames = 0
        previous_pts = None
        for decoded_frame in container.decode(stream):
            decoded_at = time.perf_counter()
            if stop.is_set():
                return
            pts = decoded_frame.time
            if pts is None or not math.isfinite(float(pts)):
                raise ValueError("SRT frame has missing or non-finite presentation timestamp (PTS)")
            if previous_pts is not None and float(pts) < previous_pts:
                raise ValueError(
                    "SRT presentation timestamps moved backwards; start a new session "
                    "after restarting or reconnecting the sender"
                )
            previous_pts = float(pts)
            frame_shape = [int(decoded_frame.height), int(decoded_frame.width), 3]
            if shape is not None and frame_shape != shape:
                raise ValueError(
                    f"SRT video resolution changed from {shape[:2]} to {frame_shape[:2]}; "
                    "keep the phone orientation and output resolution fixed for the session"
                )
            if shape is None:
                image = decoded_frame.to_ndarray(format="bgr24")
                if image.dtype != np.uint8 or list(image.shape) != frame_shape:
                    raise ValueError("SRT decoder did not produce an H x W x 3 uint8 BGR image")
                shape = frame_shape
                memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
                image_buffer = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
                np.copyto(image_buffer, image)
                control.put({"kind": "ready", "metadata": {
                    "kind": "srt", "source": str(source), "fps": source_fps,
                    "frame_shape": shape, "frame_count": None,
                    "startup_seconds": time.perf_counter() - opened_at,
                    "shared_memory_name": memory.name,
                    "codec": stream.codec_context.name,
                    "pyav_version": av.__version__,
                    "source_clock": "sender_pts_anchored_to_first_post_start_decode",
                    "processing_clock": "local_decode_ready_before_bgr_conversion",
                    "open_timeout_seconds": float(options.get("open_timeout_seconds", 120.0)),
                    "read_timeout_seconds": float(options.get("read_timeout_seconds", 5.0)),
                }})
                ready_sent = True
                # The first image belongs to startup and only warms the models.
                continue
            if epoch is None:
                try:
                    epoch = float(begin.get_nowait())
                    timeline = SRTTimeline(
                        fps, epoch, max(5.0, float(options.get("read_timeout_seconds", 5.0)))
                    )
                except queue.Empty:
                    # Keep consuming all decoded frames while models warm up;
                    # neither a packet backlog nor a Python image queue grows.
                    continue
            if decoded_at < epoch:
                continue
            frame = timeline.map(pts, decoded_at)
            if frame is None:
                continue
            image = decoded_frame.to_ndarray(format="bgr24")
            if image.dtype != np.uint8 or list(image.shape) != shape:
                raise ValueError("SRT decoder did not produce an H x W x 3 uint8 BGR image")
            frame.update(image=image, captured_frames=accepted_frames + 1)
            if recording_sink is not None:
                recording_sink.offer(image, frame["source_time"])
            dropped = _write_capture_frame(
                image_buffer, state, ready, free, lock, frame, stop, True, dropped
            )
            accepted_frames += 1
        if not stop.is_set():
            raise RuntimeError(
                "SRT video stream ended or disconnected; reconnect the sender and start a new session"
            )
    except Exception:
        if not stop.is_set():
            control.put({"kind": "error", "traceback": traceback.format_exc()})
            raise
    finally:
        try:
            if recording_sink is not None:
                recording_sink.close()
            if container is not None:
                container.close()
        finally:
            if memory is not None:
                memory.close()
                if not ready_sent:
                    memory.unlink()
