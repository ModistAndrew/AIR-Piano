"""Optional, bounded thumbnail recording on the same timeline as exported MIDI.

Capture offers BGR thumbnails to a one-frame shared-memory mailbox without waiting
for the encoder. A separate process emits CFR H.264: missing positions hold the
previous image, and positions before the first image are black. No audio is used.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import tempfile
import time

import cv2
import numpy as np


class RecordingSink:
    """Spawn-safe, single-producer latest-thumbnail mailbox.

    Only the parent ``VideoRecorder`` unlinks shared memory. Each process calls
    ``close`` to detach its own lazily opened handle. All encoder I/O stays in the
    recorder process; ``offer`` never waits to acquire the image lock.
    """

    def __init__(self, context, name, max_dimension):
        self.name = name
        self.max_dimension = max_dimension
        self.lock = context.Lock()
        self.ready = context.Event()
        self.enabled = context.Event()
        self.enabled.set()
        self.stopping = context.Event()
        self.state = context.Array("d", 4, lock=False)  # timestamp, height, width, last accepted time
        self.state[3] = -1.0
        self.counts = context.Array("q", 5, lock=False)  # offered, submitted, overwritten, lock skipped, invalid
        self.cost = context.Array("d", 2, lock=False)  # total and max milliseconds
        self.error = context.Array("c", 2048, lock=False)
        self.end_time = context.Value("d", 0.0, lock=False)
        self.encoder_pid = context.Value("q", 0, lock=False)
        self._memory = None
        self._buffer = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_memory"] = state["_buffer"] = None
        return state

    def _attach(self):
        if self._memory is None:
            self._memory = shared_memory.SharedMemory(name=self.name)
            self._buffer = np.ndarray(
                (self.max_dimension * self.max_dimension * 3,),
                dtype=np.uint8, buffer=self._memory.buf,
            )
        return self._buffer

    def _fail(self, message):
        self.enabled.clear()
        if not self.error.value:
            self.error.value = str(message).encode("utf-8", errors="replace")[:2047]

    def offer(self, image, source_time):
        if not self.enabled.is_set():
            return False
        started = time.perf_counter()
        self.counts[0] += 1
        try:
            stamp = float(source_time)
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError("Recording source_time must be finite and nonnegative")
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Recording requires H x W x 3 uint8 BGR images")
            height, width = image.shape[:2]
            if min(height, width) < 2:
                raise ValueError("Recording images must be at least 2 x 2")
            scale = min(1.0, self.max_dimension / max(height, width))
            target_height = max(2, int(height * scale) // 2 * 2)
            target_width = max(2, int(width * scale) // 2 * 2)
            thumbnail = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            buffer = self._attach()
            if not self.lock.acquire(block=False):
                self.counts[3] += 1
                return False
            try:
                if not self.enabled.is_set():
                    return False
                if self.ready.is_set():
                    self.counts[2] += 1
                buffer[:thumbnail.size] = thumbnail.reshape(-1)
                self.state[:] = (stamp, target_height, target_width, max(self.state[3], stamp))
                self.counts[1] += 1
                self.ready.set()
            finally:
                self.lock.release()
            return True
        except Exception as error:
            self.counts[4] += 1
            self._fail(f"Video recording offer failed: {type(error).__name__}: {error}")
            return False
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.cost[0] += elapsed_ms
            self.cost[1] = max(self.cost[1], elapsed_ms)

    def close(self):
        self._buffer = None
        if self._memory is not None:
            self._memory.close()
            self._memory = None


class _CFRResampler:
    """Emit complete frame slots without compressing gaps in the source clock."""

    def __init__(self, fps, emit):
        self.fps = fps
        self.emit = emit
        self.next_index = 0
        self.previous = None
        self.source_frames = 0
        self.repeated_frames = 0
        self.leading_black_frames = 0
        self.late_frames = 0

    def push(self, image, source_time):
        index = int(math.floor(float(source_time) * self.fps + 0.5))
        if index < self.next_index:
            self.late_frames += 1
            return
        gap = index - self.next_index
        filler = self.previous if self.previous is not None else np.zeros_like(image)
        for _ in range(gap):
            self.emit(filler)
        if self.previous is None:
            self.leading_black_frames += gap
        else:
            self.repeated_frames += gap
        self.emit(image)
        self.previous = image
        self.source_frames += 1
        self.next_index = index + 1

    def finish(self, end_time):
        if self.previous is None:
            return
        target = max(self.next_index, int(math.ceil(max(0.0, end_time) * self.fps - 1e-9)))
        for _ in range(target - self.next_index):
            self.emit(self.previous)
            self.repeated_frames += 1
        self.next_index = target


def _write_all(handle, image):
    view = memoryview(np.ascontiguousarray(image)).cast("B")
    while view:
        written = handle.write(view)
        if not written:
            raise BrokenPipeError("FFmpeg closed the raw video input")
        view = view[written:]


def _record_worker(output_path, fps, crf, sink, result_queue):
    process = stderr_file = resampler = None
    shape = None
    report = {"status": "no_frames", "error": None}
    try:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Video recording needs FFmpeg with libx264; install ffmpeg or disable recording")
        buffer = sink._attach()
        while True:
            frame = None
            if sink.ready.wait(timeout=0.05):
                if sink.lock.acquire(timeout=0.05):
                    try:
                        if sink.ready.is_set():
                            stamp, height, width, _ = sink.state[:]
                            height, width = int(height), int(width)
                            frame = buffer[:height * width * 3].reshape(height, width, 3).copy()
                            sink.ready.clear()
                    finally:
                        sink.lock.release()
                elif sink.stopping.is_set():
                    raise RuntimeError("Video recorder could not acquire the final thumbnail slot")
            if frame is not None:
                if shape is None:
                    shape = frame.shape
                    stderr_file = tempfile.TemporaryFile()
                    command = [
                        executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{shape[1]}x{shape[0]}",
                        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", str(crf), "-threads", "1", "-pix_fmt", "yuv420p",
                        "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "1000000",
                        "-f", "mp4", str(output_path),
                    ]
                    process = subprocess.Popen(
                        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=stderr_file, bufsize=0, start_new_session=(os.name != "nt"),
                    )
                    sink.encoder_pid.value = process.pid
                    resampler = _CFRResampler(fps, lambda image: _write_all(process.stdin, image))
                elif frame.shape != shape:
                    raise RuntimeError("Recording dimensions changed during the session")
                resampler.push(frame, stamp)
            if sink.stopping.is_set() and not sink.ready.is_set():
                break
            if sink.error.value:
                raise RuntimeError(sink.error.value.decode("utf-8", errors="replace"))
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"FFmpeg exited unexpectedly with code {process.returncode}")
        if resampler is not None:
            end_time = max(float(sink.end_time.value), float(sink.state[3]) + 1 / fps)
            resampler.finish(end_time)
            process.stdin.close()
            code = process.wait(timeout=8.0)
            if code:
                raise RuntimeError(f"FFmpeg exited with code {code}")
            report["status"] = "completed"
        if sink.error.value:
            raise RuntimeError(sink.error.value.decode("utf-8", errors="replace"))
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        sink._fail(report["error"])
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            sink.encoder_pid.value = 0
        if stderr_file is not None:
            stderr_file.seek(0, os.SEEK_END)
            length = stderr_file.tell()
            stderr_file.seek(max(0, length - 4096))
            stderr = stderr_file.read().decode("utf-8", errors="replace").strip()
            stderr_file.close()
            if stderr and report["status"] == "error":
                report["error"] += f"; FFmpeg: {stderr}"
        report.update(
            width=shape[1] if shape else None, height=shape[0] if shape else None,
            output_frames=resampler.next_index if resampler else 0,
            encoded_source_frames=resampler.source_frames if resampler else 0,
            repeated_frames=resampler.repeated_frames if resampler else 0,
            leading_black_frames=resampler.leading_black_frames if resampler else 0,
            late_frames=resampler.late_frames if resampler else 0,
        )
        sink.close()
        result_queue.put(report)


class VideoRecorder:
    """Own the optional encoder, shared-memory slot, and bounded finalization."""

    def __init__(self, output_path, max_dimension=640, fps=30, crf=30):
        if isinstance(max_dimension, bool) or int(max_dimension) != max_dimension or max_dimension < 2:
            raise ValueError("Recording max_dimension must be an integer of at least 2")
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("Recording fps must be finite and positive")
        if isinstance(crf, bool) or int(crf) != crf or not 0 <= crf <= 51:
            raise ValueError("Recording crf must be an integer from 0 to 51")
        self.path = Path(output_path)
        self.fps = float(fps)
        self.context = mp.get_context("spawn")
        self.memory = shared_memory.SharedMemory(create=True, size=int(max_dimension) ** 2 * 3)
        self.sink = RecordingSink(self.context, self.memory.name, int(max_dimension))
        self.results = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            name="airpiano-video-recorder", target=_record_worker,
            args=(str(self.path), self.fps, int(crf), self.sink, self.results),
        )
        self._report = None
        self._finished = False
        self._started = False

    def start(self):
        try:
            self.process.start()
            self._started = True
        except Exception as error:
            self.sink._fail(f"Cannot start video recorder: {type(error).__name__}: {error}")
            self._report = {"status": "error", "error": self.sink.error.value.decode("utf-8", errors="replace")}
        return self.healthcheck()

    def _receive(self):
        try:
            self._report = self.results.get_nowait()
        except queue.Empty:
            pass

    def healthcheck(self):
        if not self._finished:
            self._receive()
            if self._report is None and self._started and self.process.exitcode is not None:
                self.sink._fail(f"Video recorder exited unexpectedly: {self.process.exitcode}")
                self._report = {"status": "error", "error": self.sink.error.value.decode("utf-8", errors="replace")}
            if self.sink.error.value and self._report is None:
                self._report = {"status": "error", "error": self.sink.error.value.decode("utf-8", errors="replace")}
        return dict(self._report or {"status": "recording" if self._started else "not_started", "error": None})

    def stop_accepting(self):
        self.sink.enabled.clear()

    def _kill_encoder(self, sig):
        pid = self.sink.encoder_pid.value
        if pid:
            try:
                if os.name != "nt":
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def finish(self, end_time, timeout=10.0):
        if self._finished:
            return dict(self._report)
        self.stop_accepting()
        timeout = max(0.0, float(timeout))
        requested_end = float(end_time)
        if not math.isfinite(requested_end):
            requested_end = 0.0
            self.sink._fail("Recording final duration was non-finite")
        self.sink.end_time.value = max(0.0, requested_end, self.sink.state[3] + 1 / self.fps)
        self.sink.stopping.set()
        forced_error = None
        try:
            if self._started:
                self.process.join(timeout=timeout)
                if self.process.is_alive():
                    forced_error = "Video recorder finalization timed out; partial fragmented MP4 may be recoverable"
                    self._kill_encoder(signal.SIGTERM)
                    self.process.join(timeout=1.0)
                if self.process.is_alive():
                    self._kill_encoder(signal.SIGKILL)
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1.0)
                # A worker killed externally cannot reap its FFmpeg child.
                # Own that process group independently of the worker's liveness.
                if self.sink.encoder_pid.value:
                    self._kill_encoder(signal.SIGKILL)
                    self.sink.encoder_pid.value = 0
                self._receive()
            if self._report is None:
                if not self._started and not self.sink.error.value:
                    self._report = {"status": "no_frames", "error": None}
                else:
                    error = self.sink.error.value.decode("utf-8", errors="replace") or f"Video recorder returned no report (exit={self.process.exitcode})"
                    self._report = {"status": "error", "error": error}
            if forced_error:
                self._report.update(status="error", error=forced_error)
            for field, default in (("width", None), ("height", None), ("output_frames", 0),
                                   ("encoded_source_frames", 0), ("repeated_frames", 0),
                                   ("leading_black_frames", 0), ("late_frames", 0)):
                self._report.setdefault(field, default)
            offered, submitted, overwritten, skipped, invalid = self.sink.counts[:]
            output_frames = self._report["output_frames"]
            self._report.update(
                path=str(self.path), fps=self.fps,
                offered_frames=int(offered), submitted_frames=int(submitted),
                overwritten_frames=int(overwritten), lock_skipped_frames=int(skipped), invalid_frames=int(invalid),
                dropped_offer_frames=int(overwritten + skipped + invalid),
                duration_seconds=output_frames / self.fps,
                requested_end_seconds=max(0.0, requested_end),
                offer_cost_ms={"count": int(offered), "average": self.sink.cost[0] / offered if offered else None,
                               "max": self.sink.cost[1] if offered else None, "total": self.sink.cost[0]},
                size_bytes=self.path.stat().st_size if self.path.exists() else 0,
            )
        finally:
            self.sink.close()
            self.memory.close()
            self.memory.unlink()
            self.results.cancel_join_thread()
            self.results.close()
            self._finished = True
        return dict(self._report)
