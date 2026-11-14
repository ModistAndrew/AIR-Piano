"""Bounded OpenCV/SRT capture and spawned WiLoR workers for streaming inference.

Images from capture are BGR, as expected by the existing WiLoR backend. Camera
acquisition timestamps remain host read-completion times; an optional calibrated
delay shifts the source timeline to estimated capture time.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import math
import queue
import time
import traceback

import cv2
import numpy as np


def _close_processes(processes, queues, stop=None, grace_seconds=2.0):
    if stop is not None:
        stop.set()
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
    # A dead receiver cannot drain a Queue feeder. Never wait for that feeder
    # during shutdown; outstanding messages belong to the stopped session.
    for channel in queues:
        channel.cancel_join_thread()
        channel.close()


def _write_capture_frame(image_buffer, state, ready, free, lock, frame,
                         stop, drop_stale, dropped):
    """Publish one image without serializing it through a multiprocessing pipe."""
    if not drop_stale:
        while not stop.is_set():
            if free.wait(timeout=0.1):
                break
        else:
            return dropped
    if stop.is_set():
        return dropped
    with lock:
        if ready.is_set():
            dropped += 1
        np.copyto(image_buffer, frame["image"])
        state[:] = (
            frame["sequence"], frame["source_time"], frame["acquired_at"],
            frame["scheduled_at"], dropped, frame["read_count"],
            frame["duplicate_reads_dropped"], frame["captured_frames"],
            frame.get("source_pts_seconds", float("nan")),
            frame.get("decoded_at", float("nan")),
            frame.get("arrival_pts_drift_ms", float("nan")),
        )
        free.clear()
        ready.set()
    return dropped


def _capture_worker(source, kind, fps, width, height, drop_stale,
                    state, ready, free, lock, control, begin, stop, recording_sink=None,
                    capture_delay_seconds=0.0):
    capture = memory = None
    ready_sent = False
    read_count = 0
    latest_camera_frame = None
    dropped = 0
    try:
        opened_at = time.perf_counter()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {kind} source: {source}")
        first_frame = None
        if kind == "camera":
            # Select MJPEG before size/rate: raw YUYV on USB webcams can cap
            # 1920x1080 at 5 Hz even when the camera supports MJPEG at 30 Hz.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            # Keep one driver buffer available while OpenCV decodes the other.
            # A single buffer halves the W1 webcam's delivered frame rate.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if source_fps > 0 and abs(source_fps - fps) > 0.1:
                raise ValueError(
                    f"Camera must provide approximately {fps:g} Hz, got {source_fps:g}"
                )
            frame_count = None
            shape = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 3]
        else:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if abs(source_fps - fps) > 1e-3:
                raise ValueError(f"Video must be native {fps:g} Hz, got {source_fps:g}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, first_frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode first frame: {source}")
            shape = list(first_frame.shape)
        memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        image_buffer = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
        if first_frame is not None:
            np.copyto(image_buffer, first_frame)
        control.put({"kind": "ready", "metadata": {
            "kind": kind, "source": str(source), "fps": source_fps,
            "frame_shape": shape, "frame_count": frame_count,
            "startup_seconds": time.perf_counter() - opened_at,
            "shared_memory_name": memory.name,
            **({"frame_index_basis": "camera_read_sequence",
                "camera_capture_delay_ms": capture_delay_seconds * 1000.0,
                "source_time_basis": ("estimated_capture_time" if capture_delay_seconds
                                      else "read_completion")}
               if kind == "camera" else {}),
        }})
        ready_sent = True
        while not stop.is_set():
            try:
                epoch = begin.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        else:
            return
        sequence = 0
        accepted_frames = 0
        while not stop.is_set():
            if kind == "video":
                scheduled = epoch + sequence / fps
                if stop.wait(max(0.0, scheduled - time.perf_counter())):
                    return
                if sequence == 0:
                    ok, image = True, first_frame
                    acquired = scheduled
                else:
                    ok, image = capture.read()
                    acquired = time.perf_counter()
                source_time = sequence / fps
            else:
                ok, image = capture.read()
                acquired = time.perf_counter()
                # Drain frames estimated to precede the session, including
                # those still in the capture pipeline when the clock starts.
                if ok and acquired < epoch + capture_delay_seconds:
                    continue
                source_time = acquired - (epoch + capture_delay_seconds)
                if ok:
                    read_count += 1
                    latest_camera_frame = {
                        "sequence": read_count - 1, "source_time": source_time,
                        "acquired_at": acquired, "scheduled_at": acquired,
                    }
                scheduled = acquired
            if not ok:
                if kind == "camera":
                    raise RuntimeError(f"Camera read failed: {source}")
                if frame_count > 0 and sequence < frame_count:
                    raise RuntimeError(
                        f"Video ended at frame {sequence}, expected {frame_count}: {source}"
                    )
                control.put({"kind": "eof", "frames": sequence,
                             "read_count": sequence, "last_sequence": sequence - 1,
                             "dropped": dropped})
                return
            frame = {"sequence": sequence, "source_time": source_time,
                     "acquired_at": acquired, "scheduled_at": scheduled,
                     "image": image,
                     "read_count": read_count if kind == "camera" else sequence + 1,
                     "duplicate_reads_dropped": 0,
                     "captured_frames": accepted_frames + 1}
            if recording_sink is not None:
                recording_sink.offer(image, source_time)
            dropped = _write_capture_frame(
                image_buffer, state, ready, free, lock, frame, stop, drop_stale, dropped
            )
            accepted_frames += 1
            # Camera IDs count successful post-start reads, independent of
            # arrival jitter. Source timestamps still carry the actual timing.
            sequence += 1
    except Exception:
        control.put({"kind": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        if kind == "camera":
            # A read may complete after stop or remain unread in the mailbox.
            # Preserve its count/timing independently of image publication.
            control.put({"kind": "camera_stopped", "frames": read_count,
                         "read_count": read_count, "last_sequence": read_count - 1,
                         "latest_frame": latest_camera_frame, "dropped": dropped,
                         "stopped_at": time.perf_counter()})
        if recording_sink is not None:
            recording_sink.close()
        if capture is not None:
            capture.release()
        if memory is not None:
            memory.close()
            if not ready_sent:
                memory.unlink()


class CaptureSource:
    """A terminable capture process with one shared-memory latest-frame slot.

    ``start(); wait_ready(); begin(epoch)`` separates initialization from capture.
    Camera reads start only after ``begin``. Video frame 0 is primed before ready.
    ``drop_stale=False`` retains every video frame, exposing lag in its original
    schedule. Camera sequence numbers count successful post-epoch reads: arrival
    jitter and capture stalls do not discard reads or create synthetic IDs.
    ``source_time`` is read completion minus the optional calibrated camera delay;
    ``acquired_at`` and processing deadlines always retain real read completion.
    Estimated pre-session frames are drained before assigning IDs; ``read_count``
    counts the remaining successful reads. The legacy duplicate counter stays zero.
    SRT uses optional PyAV and decodes continuously through warmup. Its model
    timeline follows sender PTS, while processing deadlines begin at local decode
    completion. SRT requires ``drop_stale=True`` and has no automatic reconnect.
    """

    def __init__(self, source, *, kind="video", fps=30.0, width=1920,
                 height=1080, capacity=1, drop_stale=True, srt_options=None,
                 recording_sink=None, capture_delay_ms=0.0):
        if kind not in {"video", "camera", "srt"}:
            raise ValueError("kind must be 'video', 'camera', or 'srt'")
        if kind == "srt" and not drop_stale:
            raise ValueError("SRT live input requires drop_stale=True")
        if kind == "srt" and not str(source).lower().startswith("srt://"):
            raise ValueError("SRT source must be an srt:// URL")
        if capacity != 1 or fps <= 0:
            raise ValueError("capture uses one latest-frame slot; fps must be positive")
        capture_delay_ms = float(capture_delay_ms)
        if not math.isfinite(capture_delay_ms) or capture_delay_ms < 0:
            raise ValueError("capture_delay_ms must be finite and nonnegative")
        if kind != "camera" and capture_delay_ms != 0:
            raise ValueError("capture_delay_ms is supported only for camera input")
        self.context = mp.get_context("spawn")
        self.control = self.context.Queue()
        self.begin_queue = self.context.Queue(maxsize=1)
        self.stop = self.context.Event()
        self.frame_state = self.context.Array("d", 11, lock=False)
        self.frame_ready = self.context.Event()
        self.frame_free = self.context.Event()
        self.frame_free.set()
        self.frame_lock = self.context.Lock()
        shared_args = (self.frame_state, self.frame_ready, self.frame_free, self.frame_lock,
                       self.control, self.begin_queue, self.stop)
        if kind == "srt":
            from .srt import _srt_capture_worker
            target = _srt_capture_worker
            args = (source, float(fps), dict(srt_options or {}), *shared_args, recording_sink)
        else:
            target = _capture_worker
            args = (source, kind, float(fps), width, height, drop_stale, *shared_args,
                    recording_sink, capture_delay_ms / 1000.0)
        self.process = self.context.Process(
            name="airpiano-capture", target=target, args=args,
        )
        self.metadata = None
        self.memory = None
        self._image = None
        self._terminal = None
        self._camera_stopped = None
        self._last_sequence = -1
        self._eof_delivered = False
        self._closed = False

    def start(self):
        self.process.start()

    def _attach(self, metadata):
        self.metadata = metadata
        self.memory = shared_memory.SharedMemory(name=metadata["shared_memory_name"])
        self._image = np.ndarray(metadata["frame_shape"], dtype=np.uint8,
                                 buffer=self.memory.buf)
        self.metadata["example_image"] = (
            self._image.copy() if metadata["kind"] in {"video", "srt"} else None
        )

    def _receive_control(self):
        failure = None
        while True:
            try:
                message = self.control.get_nowait()
            except queue.Empty:
                if failure is not None:
                    raise failure
                return
            if message["kind"] == "error":
                failure = RuntimeError("Capture process failed:\n" + message["traceback"])
            elif message["kind"] == "ready":
                self._attach(message["metadata"])
            elif message["kind"] == "camera_stopped":
                self._camera_stopped = message
            else:
                self._terminal = message

    def healthcheck(self):
        self._receive_control()
        if not self._closed and self.process.exitcode is not None:
            if self.process.exitcode != 0 or self._terminal is None:
                raise RuntimeError(
                    f"Capture process exited unexpectedly: {self.process.exitcode}"
                )

    def wait_ready(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while self.metadata is None:
            self.healthcheck()
            if time.monotonic() >= deadline:
                raise TimeoutError("Capture initialization timed out")
            time.sleep(0.01)
        return self.metadata

    def begin(self, epoch):
        self.begin_queue.put(float(epoch))

    def poll(self, timeout=0.0):
        self.healthcheck()
        if self.frame_ready.wait(timeout=max(0.0, timeout)):
            if not self.frame_lock.acquire(timeout=0.05):
                self.healthcheck()
                return None
            try:
                if self.frame_ready.is_set():
                    (sequence, source_time, acquired, scheduled, dropped, reads, duplicates,
                     captured, pts, decoded, drift) = self.frame_state[:]
                    frame = {
                        "kind": "frame", "sequence": int(sequence),
                        "source_time": source_time, "acquired_at": acquired,
                        "scheduled_at": scheduled, "dropped_total": int(dropped),
                        "read_count": int(reads), "duplicate_reads_dropped": int(duplicates),
                        "captured_frames": int(captured), "image": self._image.copy(),
                        "source_pts_seconds": None if np.isnan(pts) else pts,
                        "decoded_at": None if np.isnan(decoded) else decoded,
                        "arrival_pts_drift_ms": None if np.isnan(drift) else drift,
                    }
                    self.frame_ready.clear()
                    self.frame_free.set()
                    self._last_sequence = int(sequence)
                    return frame
            finally:
                self.frame_lock.release()
        self.healthcheck()
        if (self._terminal is not None and not self._eof_delivered
                and self._last_sequence >= self._terminal["last_sequence"]):
            self._eof_delivered = True
            return self._terminal
        return None

    def close(self):
        """Stop capture and return camera-only final read accounting when available."""
        if self._closed:
            return self._camera_stopped
        self._closed = True
        _close_processes([self.process], [], self.stop)
        # A caller can stop before wait_ready(). Recover the advertised buffer
        # after the child exits so the parent still owns all unlink operations.
        failure = None
        try:
            self._receive_control()
        except RuntimeError as error:
            failure = error
        finally:
            self._image = None
            if self.memory is not None:
                self.memory.close()
                self.memory.unlink()
                self.memory = None
            _close_processes([], [self.control, self.begin_queue])
        if failure is not None:
            raise failure
        return self._camera_stopped


def _warm_pose_shapes(model, model_cfg, device, hand_counts):
    """Warm the same normal-tensor dispatch guards as production crop batches."""
    import torch

    image_size = int(model_cfg.MODEL.IMAGE_SIZE)
    for count in hand_counts:
        # Production DataLoader/.to creates normal tensors before inference_mode.
        # Creating these inside it would warm different Dynamo dispatch guards.
        crops = torch.zeros(
            (count, 3, image_size, image_size), device=device,
            dtype=next(model.parameters()).dtype,
        )
        with torch.inference_mode():
            model({"img": crops})


def _wilor_worker(device, name, shape, jobs, results, control, stop,
                  fast, warmup, hand_counts, threads, backend_kwargs):
    memory = None
    try:
        import torch
        from airpiano.preprocessing.video import VideoToWiLoR

        if threads is not None:
            torch.set_num_threads(threads)
        torch.cuda.set_device(torch.device(device))
        memory = shared_memory.SharedMemory(name=name)
        frames = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
        startup = time.perf_counter()
        backend = VideoToWiLoR(
            **dict(backend_kwargs, device=device, frame_batch_size=1, fast=fast)
        )._backend
        models = backend.models
        # Warm the complete pose network, including MANO and refinement, even
        # when camera startup supplies a black detector example with no hands.
        _warm_pose_shapes(models.model, models.model_cfg, models.device, hand_counts)
        for _ in range(warmup):
            example = backend.build_chunk([0], [0.0], [frames[0]])
        torch.cuda.synchronize(models.device)
        control.put({"kind": "ready", "device": device, "arrays": example,
                     "metadata": {
                         "device_name": torch.cuda.get_device_name(models.device),
                         "torch_threads": torch.get_num_threads(),
                         "pose_dtype": str(next(models.model.parameters()).dtype),
                         "fast": fast, "warmup_hand_counts": list(hand_counts),
                         "startup_seconds": time.perf_counter() - startup,
                     }})
        while not stop.is_set():
            try:
                job = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            sequence, slot, source_time = job
            started = time.perf_counter()
            arrays = backend.build_chunk([sequence], [source_time], [frames[slot]])
            torch.cuda.synchronize(models.device)
            results.put({"sequence": sequence, "slot": slot, "device": device,
                         "started_at": started, "finished_at": time.perf_counter(),
                         "arrays": arrays})
    except Exception:
        control.put({"kind": "error", "device": device,
                     "traceback": traceback.format_exc()})
        raise
    finally:
        if memory is not None:
            memory.close()


class WiLoRWorkerPool:
    """Spawned GPU workers with bounded shared-memory frame slots.

    ``poll`` releases an image slot; the caller separately limits outstanding
    frames until ordered AirPiano consumption to bound its own reorder buffer.
    Call submit/poll from one orchestration thread.
    """

    def __init__(self, devices, *, fast=True, warmup=10,
                 warmup_hand_counts=(1, 2, 3), capacity=4,
                 frame_shape=(1080, 1920, 3), torch_threads=None,
                 backend_kwargs=None):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("devices must be nonempty and distinct")
        if capacity < 1 or warmup < 1 or not warmup_hand_counts or any(
            count < 1 for count in warmup_hand_counts
        ):
            raise ValueError("capacity, warmup and hand counts must be nonempty/positive")
        self.context = mp.get_context("spawn")
        self.devices = list(devices)
        self.capacity = capacity
        self.frame_shape = tuple(frame_shape)
        self.fast = fast
        self.warmup = warmup
        self.hand_counts = tuple(warmup_hand_counts)
        self.threads = torch_threads
        self.backend_kwargs = backend_kwargs or {}
        self.jobs = self.context.Queue(maxsize=capacity)
        self.results = self.context.Queue(maxsize=capacity)
        self.control = self.context.Queue()
        self.stop = self.context.Event()
        self.workers = []
        self.memory = None
        self.frames = None
        self.free_slots = list(range(capacity))
        self.metadata = {}
        self.example_arrays = None
        self._closed = False

    def start(self, example_image=None):
        if example_image is not None:
            self.frame_shape = tuple(example_image.shape)
        shape = (self.capacity, *self.frame_shape)
        self.memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self.frames = np.ndarray(shape, dtype=np.uint8, buffer=self.memory.buf)
        self.frames[0] = 0 if example_image is None else example_image
        for device in self.devices:
            worker = self.context.Process(
                name=f"airpiano-wilor-{device}", target=_wilor_worker,
                args=(device, self.memory.name, shape, self.jobs, self.results,
                      self.control, self.stop, self.fast, self.warmup,
                      self.hand_counts, self.threads, self.backend_kwargs),
            )
            self.workers.append(worker)
            worker.start()

    def healthcheck(self):
        while True:
            try:
                message = self.control.get_nowait()
            except queue.Empty:
                break
            if message["kind"] == "error":
                raise RuntimeError(
                    f"WiLoR {message['device']} failed:\n{message['traceback']}"
                )
            self.metadata[message["device"]] = message["metadata"]
            self.example_arrays = message["arrays"]
        if not self._closed:
            for worker in self.workers:
                if worker.exitcode is not None:
                    raise RuntimeError(
                        f"WiLoR worker {worker.name} exited unexpectedly: {worker.exitcode}"
                    )

    def wait_ready(self, timeout=600.0):
        deadline = time.monotonic() + timeout
        while len(self.metadata) < len(self.devices):
            self.healthcheck()
            if time.monotonic() >= deadline:
                raise TimeoutError("WiLoR worker initialization timed out")
            time.sleep(0.01)
        return {"workers": self.metadata, "arrays": self.example_arrays}

    def submit(self, sequence, image, source_time):
        self.healthcheck()
        if not self.free_slots:
            return False
        if tuple(image.shape) != self.frame_shape or image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 BGR frame of shape {self.frame_shape}")
        slot = self.free_slots.pop()
        self.frames[slot] = image
        self.jobs.put_nowait((int(sequence), slot, float(source_time)))
        return True

    def poll(self, timeout=0.0):
        self.healthcheck()
        try:
            result = self.results.get(timeout=max(0.0, timeout))
        except queue.Empty:
            self.healthcheck()
            return None
        self.free_slots.append(result.pop("slot"))
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        _close_processes(self.workers, [self.jobs, self.results, self.control], self.stop)
        self.frames = None
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None
