"""Bounded, best-effort WebSocket note-ons for score releases and count-in cues.

Only the transport thread performs network I/O or calls ``emit``. Local send
completion is observable; remote receipt and audible playback aren't acknowledged.
A failed send is never retried because the peer may already have received it.
"""

from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from urllib.parse import urlsplit


class WebSocketMidiOutput:
    def __init__(self, url, capacity=32, max_age_ms=100, open_timeout_seconds=1,
                 close_timeout_seconds=.5, reconnect_seconds=1, emit=None):
        if urlsplit(str(url)).scheme not in {"ws", "wss"} or not urlsplit(str(url)).hostname:
            raise ValueError("MIDI output URL must be ws:// or wss:// with a host")
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("MIDI output capacity must be a positive integer")
        for name, value in (("max_age_ms", max_age_ms), ("open_timeout_seconds", open_timeout_seconds),
                            ("close_timeout_seconds", close_timeout_seconds), ("reconnect_seconds", reconnect_seconds)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"MIDI output {name} must be finite and positive")
        self.url = str(url)
        self.max_age_ms = float(max_age_ms)
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.close_timeout_seconds = float(close_timeout_seconds)
        self.reconnect_seconds = float(reconnect_seconds)
        self.emit = emit
        self._queue = queue.Queue(maxsize=int(capacity))
        self._notifications = queue.Queue(maxsize=int(capacity) * 2 + 8)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closing = threading.Event()
        self._connected = False
        self._accepting = True
        self._connection = None
        self._started = False
        self._status = "not_started"
        self._last_error = None
        self._stats = {name: 0 for name in (
            "submitted_events", "accepted_events", "sent_events", "send_failures",
            "connection_attempts", "connection_failures", "connections", "reconnections",
            "dropped_disconnected", "dropped_queue_full", "dropped_stale", "dropped_shutdown",
            "dropped_invalid", "callback_errors", "notification_overflow",
        )}
        self._queue_age_total = self._queue_age_max = 0.0
        self._send_age_total = self._send_age_max = 0.0
        self._shutdown_timed_out = False
        self.thread = threading.Thread(target=self._run, name="midi-websocket-output", daemon=True)

    def start(self):
        with self._lock:
            if self._started or self._closing.is_set():
                return self._snapshot_locked()
            self._started = True
            self._status = "connecting"
        try:
            self.thread.start()
        except Exception as error:
            with self._lock:
                self._status, self._last_error = "disabled", str(error)
                self._accepting = False
        return self.snapshot()

    def _notice_locked(self, record):
        try:
            self._notifications.put_nowait(record)
        except queue.Full:
            self._stats["notification_overflow"] += 1

    def _emit(self, record):
        if self.emit is not None:
            try:
                self.emit(record)
            except Exception as error:
                with self._lock:
                    self._stats["callback_errors"] += 1
                    self._last_error = f"Transport log callback failed: {error}"

    def _flush_notices(self):
        while True:
            try:
                self._emit(self._notifications.get_nowait())
            except queue.Empty:
                return

    def submit(self, release, *, released_at=None):
        """Queue one release or count-in cue without network, callback, or file I/O."""
        now = time.perf_counter()
        record = {"chord_id": release.get("chord_id"), "release_time": release.get("release_time")}
        if release.get("event_type") == "countdown":
            record.update(event_type="countdown", countdown_remaining=release["countdown_remaining"])
        try:
            pitches = list(release["pitches"])
            if not pitches or any(isinstance(pitch, bool) or int(pitch) != pitch or not 0 <= pitch <= 127 for pitch in pitches):
                raise ValueError("MIDI pitches must be integers from 0 through 127")
            pitches = [int(pitch) for pitch in pitches]
            source = now if released_at is None else float(released_at)
            if not math.isfinite(source):
                raise ValueError("released_at must be a finite monotonic timestamp")
            record.update(pitches=pitches, released_at=source)
            payload = str(pitches[0]) if len(pitches) == 1 else json.dumps(pitches, separators=(",", ":"))
        except Exception as error:
            with self._lock:
                self._stats["submitted_events"] += 1
                self._stats["dropped_invalid"] += 1
                self._notice_locked({**record, "status": "dropped", "reason": "invalid", "error": str(error), "timestamp": now})
            return False
        with self._lock:
            self._stats["submitted_events"] += 1
            reason = None
            if not self._accepting:
                reason = "shutdown"
            elif (now - source) * 1000 > self.max_age_ms:
                reason = "stale"
            elif not self._connected:
                reason = "disconnected"
            else:
                try:
                    self._queue.put_nowait((record, payload))
                    self._stats["accepted_events"] += 1
                    return True
                except queue.Full:
                    reason = "queue_full"
            self._stats[f"dropped_{reason}"] += 1
            self._notice_locked({**record, "status": "dropped", "reason": reason,
                                 "queue_delay_ms": max(0.0, (now - source) * 1000), "timestamp": now})
        return False

    def _drop_pending(self, reason):
        # Gate and clear under the producer's lock so nothing can slip into an
        # offline backlog between disconnect and reconnection.
        with self._lock:
            self._connected = False
            self._connection = None
            while True:
                try:
                    record, _ = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._stats[f"dropped_{reason}"] += 1
                self._notice_locked({**record, "status": "dropped", "reason": reason,
                                     "timestamp": time.perf_counter()})

    def _send_pending(self, connection):
        while not self._stop.is_set():
            self._flush_notices()
            if self._closing.is_set() and self._queue.empty():
                return
            try:
                record, payload = self._queue.get(timeout=.05)
            except queue.Empty:
                # Detect idle peer closure without waiting for the next note.
                if getattr(getattr(connection, "state", None), "name", None) == "CLOSED":
                    raise ConnectionError("WebSocket peer disconnected")
                continue
            now = time.perf_counter()
            age_ms = max(0.0, (now - record["released_at"]) * 1000)
            reason = "shutdown" if self._stop.is_set() else "stale" if age_ms > self.max_age_ms else None
            if reason is not None:
                with self._lock:
                    self._stats[f"dropped_{reason}"] += 1
                self._emit({**record, "status": "dropped", "reason": reason,
                            "timestamp": now, "queue_delay_ms": age_ms})
                continue
            try:
                connection.send(payload)
            except Exception as error:
                failed_at = time.perf_counter()
                with self._lock:
                    self._stats["send_failures"] += 1
                    self._last_error = str(error)
                self._emit({**record, "status": "send_failed", "timestamp": failed_at,
                            "queue_delay_ms": age_ms, "error": str(error), "delivery_uncertain": True})
                # Never requeue this release: a partial/failed write can still
                # have delivered the note-on to the peer.
                raise
            completed = time.perf_counter()
            completion_age = max(0.0, (completed - record["released_at"]) * 1000)
            with self._lock:
                self._stats["sent_events"] += 1
                self._queue_age_total += age_ms
                self._queue_age_max = max(self._queue_age_max, age_ms)
                self._send_age_total += completion_age
                self._send_age_max = max(self._send_age_max, completion_age)
            self._emit({**record, "status": "sent", "timestamp": completed,
                        "queue_delay_ms": age_ms, "local_release_to_send_ms": completion_age,
                        "send_duration_ms": (completed - now) * 1000, "payload": payload})

    def _run(self):
        try:
            try:
                from websockets.sync.client import connect
            except ImportError as error:
                with self._lock:
                    self._status = "disabled"
                    self._accepting = False
                    self._last_error = "WebSocket MIDI output requires websockets>=17.4 in the AirPiano environment"
                self._emit({"status": "disabled", "timestamp": time.perf_counter(), "error": self._last_error})
                return
            while not self._stop.is_set() and not self._closing.is_set():
                self._flush_notices()
                with self._lock:
                    self._status = "connecting"
                    self._stats["connection_attempts"] += 1
                connected = False
                try:
                    # Context management is the supported synchronous API in
                    # websockets 17; proxy and compression match the PC example.
                    with connect(self.url, proxy=None, compression=None,
                                 open_timeout=self.open_timeout_seconds,
                                 close_timeout=self.close_timeout_seconds) as connection:
                        if self._stop.is_set() or self._closing.is_set():
                            break
                        with self._lock:
                            self._connection = connection
                            self._connected = connected = True
                            self._status = "connected"
                            self._stats["connections"] += 1
                            self._stats["reconnections"] = max(0, self._stats["connections"] - 1)
                        self._emit({"status": "connected", "timestamp": time.perf_counter(), "url": self.url})
                        self._send_pending(connection)
                except Exception as error:
                    with self._lock:
                        self._last_error = str(error)
                        self._status = "disconnected"
                        if not connected:
                            self._stats["connection_failures"] += 1
                    self._emit({"status": "disconnected", "timestamp": time.perf_counter(), "error": str(error)})
                finally:
                    self._drop_pending("shutdown" if self._closing.is_set() else "disconnected")
                    self._flush_notices()
                if self._closing.is_set():
                    break
                self._stop.wait(self.reconnect_seconds)
        finally:
            self._drop_pending("shutdown")
            self._flush_notices()
            with self._lock:
                self._accepting = False
                if self._status != "disabled":
                    self._status = "closed"

    def _abort_socket(self):
        with self._lock:
            connection = self._connection
        sock = getattr(connection, "socket", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _snapshot_locked(self):
        count = self._stats["sent_events"]
        return {"url": self.url, "status": self._status, "connected": self._connected,
                **self._stats, "queued_events": self._queue.qsize(), "last_error": self._last_error,
                "queue_delay_ms": {"count": count, "average": self._queue_age_total / count if count else None,
                                   "max": self._queue_age_max if count else None},
                "local_release_to_send_ms": {"count": count, "average": self._send_age_total / count if count else None,
                                             "max": self._send_age_max if count else None},
                "shutdown_timed_out": self._shutdown_timed_out,
                "timing_scope": "Local release to local WebSocket send completion; no remote receipt or audio acknowledgment"}

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    healthcheck = snapshot

    def close(self, drain=True, timeout=2.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._lock:
            self._accepting = False
        self._closing.set()
        if not drain:
            self._stop.set()
            self._abort_socket()
        if self.thread.ident is not None:
            # Reserve a short interval to interrupt a stalled send and let its
            # context manager clean up the WebSocket's reader/keepalive threads.
            self.thread.join(timeout=max(0.0, deadline - time.monotonic() - .1))
            if self.thread.is_alive():
                self._stop.set()
                self._abort_socket()
                self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        else:
            self._drop_pending("shutdown")
            with self._lock:
                self._status = "closed"
        with self._lock:
            self._shutdown_timed_out = self.thread.is_alive()
            if not self._shutdown_timed_out and self._status != "disabled":
                self._status = "closed"
            return self._snapshot_locked()
