"""SRT timing/lifecycle checks without models or an iPhone.

Set AIRPIANO_TEST_SRT_INTEGRATION=1 to include the real localhost FFmpeg sender.
"""

from fractions import Fraction
import importlib.util
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from multiprocessing import shared_memory

import numpy as np

from airpiano.streaming.io import CaptureSource
from airpiano.streaming.srt import SRTTimeline, _open_srt, _srt_capture_worker


class SRTTimelineTests(unittest.TestCase):
    def test_bursty_decode_preserves_pts_spacing_and_missing_positions(self):
        timeline = SRTTimeline(30.0, 100.0)
        first = timeline.map(10.0, 100.25)
        second = timeline.map(10.0 + 1 / 30, 100.251)
        after_gap = timeline.map(10.0 + 4 / 30, 100.39)
        self.assertEqual([frame['sequence'] for frame in (first, second, after_gap)],
                         [7, 8, 11])
        self.assertAlmostEqual(second['source_time'] - first['source_time'], 1 / 30)
        self.assertAlmostEqual(after_gap['source_time'] - first['source_time'], 4 / 30)
        self.assertEqual(second['scheduled_at'], 100.251)
        self.assertEqual(second['acquired_at'], 100.251)
        self.assertLess(second['arrival_pts_drift_ms'], 0)
        self.assertAlmostEqual(second['source_pts_seconds'], 10 + 1 / 30)

    def test_same_slot_duplicate_does_not_compress_later_timeline(self):
        timeline = SRTTimeline(30.0, 100.0)
        self.assertEqual(timeline.map(20.0, 100.0)['sequence'], 0)
        self.assertIsNone(timeline.map(20.0, 100.01))
        self.assertIsNone(timeline.map(20.005, 100.02))
        self.assertEqual(timeline.map(20.1, 100.1)['sequence'], 3)

    def test_network_delay_changes_drift_but_not_processing_start(self):
        timeline = SRTTimeline(30.0, 100.0)
        timeline.map(42.0, 100.0)
        delayed = timeline.map(42.1, 100.8)
        self.assertEqual(delayed['sequence'], 3)
        self.assertAlmostEqual(delayed['source_time'], .1)
        self.assertAlmostEqual(delayed['arrival_pts_drift_ms'], 700)
        self.assertEqual(delayed['scheduled_at'], 100.8)

    def test_missing_and_nonfinite_pts_are_rejected(self):
        for value in (None, float('nan'), float('inf'), -float('inf')):
            with self.subTest(pts=value):
                with self.assertRaises((ValueError, RuntimeError)):
                    SRTTimeline(30.0, 100.0).map(value, 100.0)

    def test_backward_pts_is_a_discontinuity_even_inside_a_duplicate_slot(self):
        timeline = SRTTimeline(30.0, 100.0)
        timeline.map(10.0, 100.0)
        timeline.map(10.005, 100.01)
        with self.assertRaisesRegex((ValueError, RuntimeError), '(?i)backward|discontinu'):
            timeline.map(10.001, 100.02)


    def test_forward_timestamp_restart_cannot_create_unbounded_missing_history(self):
        timeline = SRTTimeline(30.0, 100.0)
        timeline.map(10.0, 100.0)
        with self.assertRaisesRegex(ValueError, 'jumped too far'):
            timeline.map(3610.0, 100.1)

    def test_pre_epoch_frame_does_not_set_session_anchor(self):
        timeline = SRTTimeline(30.0, 100.0)
        self.assertIsNone(timeline.map(10.0, 99.0))
        frame = timeline.map(11.0, 100.25)
        self.assertEqual(frame['sequence'], 7)
        self.assertEqual(frame['source_time'], .25)


class SRTConfigurationTests(unittest.TestCase):
    def test_missing_pyav_explains_optional_dependency(self):
        with mock.patch.dict(sys.modules, {'av': None}):
            with self.assertRaisesRegex(RuntimeError, 'optional PyAV dependency'):
                _open_srt('srt://127.0.0.1:9001?mode=listener', {})

    def test_srt_requires_live_dropping_and_explicit_srt_protocol(self):
        with self.assertRaisesRegex(ValueError, 'drop_stale=True'):
            CaptureSource('srt://127.0.0.1:9001', kind='srt', drop_stale=False)
        with self.assertRaisesRegex(ValueError, 'srt://'):
            CaptureSource('https://example.invalid/live', kind='srt')

    def test_nonfinite_or_nonpositive_timeout_fails_before_network_open(self):
        fake_av = SimpleNamespace(open=mock.Mock())
        with mock.patch.dict(sys.modules, {'av': fake_av}):
            for option in ('open_timeout_seconds', 'read_timeout_seconds'):
                for value in (0, -1, float('nan'), float('inf')):
                    with self.subTest(option=option, value=value):
                        with self.assertRaisesRegex(ValueError, 'positive and finite'):
                            _open_srt('srt://127.0.0.1:9001', {option: value})
        fake_av.open.assert_not_called()


class FakeFrame:
    def __init__(self, pts, decoded_at, *, width=64, height=48):
        self.time = pts
        self.pts = None if pts is None else round(pts * 90000)
        self.time_base = Fraction(1, 90000)
        self.width, self.height = width, height
        self.decoded_at = decoded_at
        self.image = np.zeros((height, width, 3), dtype=np.uint8)

    def to_ndarray(self, *, format):
        if format != 'bgr24':
            raise AssertionError('WiLoR requires BGR images')
        return self.image.copy()


class SRTWorkerTests(unittest.TestCase):
    def run_worker(self, decoded, *, rate=Fraction(30), error=None, epoch=100.0, recording_sink=None):
        clock = SimpleNamespace(now=99.0)
        control = queue.Queue()
        published = []
        stop = threading.Event()
        stream = SimpleNamespace(average_rate=rate, base_rate=rate,
                                 codec_context=SimpleNamespace(name='h264', width=64, height=48),
                                 thread_type=None)

        class Begin:
            sent = False

            def get_nowait(self):
                if clock.now < epoch or self.sent:
                    raise queue.Empty
                self.sent = True
                return epoch

        class Container:
            streams = SimpleNamespace(video=[stream])
            closed = False

            def decode(self, *args, **kwargs):
                for frame in decoded:
                    clock.now = frame.decoded_at
                    yield frame
                if error:
                    raise error

            def close(self):
                self.closed = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        container = Container()
        fake_av = SimpleNamespace(open=mock.Mock(return_value=container), __version__='test')

        def publish(image_buffer, state, ready, free, lock, frame, stop, drop_stale, dropped):
            published.append(frame.copy())
            return dropped

        failure = None
        messages = []
        try:
            with mock.patch.dict(sys.modules, {'av': fake_av}), \
                 mock.patch('airpiano.streaming.srt.time.perf_counter', side_effect=lambda: clock.now), \
                 mock.patch('airpiano.streaming.io._write_capture_frame', side_effect=publish):
                try:
                    _srt_capture_worker(
                        'srt://127.0.0.1:9001?mode=listener', 30.0,
                        {'open_timeout_seconds': 120.0, 'read_timeout_seconds': 5.0},
                        np.zeros(16), threading.Event(), threading.Event(),
                        threading.Lock(), control, Begin(), stop, recording_sink,
                    )
                except Exception as exc:
                    failure = exc
        finally:
            while not control.empty():
                message = control.get_nowait()
                messages.append(message)
                if message['kind'] == 'ready':
                    memory = shared_memory.SharedMemory(name=message['metadata']['shared_memory_name'])
                    memory.close()
                    memory.unlink()
        self.assertTrue(container.closed)
        return published, messages, failure, fake_av

    def test_pre_session_frames_are_drained_and_first_live_pts_sets_anchor(self):
        frames, messages, error, fake_av = self.run_worker([
            FakeFrame(2.0, 99.0), FakeFrame(2.0 + 1 / 30, 99.5),
            FakeFrame(3.0, 100.25), FakeFrame(3.0 + 1 / 30, 100.251),
            FakeFrame(3.0 + 4 / 30, 100.4),
        ])
        self.assertEqual([frame['sequence'] for frame in frames], [7, 8, 11])
        self.assertEqual([frame['source_pts_seconds'] for frame in frames],
                         [3.0, 3.0 + 1 / 30, 3.0 + 4 / 30])
        self.assertAlmostEqual(frames[0]['source_time'], .25)
        self.assertEqual(messages[0]['kind'], 'ready')
        self.assertEqual(messages[-1]['kind'], 'error')
        self.assertIsInstance(error, RuntimeError)
        self.assertIn('mpegts', str(fake_av.open.call_args))
        self.assertIn('(120.0, 5.0)', str(fake_av.open.call_args))

    def test_recording_receives_live_media_time_without_warmup_or_duplicates(self):
        sink = mock.Mock()
        frames, _, _, _ = self.run_worker([
            FakeFrame(2.0, 99.0), FakeFrame(2.0 + 1 / 30, 99.5),
            FakeFrame(3.0, 100.25), FakeFrame(3.0, 100.251),
            FakeFrame(3.0 + 1 / 30, 100.252), FakeFrame(3.0 + 4 / 30, 100.4),
        ], recording_sink=sink)
        self.assertEqual(sink.offer.call_count, 3)
        times = [call.args[1] for call in sink.offer.call_args_list]
        self.assertAlmostEqual(times[0], 0.25)
        self.assertAlmostEqual(times[1] - times[0], 1 / 30)
        self.assertAlmostEqual(times[2] - times[0], 4 / 30)
        self.assertEqual(times, [frame["source_time"] for frame in frames])
        sink.close.assert_called_once()

    def test_stream_resolution_change_is_reported_before_publication(self):
        frames, messages, error, _ = self.run_worker([
            FakeFrame(0.0, 99.0), FakeFrame(1.0, 100.0),
            FakeFrame(1 + 1 / 30, 100.04, width=128),
        ])
        self.assertEqual(len(frames), 1)
        self.assertRegex(str(error), '(?i)resolution|shape')
        self.assertEqual(messages[-1]['kind'], 'error')

    def test_incompatible_frame_rate_is_rejected(self):
        frames, messages, error, _ = self.run_worker([FakeFrame(0.0, 99.0)], rate=Fraction(25))
        self.assertFalse(frames)
        self.assertRegex(str(error), '30|frame rate|fps')
        self.assertEqual(messages[-1]['kind'], 'error')

    def test_read_timeout_propagates_as_capture_failure(self):
        frames, messages, error, _ = self.run_worker(
            [FakeFrame(0.0, 99.0), FakeFrame(1.0, 100.0)],
            error=TimeoutError('SRT read timed out'))
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(error, TimeoutError)
        self.assertIn('SRT read timed out', messages[-1]['traceback'])

    def test_missing_pts_and_backward_warmup_pts_fail_without_publication(self):
        cases = ([FakeFrame(None, 99.0)],
                 [FakeFrame(10.0, 99.0), FakeFrame(9.0, 99.5)])
        for decoded in cases:
            with self.subTest(pts=[frame.time for frame in decoded]):
                frames, messages, error, _ = self.run_worker(decoded)
                self.assertFalse(frames)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(messages[-1]['kind'], 'error')

    def test_duplicate_reads_are_counted_but_not_published(self):
        frames, _, _, _ = self.run_worker([
            FakeFrame(0.0, 99.0), FakeFrame(1.0, 100.0),
            FakeFrame(1.0, 100.01), FakeFrame(1.0 + 1 / 30, 100.04),
        ])
        self.assertEqual([frame['sequence'] for frame in frames], [0, 1])
        self.assertEqual(frames[-1]['read_count'], 3)
        self.assertEqual(frames[-1]['duplicate_reads_dropped'], 1)
        self.assertEqual(frames[-1]['captured_frames'], 2)

    def test_disconnect_is_not_successful_video_eof(self):
        _, messages, error, _ = self.run_worker([FakeFrame(0.0, 99.0)])
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(messages[-1]['kind'], 'error')
        self.assertNotIn('eof', [message['kind'] for message in messages])


@unittest.skipUnless(os.environ.get('AIRPIANO_TEST_SRT_INTEGRATION') == '1',
                     'set AIRPIANO_TEST_SRT_INTEGRATION=1 for localhost SRT checks')
class SRTIntegrationTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec('av') is None or shutil.which('ffmpeg') is None:
            self.skipTest('PyAV and FFmpeg with libsrt are required')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as available:
            available.bind(('127.0.0.1', 0))
            self.port = available.getsockname()[1]
        self.url = f'srt://127.0.0.1:{self.port}?mode=listener&latency=20000'

    def test_local_sender_capture_timestamps_and_cleanup(self):
        source = CaptureSource(self.url, kind='srt', srt_options={
            'open_timeout_seconds': 8.0, 'read_timeout_seconds': 2.0,
        })
        sender = None
        name = None
        try:
            source.start()
            # SRT caller connection itself waits while the spawn imports PyAV.
            sender = subprocess.Popen([
                shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error',
                '-re', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=30',
                '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-g', '15', '-bf', '0', '-f', 'mpegts',
                f'srt://127.0.0.1:{self.port}?mode=caller&latency=20000&connect_timeout=5000',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            metadata = source.wait_ready(timeout=10.0)
            name = metadata['shared_memory_name']
            self.assertEqual(metadata['kind'], 'srt')
            self.assertEqual(metadata['frame_shape'], [48, 64, 3])
            # Allow continuous warmup draining before opening the session clock.
            time.sleep(.15)
            epoch = time.perf_counter() + .05
            source.begin(epoch)
            frames = []
            deadline = time.monotonic() + 4.0
            while len(frames) < 12 and time.monotonic() < deadline:
                frame = source.poll(timeout=.05)
                if frame is not None:
                    self.assertEqual(frame['kind'], 'frame')
                    frames.append(frame)
            self.assertGreaterEqual(len(frames), 12)
            self.assertGreaterEqual(frames[0]['acquired_at'], epoch)
            for previous, current in zip(frames, frames[1:]):
                self.assertGreater(current['sequence'], previous['sequence'])
                self.assertAlmostEqual(
                    current['source_time'] - previous['source_time'],
                    current['source_pts_seconds'] - previous['source_pts_seconds'], places=5)
                self.assertEqual(current['scheduled_at'], current['acquired_at'])
            self.assertEqual(frames[-1]['image'].shape, (48, 64, 3))
        finally:
            try:
                source.close()
            finally:
                if sender is not None:
                    sender.terminate()
                    try:
                        sender.communicate(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        sender.kill()
                        sender.communicate(timeout=3.0)
        self.assertFalse(source.process.is_alive())
        if name:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_close_interrupts_listener_waiting_for_phone(self):
        source = CaptureSource(self.url, kind='srt', srt_options={
            'open_timeout_seconds': 120.0, 'read_timeout_seconds': 5.0,
        })
        source.start()
        time.sleep(.3)
        started = time.monotonic()
        source.close()
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertFalse(source.process.is_alive())
        source.close()


if __name__ == '__main__':
    unittest.main()
