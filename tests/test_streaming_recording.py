"""CFR timeline, bounded mailbox, and CPU FFmpeg recording checks."""

import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from airpiano.streaming.io import CaptureSource
from airpiano.streaming.recording import VideoRecorder, _CFRResampler


class RecordingTimelineTests(unittest.TestCase):
    def test_leading_black_gaps_and_midi_tail_preserve_timeline(self):
        frames = []
        timeline = _CFRResampler(30, lambda image: frames.append(image.copy()))
        first = np.full((4, 6, 3), 50, dtype=np.uint8)
        second = np.full((4, 6, 3), 150, dtype=np.uint8)
        timeline.push(first, .1)
        timeline.push(second, .2)
        timeline.finish(.4)
        self.assertEqual([int(frame.mean()) for frame in frames], [0] * 3 + [50] * 3 + [150] * 6)
        self.assertEqual(timeline.next_index, 12)
        self.assertEqual(timeline.leading_black_frames, 3)
        self.assertEqual(timeline.repeated_frames, 7)
        self.assertEqual(timeline.source_frames, 2)

    def test_nearest_grid_and_old_images_do_not_rewrite_committed_frames(self):
        frames = []
        timeline = _CFRResampler(30, lambda image: frames.append(image.copy()))
        image = np.full((4, 6, 3), 50, dtype=np.uint8)
        timeline.push(image, .049)
        timeline.push(image + 20, .05)
        timeline.push(image + 40, .01)
        self.assertEqual(timeline.next_index, 3)
        self.assertEqual(timeline.late_frames, 1)
        self.assertEqual([int(frame.mean()) for frame in frames], [0, 50, 70])

    def test_mailbox_overwrites_and_never_waits_for_encoder_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(Path(directory) / "performance.mp4", max_dimension=64)
            name = recorder.memory.name
            try:
                image = np.full((108, 192, 3), 90, dtype=np.uint8)
                self.assertTrue(recorder.sink.offer(image, 0))
                self.assertTrue(recorder.sink.offer(image, 1 / 30))
                self.assertEqual(recorder.sink.state[1:3], [36.0, 64.0])
                with recorder.sink.lock:
                    started = time.monotonic()
                    self.assertFalse(recorder.sink.offer(image, 2 / 30))
                    self.assertLess(time.monotonic() - started, .2)
                recorder.stop_accepting()
                self.assertFalse(recorder.sink.offer(image, .1))
                report = recorder.finish(.2)
                self.assertEqual(report['offered_frames'], 3)
                self.assertEqual(report['overwritten_frames'], 1)
                self.assertEqual(report['lock_skipped_frames'], 1)
                self.assertEqual(report['offer_cost_ms']['count'], 3)
                self.assertEqual(recorder.finish(.3), report)
            finally:
                recorder.finish(0)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_bad_offer_disables_only_optional_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(Path(directory) / "performance.mp4")
            try:
                self.assertFalse(recorder.sink.offer(np.zeros((4, 4, 3), dtype=np.uint8), float('nan')))
                self.assertFalse(recorder.sink.enabled.is_set())
                self.assertEqual(recorder.healthcheck()['status'], 'error')
                report = recorder.finish(0)
                self.assertEqual(report['invalid_frames'], 1)
                self.assertIn('source_time', report['error'])
            finally:
                recorder.finish(0)


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for recorder smoke checks')
class RecordingEncoderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def wait_consumed(self, recorder):
        deadline = time.monotonic() + 5
        while recorder.sink.ready.is_set():
            self.assertNotEqual(recorder.healthcheck()['status'], 'error')
            if time.monotonic() > deadline:
                self.fail('Recorder did not consume the image')
            time.sleep(.01)

    def decode(self, path):
        reader = cv2.VideoCapture(str(path))
        try:
            self.assertTrue(reader.isOpened())
            self.assertAlmostEqual(reader.get(cv2.CAP_PROP_FPS), 30, places=3)
            frames = []
            while True:
                ok, frame = reader.read()
                if not ok:
                    return frames
                frames.append(frame)
        finally:
            reader.release()

    def test_real_h264_has_zero_start_leading_black_gaps_and_end_tail(self):
        output = self.root / 'performance.mp4'
        recorder = VideoRecorder(output, max_dimension=64)
        recorder.start()
        try:
            image = np.full((48, 64, 3), 50, dtype=np.uint8)
            self.assertTrue(recorder.sink.offer(image, .1))
            self.wait_consumed(recorder)
            self.assertTrue(recorder.sink.offer(image + 100, .2))
            self.wait_consumed(recorder)
            report = recorder.finish(.4)
        finally:
            recorder.finish(0)
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual(report['output_frames'], 12)
        self.assertEqual(report['repeated_frames'], 7)
        self.assertEqual(report['leading_black_frames'], 3)
        self.assertEqual((report['width'], report['height']), (64, 48))
        frames = self.decode(output)
        self.assertEqual(len(frames), 12)
        self.assertTrue(all(frame.mean() < 5 for frame in frames[:3]))
        self.assertTrue(all(abs(frame.mean() - 50) < 8 for frame in frames[3:6]))
        self.assertTrue(all(abs(frame.mean() - 150) < 8 for frame in frames[6:]))
        if shutil.which('ffprobe'):
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output),
            ]))
            self.assertEqual(len(probe['streams']), 1)
            self.assertEqual(probe['streams'][0]['codec_name'], 'h264')
            self.assertAlmostEqual(float(probe['streams'][0]['start_time']), 0, places=6)
            self.assertAlmostEqual(float(probe['format']['duration']), .4, places=3)

    def test_capture_recording_does_not_depend_on_inference_mailbox_consumption(self):
        source_path = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 15, dtype=np.uint8))
        writer.release()
        recorder = VideoRecorder(self.root / 'performance.mp4', max_dimension=64)
        recorder.start()
        capture = CaptureSource(str(source_path), kind='video', recording_sink=recorder.sink)
        try:
            capture.start()
            capture.wait_ready()
            capture.begin(time.perf_counter() + .1)
            time.sleep(.7)
            frames = []
            while True:
                message = capture.poll(timeout=.1)
                if message is None:
                    self.fail('Capture failed to complete its short video')
                if message['kind'] == 'eof':
                    break
                frames.append(message)
            self.assertLess(len(frames), 12)
            recorder.stop_accepting()
            capture.close()
            report = recorder.finish(.4)
        finally:
            recorder.stop_accepting()
            capture.close()
            recorder.finish(0)
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual(report['offered_frames'], 12)
        self.assertEqual(report['output_frames'], 12)
        self.assertEqual(len(self.decode(self.root / 'performance.mp4')), 12)

    def test_no_input_does_not_create_fake_video(self):
        output = self.root / 'performance.mp4'
        recorder = VideoRecorder(output)
        recorder.start()
        report = recorder.finish(1)
        self.assertEqual(report['status'], 'no_frames', report)
        self.assertEqual(report['output_frames'], 0)
        self.assertFalse(output.exists())
        self.assertFalse(recorder.process.is_alive())

    def test_invalid_output_does_not_raise_or_leave_encoder_alive(self):
        recorder = VideoRecorder(self.root / 'missing' / 'performance.mp4', max_dimension=64)
        recorder.start()
        recorder.sink.offer(np.zeros((48, 64, 3), dtype=np.uint8), 0)
        report = recorder.finish(.1)
        self.assertEqual(report['status'], 'error', report)
        self.assertFalse(recorder.sink.enabled.is_set())
        self.assertFalse(recorder.process.is_alive())
        self.assertIn('FFmpeg', report['error'])

    def test_dead_worker_does_not_leave_its_encoder_running(self):
        fake_directory = self.root / 'bin'
        fake_directory.mkdir()
        executable = fake_directory / 'ffmpeg'
        executable.write_text('#!/usr/bin/python3\nimport time\ntime.sleep(60)\n')
        executable.chmod(0o755)
        with mock.patch.dict(os.environ, {'PATH': str(fake_directory) + os.pathsep + os.environ['PATH']}):
            recorder = VideoRecorder(self.root / 'performance.mp4', max_dimension=64)
            recorder.start()
            try:
                recorder.sink.offer(np.zeros((48, 64, 3), dtype=np.uint8), 0)
                deadline = time.monotonic() + 5
                while not recorder.sink.encoder_pid.value and time.monotonic() < deadline:
                    time.sleep(.01)
                pid = recorder.sink.encoder_pid.value
                self.assertGreater(pid, 0)
                recorder.process.terminate()
                recorder.process.join(timeout=2)
                self.assertFalse(recorder.process.is_alive())
                report = recorder.finish(.1, timeout=.1)
                self.assertEqual(report['status'], 'error')
                # A reparented process may remain a zombie briefly; it must not run.
                stat = Path(f'/proc/{pid}/stat')
                deadline = time.monotonic() + 2
                while stat.exists() and time.monotonic() < deadline:
                    try:
                        state = stat.read_text().rpartition(')')[2].split()[0]
                    except FileNotFoundError:
                        break
                    if state == 'Z':
                        break
                    time.sleep(.01)
                else:
                    if stat.exists():
                        self.fail('Orphan encoder was still running after recorder cleanup')
            finally:
                recorder.finish(0, timeout=.1)

    def test_blocked_encoder_shutdown_is_bounded(self):
        fake_directory = self.root / 'bin'
        fake_directory.mkdir()
        executable = fake_directory / 'ffmpeg'
        executable.write_text('#!/usr/bin/python3\nimport time\ntime.sleep(60)\n')
        executable.chmod(0o755)
        with mock.patch.dict(os.environ, {'PATH': str(fake_directory) + os.pathsep + os.environ['PATH']}):
            recorder = VideoRecorder(self.root / 'performance.mp4', max_dimension=64)
            recorder.start()
            try:
                recorder.sink.offer(np.zeros((48, 64, 3), dtype=np.uint8), 0)
                self.wait_consumed(recorder)
                recorder.sink.offer(np.ones((48, 64, 3), dtype=np.uint8), 10)
                self.wait_consumed(recorder)
                started = time.monotonic()
                report = recorder.finish(10.1, timeout=.1)
                self.assertLess(time.monotonic() - started, 4)
                self.assertEqual(report['status'], 'error')
                self.assertIn('timed out', report['error'])
                self.assertFalse(recorder.process.is_alive())
            finally:
                recorder.finish(0, timeout=.1)


if __name__ == '__main__':
    unittest.main()
