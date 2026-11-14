"""CPU-only checks for capture ordering, bounded transport and worker failures."""

import tempfile
import queue
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from airpiano.streaming.io import (
    CaptureSource, WiLoRWorkerPool, _capture_worker, _close_processes, _warm_pose_shapes,
)


class CaptureSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.video = Path(self.directory.name) / "source.avi"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        self.count = 12
        for index in range(self.count):
            writer.write(np.full((48, 64, 3), index * 15, dtype=np.uint8))
        writer.release()
        reader = cv2.VideoCapture(str(self.video))
        self.expected_images = [reader.read()[1] for _ in range(self.count)]
        reader.release()

    def tearDown(self):
        self.directory.cleanup()

    def collect(self, source):
        frames = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            message = source.poll(timeout=0.05)
            if message is None:
                continue
            if message["kind"] == "eof":
                return frames, message
            frames.append(message)
        self.fail("Capture did not deliver EOF")

    def test_lossless_backpressure_preserves_schedule_and_final_frame(self):
        source = CaptureSource(str(self.video), capacity=1, drop_stale=False)
        try:
            source.start()
            metadata = source.wait_ready()
            self.assertEqual(metadata["frame_shape"], [48, 64, 3])
            self.assertEqual(metadata["frame_count"], self.count)
            self.assertEqual(metadata["example_image"].shape, (48, 64, 3))
            memory_name = metadata["shared_memory_name"]
            self.assertEqual(source.memory.name, memory_name)
            epoch = time.perf_counter() + 0.03
            source.begin(epoch)
            time.sleep(0.2)
            frames, eof = self.collect(source)
            self.assertEqual([f["sequence"] for f in frames], list(range(self.count)))
            self.assertEqual(eof["last_sequence"], self.count - 1)
            self.assertEqual(eof["dropped"], 0)
            for frame in frames:
                self.assertAlmostEqual(frame["source_time"], frame["sequence"] / 30)
                np.testing.assert_array_equal(
                    frame["image"], self.expected_images[frame["sequence"]]
                )
                self.assertAlmostEqual(frame["scheduled_at"], epoch + frame["sequence"] / 30)
            self.assertIsNone(source.poll())
        finally:
            source.close()
        self.assertFalse(source.process.is_alive())
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=memory_name)

    def test_latest_mailbox_drops_old_frames_but_eof_follows_last(self):
        source = CaptureSource(str(self.video), capacity=1, drop_stale=True)
        try:
            source.start()
            source.wait_ready()
            source.begin(time.perf_counter() + 0.02)
            time.sleep(0.65)
            frames, eof = self.collect(source)
            self.assertEqual(frames[-1]["sequence"], self.count - 1)
            self.assertLess(len(frames), self.count)
            self.assertEqual(eof["frames"], self.count)
            self.assertEqual(eof["dropped"], self.count - len(frames))
            self.assertEqual(frames[-1]["dropped_total"], eof["dropped"])
        finally:
            source.close()

    def test_camera_arrivals_keep_dense_ids_and_actual_timestamps(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        fake = mock.Mock()
        fake.isOpened.return_value = True
        fake.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48}[prop]
        fake.read.side_effect = [(True, image)] * 3 + [(False, None)]
        frames, control, begin = [], queue.Queue(), queue.Queue()
        epoch = 100.0
        begin.put(epoch)
        def record_frame(image_buffer, state, ready, free, lock, frame,
                         stop, drop_stale, dropped):
            frames.append(frame.copy())
            return dropped
        # Nearby reads and a later stall keep their IDs and actual timestamps.
        try:
            with mock.patch("airpiano.streaming.io.cv2.VideoCapture", return_value=fake), \
                 mock.patch("airpiano.streaming.io._write_capture_frame", side_effect=record_frame), \
                 mock.patch("airpiano.streaming.io.time.perf_counter", side_effect=[
                     epoch, epoch, epoch + .001, epoch + .012, epoch + .110, epoch + .111, epoch + .112]):
                with self.assertRaisesRegex(RuntimeError, "Camera read failed"):
                    _capture_worker(0, "camera", 30.0, 64, 48, True,
                                    None, None, None, None, control, begin, threading.Event())
            self.assertEqual([frame["sequence"] for frame in frames], [0, 1, 2])
            for frame, source_time in zip(frames, (.001, .012, .110)):
                self.assertAlmostEqual(frame["source_time"], source_time)
                self.assertAlmostEqual(frame["acquired_at"], epoch + source_time)
                self.assertEqual(frame["duplicate_reads_dropped"], 0)
            self.assertEqual(frames[-1]["read_count"], 3)
            self.assertEqual(frames[-1]["captured_frames"], 3)
            fake.release.assert_called_once()
        finally:
            metadata = control.get_nowait()["metadata"]
            memory = shared_memory.SharedMemory(name=metadata["shared_memory_name"])
            memory.close()
            memory.unlink()

    def test_camera_rejects_a_known_incompatible_frame_rate(self):
        fake = mock.Mock()
        fake.isOpened.return_value = True
        fake.get.return_value = 25.0
        with mock.patch("airpiano.streaming.io.cv2.VideoCapture", return_value=fake):
            with self.assertRaisesRegex(ValueError, "approximately 30 Hz, got 25"):
                _capture_worker(0, "camera", 30.0, 64, 48, True,
                                None, None, None, None, queue.Queue(), queue.Queue(),
                                threading.Event())
        fake.read.assert_not_called()
        fake.release.assert_called_once()

    def test_close_interrupts_wait_for_begin(self):
        source = CaptureSource(str(self.video))
        source.start()
        source.wait_ready()
        started = time.monotonic()
        source.close()
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertFalse(source.process.is_alive())
        source.close()


class PoseWarmupTests(unittest.TestCase):
    def test_warmup_matches_production_tensor_guards_and_crop_layout(self):
        observations = []

        class Pose(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))

            def forward(self, batch):
                image = batch["img"]
                # WiLoR passes this exact view, including its margins, to the
                # compiled backbone. Both tensor kind and strides are guarded.
                cropped = image[:, :, :, 32:-32]
                observations.append({
                    "inference_tensor": torch.is_inference(image),
                    "cropped_inference_tensor": torch.is_inference(cropped),
                    "inference_enabled": torch.is_inference_mode_enabled(),
                    "shape": tuple(image.shape), "dtype": image.dtype,
                    "contiguous": image.is_contiguous(),
                    "cropped_shape": tuple(cropped.shape),
                    "cropped_stride": cropped.stride(),
                })

        config = SimpleNamespace(MODEL=SimpleNamespace(IMAGE_SIZE=256))
        _warm_pose_shapes(Pose().eval(), config, torch.device("cpu"), (1, 2, 3))
        self.assertEqual(len(observations), 3)
        for count, record in enumerate(observations, 1):
            self.assertFalse(record["inference_tensor"])
            self.assertFalse(record["cropped_inference_tensor"])
            self.assertTrue(record["inference_enabled"])
            self.assertTrue(record["contiguous"])
            self.assertEqual(record["dtype"], torch.float16)
            self.assertEqual(record["shape"], (count, 3, 256, 256))
            self.assertEqual(record["cropped_shape"], (count, 3, 256, 192))
            self.assertEqual(record["cropped_stride"], (196608, 65536, 256, 1))


class WorkerPoolTests(unittest.TestCase):
    def test_slots_are_not_reused_until_the_result_is_consumed(self):
        pool = WiLoRWorkerPool(["cuda:1"], capacity=2, frame_shape=(4, 4, 3))
        # Exercise the real queue/slot protocol without loading CUDA models.
        pool.frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        try:
            self.assertTrue(pool.submit(4, frame, 4 / 30))
            self.assertTrue(pool.submit(5, frame * 2, 5 / 30))
            self.assertFalse(pool.submit(6, frame, 6 / 30))
            first = pool.jobs.get(timeout=1)
            second = pool.jobs.get(timeout=1)
            self.assertNotEqual(first[1], second[1])
            np.testing.assert_array_equal(pool.frames[first[1]], frame)
            pool.results.put({"sequence": first[0], "slot": first[1],
                              "device": "cuda:1", "arrays": {}})
            result = pool.poll(timeout=1)
            self.assertEqual(result["sequence"], 4)
            self.assertNotIn("slot", result)
            self.assertTrue(pool.submit(6, frame, 6 / 30))
        finally:
            pool.close()

    def test_shutdown_terminates_a_process_that_cannot_observe_stop(self):
        # Model a driver blocked in native read(): it cannot poll a stop Event.
        process = mp.get_context("spawn").Process(target=time.sleep, args=(60,))
        process.start()
        started = time.monotonic()
        _close_processes([process], [], grace_seconds=0.05)
        self.assertFalse(process.is_alive())
        self.assertLess(time.monotonic() - started, 3.0)

    def test_unexpected_success_exit_is_a_failure(self):
        pool = WiLoRWorkerPool(["cuda:1"])
        pool.workers = [SimpleNamespace(name="test-worker", exitcode=0)]
        try:
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly: 0"):
                pool.healthcheck()
        finally:
            pool.workers = []
            pool.close()


if __name__ == "__main__":
    unittest.main()
