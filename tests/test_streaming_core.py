"""Frame-clock, missing-observation, and concurrent release regression tests."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event as ThreadEvent
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from airpiano import data
from airpiano.data.hands import OffsetScaleTransform
from airpiano.full_song_real_time import Event
from airpiano.streaming.core import AirPianoStepper, CanonicalFrameAdapter
from scripts.streaming_latency_components import AirPianoStepper as NumpyStepper
from scripts.benchmark_realtime_latency import instrument_inference


def components(delay=0.0):
    return {
        "tokenizer": data.ChordTokenizer(),
        "slicer": data.FutureChordSlicer(segment_frames=50),
        "target_generator": data.TargetGenerator(
            data.ChordOnsetEvents(delay), data.TriangularSoftener(0.2)
        ),
        "hand_loader": SimpleNamespace(
            transform=OffsetScaleTransform([0.1, 0.2, 0.3], [2.0, 3.0, 4.0])
        ),
    }


def chords():
    note = data.Note(0, 1.0, 2.0, 60, 80)
    return (data.Chord(0, (note,), 1.0, 24, 1.0, 24),)


class RecordingModel(torch.nn.Module):
    def __init__(self, trigger=False):
        super().__init__()
        self.calls = []
        self.trigger = trigger

    def forward(self, **batch):
        self.calls.append({name: value.clone() for name, value in batch.items()})
        return torch.full(
            (1, batch["hands"].shape[1], batch["chords"].shape[1]),
            2.0 if self.trigger else -2.0,
        )


class StreamingCoreTest(unittest.TestCase):
    def test_logical_gaps_keep_positions_without_running_synthetic_frames(self):
        model = RecordingModel()
        stepper = AirPianoStepper(model, chords(), components(), torch.device("cpu"))
        for index in (0, 1, 4, 5):
            result = stepper.step(
                torch.full((2, 21, 3), float(index + 1)), torch.ones(2, dtype=torch.bool),
                frame_index=index, release=False,
            )
        self.assertEqual([call["hands"].shape[1] for call in model.calls], [1, 2, 5, 6])
        batch = model.calls[-1]
        torch.testing.assert_close(batch["hands"][0, 2:4], torch.zeros(2, 2, 21, 3))
        self.assertFalse(batch["hand_presence"][0, 2:4].any())
        self.assertTrue(batch["frame_mask"].all())
        self.assertEqual(stepper.last_frame_index, 5)
        self.assertAlmostEqual(stepper.last_source_time, 5 / 30)
        self.assertEqual(result["history_frames"], 6)
        self.assertEqual(stepper.controller.event_records, [])
        with self.assertRaisesRegex(ValueError, "indices must increase"):
            stepper.step(torch.zeros(2, 21, 3), torch.zeros(2, dtype=torch.bool), frame_index=4)
        self.assertEqual(len(stepper.video_frames), 6)
        self.assertEqual(len(model.calls), 4)

    def test_history_rewind_retains_original_cpu_frames(self):
        model = RecordingModel()
        stepper = AirPianoStepper(model, chords(), components(), torch.device("cpu"))
        for index in range(2):
            stepper.step(torch.full((2, 21, 3), float(index)), torch.ones(2, dtype=torch.bool))
        stepper.controller.video_history_start = 1
        stepper.step(torch.full((2, 21, 3), 2.0), torch.ones(2, dtype=torch.bool))
        stepper.controller.video_history_start = 0
        stepper.step(torch.full((2, 21, 3), 3.0), torch.ones(2, dtype=torch.bool))
        self.assertEqual(model.calls[-2]["hands"][0, :, 0, 0, 0].tolist(), [1.0, 2.0])
        self.assertEqual(model.calls[-1]["hands"][0, :, 0, 0, 0].tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_benchmark_wrapper_transforms_once_and_preserves_target_delay(self):
        raw = np.ones((2, 21, 3), dtype=np.float32)
        present = np.array([True, False])
        for delay in (0.0, -0.1):
            with self.subTest(delay=delay):
                config = components(delay)
                transformed = config["hand_loader"].transform(
                    torch.from_numpy(raw), torch.from_numpy(present)
                )
                tensor_model, numpy_model = RecordingModel(True), RecordingModel(True)
                direct = AirPianoStepper(tensor_model, chords(), config, torch.device("cpu"))
                wrapper = NumpyStepper(numpy_model, chords(), config, torch.device("cpu"))
                for _ in range(9):
                    direct.step(transformed, torch.from_numpy(present))
                    wrapper.step(raw, present)
                torch.testing.assert_close(tensor_model.calls[0]["hands"][0, 0], transformed)
                for actual, expected in zip(numpy_model.calls, tensor_model.calls):
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key])
                self.assertAlmostEqual(direct.controller.event_records[0]["time"], 0.1 - delay)
                self.assertEqual(wrapper.controller.event_records, direct.controller.event_records)
                self.assertEqual(wrapper.controller.release_records, direct.controller.release_records)

    def test_live_clock_releases_without_new_frames_or_simulated_offset(self):
        stepper = AirPianoStepper(
            RecordingModel(True), chords(), components(-0.1), torch.device("cpu")
        )
        stepper.step(
            torch.zeros(2, 21, 3), torch.ones(2, dtype=torch.bool),
            frame_index=4, source_time=0.137, release=False,
        )
        controller = stepper.controller
        self.assertAlmostEqual(controller.event_records[0]["time"], 0.337)
        self.assertEqual(controller.release_records, [])
        self.assertEqual(controller.release_due(4, now=0.336, frame_time=0.137), [])
        release, = controller.release_due(4, now=0.338, frame_time=0.137)
        self.assertEqual(release["release_time"], 0.338)
        self.assertEqual(release["frame_time"], 0.137)
        self.assertAlmostEqual(release["predicted_onset"], 0.337)
        self.assertEqual(controller.release_due(4, now=1.0, frame_time=0.137), [])

    def test_scheduler_can_release_while_model_forward_is_blocked(self):
        entered = ThreadEvent()
        proceed = ThreadEvent()

        class BlockingModel(RecordingModel):
            def forward(self, **batch):
                entered.set()
                if not proceed.wait(5):
                    raise TimeoutError("test model was never released")
                return super().forward(**batch)

        stepper = AirPianoStepper(BlockingModel(), chords(), components(), torch.device("cpu"))
        stepper.controller.process_event(Event(1, 0.1, 0.9), 0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            inference = pool.submit(
                stepper.step, torch.zeros(2, 21, 3), torch.ones(2, dtype=torch.bool), release=False
            )
            try:
                self.assertTrue(entered.wait(2))
                release = pool.submit(stepper.controller.release_due, 0, now=0.15, frame_time=0.0)
                self.assertEqual(len(release.result(timeout=2)), 1)
            finally:
                proceed.set()
            inference.result(timeout=2)

    def test_benchmark_instrumentation_times_each_shared_step(self):
        config = components()
        score = chords()
        song = data.Song(
            "benchmark", tuple(chord.onset_notes[0] for chord in score), score,
            torch.zeros(3, 2, 21, 3), torch.ones(3, 2, dtype=torch.bool), 30.0,
        )
        infer, namespace = instrument_inference()
        with tempfile.TemporaryDirectory() as directory:
            infer(
                RecordingModel(), song, config["slicer"], config["target_generator"],
                torch.device("cpu"), Path(directory) / "timed.npz",
                tokenizer=config["tokenizer"], fps=30.0, bf16=False,
            )
        self.assertEqual([row[0] for row in namespace["_rows"]], [0, 1, 2])
        self.assertEqual([row[2] for row in namespace["_rows"]], [1, 2, 3])
        self.assertTrue(all(row[1] >= 0 for row in namespace["_rows"]))

    def test_canonical_wrist_delta_restarts_after_a_dropped_interval(self):
        adapter = CanonicalFrameAdapter()
        deltas = []
        for index, wrist_x in ((0, 1.0), (1, 3.0), (4, 9.0), (5, 12.0)):
            arrays = {
                "frame_detection_offsets": np.array([0, 1]),
                "detector_confidence": np.array([0.9]),
                "is_right": np.array([0]),
                "pred_keypoints_3d": np.zeros((1, 21, 3), dtype=np.float32),
                "pred_cam_t_full": np.array([[wrist_x, 0.0, 1.0]], dtype=np.float32),
                "pred_keypoints_2d_full": np.zeros((1, 21, 2), dtype=np.float32),
            }
            pose, presence = adapter.step(arrays, frame_index=index)
            self.assertEqual(presence.tolist(), [True, False])
            deltas.append(float(pose[0, 0, 0]))
        self.assertEqual(deltas, [0.0, 2.0, 0.0, 3.0])
        with self.assertRaisesRegex(ValueError, "indices must increase"):
            adapter.step(arrays, frame_index=4)
        self.assertEqual(adapter.last_frame_index, 5)


if __name__ == "__main__":
    unittest.main()
