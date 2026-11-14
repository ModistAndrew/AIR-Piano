"""CPU-only live-clock and mocked end-to-end orchestration regressions."""

from contextlib import ExitStack
import csv
import json
from pathlib import Path
import tempfile
from threading import Event as ThreadEvent
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import torch

from airpiano.full_song_real_time import Event, RealTimeController
import airpiano.streaming.runtime as runtime
from test_streaming_core import RecordingModel, chords, components


def empty_arrays():
    return {
        "frame_detection_offsets": np.array([0, 0]),
        "detector_confidence": np.empty(0),
        "is_right": np.empty(0, dtype=np.uint8),
        "pred_keypoints_3d": np.empty((0, 21, 3), dtype=np.float32),
        "pred_cam_t_full": np.empty((0, 3), dtype=np.float32),
        "pred_keypoints_2d_full": np.empty((0, 21, 2), dtype=np.float32),
    }


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class Capture:
    def __init__(self, clock, indices):
        self.clock = clock
        self.indices = iter(indices)
        self.count = max(indices, default=-1) + 1
        self.closed = False
        self.finished = False
        self.epoch = None

    def start(self):
        pass

    def wait_ready(self, **kwargs):
        return {"frame_shape": [4, 4, 3], "frame_count": self.count,
                "fps": 30.0, "example_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def begin(self, epoch):
        self.epoch = epoch

    def healthcheck(self):
        pass

    def poll(self, **kwargs):
        if self.finished:
            return None
        index = next(self.indices, None)
        if index is None:
            self.finished = True
            self.clock.value = max(self.clock.value, self.epoch + self.count / 30)
            return {"kind": "eof", "last_sequence": self.count - 1, "frames": self.count}
        scheduled = self.epoch + index / 30
        self.clock.value = max(self.clock.value, scheduled)
        return {"kind": "frame", "sequence": index, "source_time": index / 30,
                "scheduled_at": scheduled, "acquired_at": scheduled,
                "image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def close(self):
        self.closed = True


class Pool:
    def __init__(self, clock, delays=None, fail=False):
        self.clock = clock
        self.delays = delays or {}
        self.fail = fail
        self.pending = {}
        self.submitted = []
        self.arrival_order = []
        self.closed = False

    def start(self, **kwargs):
        pass

    def wait_ready(self, **kwargs):
        return {"workers": {"cpu": {}}, "arrays": empty_arrays()}

    def healthcheck(self):
        if self.fail and self.submitted:
            raise RuntimeError("worker failed")

    def submit(self, sequence, image, source_time):
        self.submitted.append(sequence)
        started = self.clock()
        self.pending[sequence] = {
            "sequence": sequence, "device": "cpu", "started_at": started,
            "finished_at": started + self.delays.get(sequence, 0.0),
            "arrays": empty_arrays(),
        }
        return True

    def poll(self, **kwargs):
        ready = [packet for packet in self.pending.values()
                 if packet["finished_at"] <= self.clock()]
        if not ready:
            return None
        packet = min(ready, key=lambda item: (item["finished_at"], item["sequence"]))
        self.arrival_order.append(packet["sequence"])
        return self.pending.pop(packet["sequence"])

    def close(self):
        self.closed = True


class IdleScheduler:
    """No thread is needed in scenarios whose model never triggers notes."""

    def __init__(self, *args, **kwargs):
        self.wake = ThreadEvent()

    def start(self):
        pass

    def check(self):
        pass

    def close(self, **kwargs):
        pass


class ReleaseSchedulerTest(unittest.TestCase):
    def make_stepper(self):
        controller = RealTimeController(chords(), target_delay_seconds=-0.1)
        event, = controller.collect_events([0.9], 1, 0, source_time=0.013)
        controller.process_event(event, 0)
        return SimpleNamespace(controller=controller, last_frame_index=0, last_source_time=0.013)

    def test_clock_boundary_has_no_simulated_latency_and_does_not_need_frames(self):
        clock = Clock()
        emitted = []
        stepper = self.make_stepper()
        scheduler = runtime.ReleaseScheduler(stepper, 100.0, emitted.append, clock=clock)
        clock.value = 100.212
        self.assertEqual(scheduler.release_once(), [])
        clock.value = 100.214
        release, = scheduler.release_once()
        self.assertAlmostEqual(release["release_time"], 0.214)
        self.assertAlmostEqual(release["predicted_onset"], 0.213)
        self.assertEqual(release["frame_index"], 0)
        self.assertEqual(release["frame_time"], 0.013)
        clock.value = 101.0
        self.assertEqual(scheduler.release_once(), [])
        self.assertEqual(len(emitted), 1)

    def test_scheduler_thread_emits_without_any_new_inference(self):
        clock = Clock()
        emitted = ThreadEvent()
        scheduler = runtime.ReleaseScheduler(
            self.make_stepper(), 100.0, lambda record: emitted.set(), clock=clock
        )
        scheduler.start()
        try:
            self.assertFalse(emitted.is_set())
            clock.value = 100.25
            scheduler.wake.set()
            self.assertTrue(emitted.wait(2))
        finally:
            scheduler.close()
        self.assertFalse(scheduler.thread.is_alive())

    def test_scheduler_failure_is_reported_to_main_thread(self):
        clock = Clock()
        clock.value = 101.0
        failure = OSError("event log unavailable")
        scheduler = runtime.ReleaseScheduler(
            self.make_stepper(), 100.0, mock.Mock(side_effect=failure), clock=clock
        )
        scheduler.start()
        self.assertTrue(scheduler.stopped.wait(2))
        with self.assertRaisesRegex(RuntimeError, "scheduler failed") as raised:
            scheduler.close()
        self.assertIs(raised.exception.__cause__, failure)



class StartupAndLoggingTests(unittest.TestCase):
    def test_countdown_arms_capture_before_waiting_and_uses_one_epoch(self):
        clock = Clock()
        capture, pool = mock.Mock(), mock.Mock()
        counts = []

        def output(message, **kwargs):
            if "starting in" in message:
                capture.begin.assert_called_once_with(105.0)
                counts.append(message)

        with mock.patch.object(runtime.time, "perf_counter", clock), \
             mock.patch.object(runtime.time, "sleep", clock.sleep), \
             mock.patch("builtins.print", side_effect=output):
            epoch = runtime.prepare_start(capture, pool, 5)
        self.assertEqual(epoch, 105.0)
        self.assertAlmostEqual(clock(), epoch)
        self.assertEqual(counts, [f"[streaming] starting in {i}..." for i in (5, 4, 3, 2, 1)])
        self.assertGreater(capture.healthcheck.call_count, 5)
        self.assertGreater(pool.healthcheck.call_count, 5)

    def test_zero_countdown_retains_brief_camera_drain_without_counting(self):
        clock = Clock()
        capture, pool = mock.Mock(), mock.Mock()
        with mock.patch.object(runtime.time, "perf_counter", clock), \
             mock.patch.object(runtime.time, "sleep", clock.sleep), \
             mock.patch("builtins.print") as output:
            epoch = runtime.prepare_start(capture, pool, 0)
        self.assertAlmostEqual(epoch, 100.2)
        self.assertAlmostEqual(clock(), epoch)
        self.assertFalse(any("starting in" in call.args[0] for call in output.call_args_list))

    def test_countdown_detects_failed_capture_before_start(self):
        capture, pool = mock.Mock(), mock.Mock()
        capture.healthcheck.side_effect = RuntimeError("SRT disconnected during preparation")
        with mock.patch("builtins.print"), mock.patch.object(runtime.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "during preparation"):
                runtime.prepare_start(capture, pool, 5)
        sleep.assert_not_called()

    def test_console_trigger_and_release_match_persisted_events(self):
        trigger = {"chord_id": 3, "pitches": [60, 64, 67], "time": 0.15, "score": 0.875}
        release = {"chord_id": 3, "pitches": [60, 64, 67], "predicted_onset": 0.15,
                   "release_time": 0.152, "duration": 1.0}
        for enabled in (True, False):
            with self.subTest(console=enabled), tempfile.TemporaryDirectory() as directory:
                log = runtime.SessionLog(Path(directory), log_midi_events=enabled)
                try:
                    with mock.patch("builtins.print") as output:
                        log.event("trigger", trigger)
                        log.event("release", release)
                    records = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
                    self.assertEqual(records, [{"kind": "trigger", **trigger}, {"kind": "release", **release}])
                    if enabled:
                        self.assertEqual(output.call_count, 2)
                        self.assertIn("[midi trigger] chord=3 pitches=[60, 64, 67]", output.call_args_list[0].args[0])
                        self.assertIn("confidence=0.875", output.call_args_list[0].args[0])
                        self.assertIn("released=0.152s duration=1.000s", output.call_args_list[1].args[0])
                        self.assertTrue(all(call.kwargs["flush"] for call in output.call_args_list))
                    else:
                        output.assert_not_called()
                finally:
                    log.close()


class StreamingRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/default.yaml")
        self.cfg.streaming.source.kind = "video"
        self.cfg.streaming.source.video_path = str(self.root / "source.mp4")
        self.cfg.streaming.score_midi = str(self.root / "score.mid")
        self.cfg.streaming.beat_path = str(self.root / "beats.txt")
        Path(self.cfg.streaming.score_midi).touch()
        self.cfg.streaming.output_dir = str(self.root / "session")
        self.cfg.streaming.airpiano_device = "cpu"
        self.cfg.streaming.wilor_devices = ["cpu"]
        self.cfg.streaming.preview = False
        self.cfg.streaming.recording.enabled = False
        self.cfg.streaming.midi_output.enabled = False
        self.cfg.streaming.deadline_ms = None
        self.cfg.streaming.warmup_frames = 1
        self.cfg.streaming.warmup_airpiano_frames = 1
        self.cfg.streaming.torch_threads = None
        self.cfg.train.bf16 = False
        self.output = Path(self.cfg.streaming.output_dir)

    def run_session(self, indices, *, delays=None, fail=False, model=None, export_error=None, capture_factory=Capture, scheduler_factory=IdleScheduler):
        self.clock = Clock()
        self.capture = capture_factory(self.clock, indices)
        self.capture_constructor = mock.Mock(return_value=self.capture)
        self.pool = Pool(self.clock, delays, fail)
        self.model = model if model is not None else RecordingModel()
        config = components()
        config["midi_loader"] = SimpleNamespace(load=lambda *args: SimpleNamespace(chords=chords()))
        with ExitStack() as stack:
            for target, replacement in (
                ("airpiano.streaming.io.CaptureSource", self.capture_constructor),
                ("airpiano.streaming.io.WiLoRWorkerPool", mock.Mock(return_value=self.pool)),
                ("airpiano.streaming.runtime.data.build_data_components", mock.Mock(return_value=config)),
                ("airpiano.streaming.runtime.load_model_checkpoint", mock.Mock(return_value=self.model)),
                ("airpiano.streaming.runtime.check_checkpoint_timing", mock.Mock(return_value={})),
                ("airpiano.streaming.runtime.ReleaseScheduler", scheduler_factory),
                ("airpiano.streaming.runtime.time.perf_counter", self.clock),
                ("airpiano.streaming.runtime.time.sleep", self.clock.sleep),
            ):
                stack.enter_context(mock.patch(target, replacement))
            if export_error is not None:
                stack.enter_context(mock.patch("airpiano.streaming.runtime.write_midi", side_effect=export_error))
            return runtime.streaming(self.cfg)

    def rows(self):
        with (self.output / "frames.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_out_of_order_results_are_processed_in_order_and_resources_close(self):
        self.cfg.streaming.drop_stale = False
        self.cfg.streaming.max_inflight = 2
        summary = self.run_session([0, 1], delays={0: 0.08, 1: 0.001})
        self.assertEqual(self.pool.arrival_order, [1, 0])
        self.assertEqual([int(row["frame_index"]) for row in self.rows()], [0, 1])
        self.assertEqual(summary["peak_inflight"], 2)
        self.assertEqual(summary["processed_frames"], 2)
        self.assertEqual(summary["dropped_frames"], 0)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(self.capture.closed and self.pool.closed)
        self.assertTrue((self.output / "performance.mid").is_file())

    def test_capture_gaps_are_counted_once_and_reach_logical_history(self):
        summary = self.run_session([0, 3])
        self.assertEqual(summary["source_frame_count"], 4)
        self.assertEqual(summary["processed_frames"], 2)
        self.assertEqual(summary["drop_reasons"], {"capture_overwrite": 2})
        processed = [row for row in self.rows() if row["status"] == "processed"]
        self.assertEqual([int(row["history_frames"]) for row in processed], [1, 4])
        self.assertEqual(summary["drop_fraction"], 0.5)

    def test_full_credit_budget_replaces_waiting_capture_without_growing_queue(self):
        self.cfg.streaming.max_inflight = 1
        summary = self.run_session([0, 1, 2], delays={0: 0.12})
        self.assertEqual(self.pool.submitted, [0, 2])
        self.assertEqual(summary["peak_inflight"], 1)
        self.assertEqual(summary["drop_reasons"], {"replaced_waiting_capture": 1})
        dropped, = [row for row in self.rows() if row["status"] == "dropped"]
        self.assertAlmostEqual(float(dropped["scheduled_at"]), 1 / 30)
        self.assertAlmostEqual(float(dropped["acquired_at"]), 1 / 30)
        self.assertEqual(summary["processed_frames"] + summary["dropped_frames"], 3)

    def test_expired_inflight_is_counted_once_when_its_late_result_arrives(self):
        self.cfg.streaming.max_inflight = 2
        self.cfg.streaming.deadline_ms = 50.0
        summary = self.run_session([0, 1], delays={0: 0.12, 1: 0.0})
        self.assertEqual(self.pool.arrival_order, [1, 0])
        self.assertEqual(summary["processed_frames"], 1)
        self.assertEqual(summary["drop_reasons"], {"inference_deadline": 1})
        self.assertEqual(summary["peak_inflight"], 2)
        self.assertEqual(len(self.rows()), 2)

    def test_worker_error_survives_export_error_and_summary_is_saved(self):
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.run_session([0, 1], fail=True, export_error=OSError("MIDI export failed"))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error"], "worker failed")
        self.assertIn("OSError: MIDI export failed", summary["cleanup_errors"])
        self.assertEqual(summary["drop_reasons"], {"session_stopped": 1})
        self.assertTrue(self.capture.closed and self.pool.closed)

    def test_failed_airpiano_frame_is_recorded_as_unfinished(self):
        class FailingModel(RecordingModel):
            def forward(self, **batch):
                if len(self.calls) == 1:
                    raise RuntimeError("inference failed")
                return super().forward(**batch)
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_session([0], model=FailingModel())
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["source_frame_count"], 1)
        self.assertEqual(summary["processed_frames"], 0)
        self.assertEqual(summary["drop_reasons"], {"session_stopped": 1})
        self.assertTrue(self.capture.closed and self.pool.closed)

    def test_legacy_checkpoint_delay_is_zero_and_mismatch_is_rejected(self):
        saved = {"config": {"dataset": {
            "fps": 30.0, "target_generator": {"softener": {"radius_seconds": 0.2}}
        }}}
        key = "dataset.target_generator.event_generator.delay_seconds"
        with mock.patch("airpiano.streaming.runtime.torch.load", return_value=saved):
            report = runtime.check_checkpoint_timing(self.cfg, self.root / "legacy.pt")
            self.assertEqual(report["legacy_assumptions"], {key: 0.0})
            self.assertEqual(report["verified"][key], 0.0)
            self.cfg.dataset.target_generator.event_generator.delay_seconds = -0.1
            with self.assertRaisesRegex(ValueError, "Checkpoint timing mismatch"):
                runtime.check_checkpoint_timing(self.cfg, self.root / "legacy.pt")

    def test_entrypoint_dispatches_streaming_stage(self):
        import run
        handler = mock.Mock()
        self.cfg.stage = "streaming"
        self.assertIs(run.STAGES["streaming"], runtime.streaming)
        with mock.patch.dict(run.STAGES, {"streaming": handler}):
            run.main.__wrapped__(self.cfg)
        handler.assert_called_once_with(self.cfg)

    def test_srt_config_validates_protocol_mode_and_timeouts(self):
        self.cfg.streaming.source.kind = "srt"
        runtime.validate_config(self.cfg)
        valid_url = self.cfg.streaming.source.srt_url
        for url in ("rtmp://host/live", "srt://host", "srt://host:99999", "srt://host:0",
                    "srt://host:9001?mode=invalid", "srt://host:9001?mode=caller&mode=listener"):
            with self.subTest(url=url):
                self.cfg.streaming.source.srt_url = url
                with self.assertRaisesRegex(ValueError, "srt_url"):
                    runtime.validate_config(self.cfg)
        self.cfg.streaming.source.srt_url = valid_url
        for name in ("srt_open_timeout_seconds", "srt_read_timeout_seconds"):
            original = self.cfg.streaming.source[name]
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(name=name, value=value):
                    self.cfg.streaming.source[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        runtime.validate_config(self.cfg)
            self.cfg.streaming.source[name] = original
        self.cfg.streaming.drop_stale = False
        with self.assertRaisesRegex(ValueError, "videos only"):
            runtime.validate_config(self.cfg)

    def test_srt_runtime_preserves_media_time_and_logs_local_processing(self):
        class SRTCapture(Capture):
            def wait_ready(self, **kwargs):
                self.ready_timeout = kwargs["timeout"]
                result = super().wait_ready(**kwargs)
                result.update(kind="srt", frame_count=None)
                return result

            def poll(self, **kwargs):
                frame = super().poll(**kwargs)
                if frame is not None and frame["kind"] == "frame":
                    # Media position and decode arrival deliberately differ.
                    frame.update(source_pts_seconds=500.0 + frame["source_time"],
                                 decoded_at=frame["acquired_at"],
                                 source_time=frame["source_time"] + 0.015,
                                 arrival_pts_drift_ms=-15.0,
                                 captured_frames=frame["sequence"] + 1,
                                 read_count=frame["sequence"] + 2,
                                 duplicate_reads_dropped=1)
                return frame

        self.cfg.streaming.source.kind = "srt"
        self.cfg.streaming.source.srt_open_timeout_seconds = 4.0
        self.cfg.streaming.source.srt_read_timeout_seconds = 2.0
        summary = self.run_session([0, 1], capture_factory=SRTCapture)
        self.assertEqual(self.capture_constructor.call_args.args,
                         (self.cfg.streaming.source.srt_url,))
        self.assertEqual(self.capture_constructor.call_args.kwargs["srt_options"],
                         {"open_timeout_seconds": 4.0, "read_timeout_seconds": 2.0})
        self.assertGreater(self.capture.ready_timeout, 6.0)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["source_pts_seconds"]), 500.0)
        self.assertAlmostEqual(float(rows[0]["source_time"]), 0.015)
        self.assertAlmostEqual(float(rows[0]["decoded_at"]), 0.0)
        self.assertEqual(rows[0]["decoded_at"], rows[0]["scheduled_at"])
        self.assertTrue(all(float(row["latency_ms"]) >= 0 for row in rows))
        self.assertAlmostEqual(summary["srt_arrival_pts_drift_ms"]["average"], -15.0)
        self.assertEqual(summary["srt_read_count"], 3)
        self.assertEqual(summary["srt_duplicate_reads_dropped"], 1)
        self.assertIn("excludes phone", summary["timing_scope"])

    def test_srt_disconnect_preserves_partial_outputs_and_closes_workers(self):
        class DisconnectingCapture(Capture):
            def poll(self, **kwargs):
                if self.finished:
                    return None
                frame = super().poll(**kwargs)
                if frame is not None and frame["kind"] == "eof":
                    raise RuntimeError("SRT sender disconnected")
                return frame

        self.cfg.streaming.source.kind = "srt"
        with self.assertRaisesRegex(RuntimeError, "SRT sender disconnected"):
            self.run_session([0, 1], capture_factory=DisconnectingCapture)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "error")
        self.assertGreaterEqual(summary["processed_frames"], 1)
        self.assertTrue((self.output / "performance.mid").is_file())
        self.assertTrue(self.capture.closed and self.pool.closed)

    def test_trigger_console_contains_score_pitches_and_countdown_is_excluded(self):
        self.cfg.streaming.countdown_seconds = 5
        with mock.patch("builtins.print") as output:
            summary = self.run_session([0, 1], model=RecordingModel(trigger=True))
        self.assertAlmostEqual(self.capture.epoch, 105.0)
        self.assertLess(summary["elapsed_seconds"], 1.0)
        messages = [call.args[0] for call in output.call_args_list]
        go = messages.index("[streaming] GO! Start playing.")
        triggers = [i for i, message in enumerate(messages) if "[midi trigger]" in message]
        self.assertTrue(triggers)
        self.assertGreater(triggers[0], go)
        self.assertIn("pitches=[60]", messages[triggers[0]])
        records = [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["pitches"], [60])

    def test_interruption_during_countdown_cleans_up_without_starting_session(self):
        class InterruptedPreparation(Capture):
            def healthcheck(self):
                if self.epoch is not None:
                    raise KeyboardInterrupt

        with mock.patch("builtins.print") as output:
            summary = self.run_session([0], capture_factory=InterruptedPreparation)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["elapsed_seconds"], 0.0)
        self.assertEqual(summary["processed_frames"], 0)
        self.assertTrue(self.capture.closed and self.pool.closed)
        self.assertFalse(any("GO!" in call.args[0] for call in output.call_args_list))

    def test_countdown_requires_nonnegative_integer_seconds(self):
        for value in (-1, 1.5, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.cfg.streaming.countdown_seconds = value
                with self.assertRaisesRegex(ValueError, "countdown_seconds"):
                    runtime.validate_config(self.cfg)
        self.cfg.streaming.countdown_seconds = 0
        runtime.validate_config(self.cfg)

    def test_recording_stays_optional_when_encoder_fails(self):
        self.cfg.streaming.recording.enabled = True
        recorder = mock.Mock()
        recorder.healthcheck.return_value = {"status": "error", "error": "encoder unavailable"}
        recorder.finish.return_value = {"status": "error", "error": "encoder unavailable"}
        with mock.patch("airpiano.streaming.recording.VideoRecorder", return_value=recorder):
            summary = self.run_session([0, 1])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["processed_frames"], 2)
        self.assertEqual(summary["recording"]["error"], "encoder unavailable")
        self.assertTrue((self.output / "performance.mid").is_file())
        self.assertEqual(json.loads((self.output / "recording.json").read_text()), summary["recording"])
        recorder.stop_accepting.assert_called_once()
        recorder.finish.assert_called_once()

    def test_recording_uses_capture_sink_and_extends_to_last_midi_note_end(self):
        class ReleasingScheduler(IdleScheduler):
            def __init__(self, stepper, epoch, emit):
                super().__init__()
                self.stepper, self.emit = stepper, emit

            def close(self, **kwargs):
                controller = self.stepper.controller
                for release in controller.release_due(self.stepper.last_frame_index,
                                                       now=0.5, frame_time=self.stepper.last_source_time):
                    self.emit(release)

        self.cfg.streaming.recording.enabled = True
        recorder = mock.Mock()
        recorder.healthcheck.return_value = {"status": "running"}
        recorder.finish.return_value = {"status": "completed", "frame_count": 45}
        with mock.patch("airpiano.streaming.recording.VideoRecorder", return_value=recorder) as factory:
            summary = self.run_session([0, 1], model=RecordingModel(trigger=True),
                                       scheduler_factory=ReleasingScheduler)
        self.assertIs(self.capture_constructor.call_args.kwargs["recording_sink"], recorder.sink)
        self.assertEqual(factory.call_args.args, (self.output / "performance.mp4",))
        self.assertEqual(factory.call_args.kwargs, {"max_dimension": 640, "fps": 30, "crf": 30})
        self.assertAlmostEqual(recorder.finish.call_args.args[0], 1.5)
        self.assertEqual(summary["released_chords"], 1)
        self.assertLess(summary["elapsed_seconds"], 1.5)

    def test_recording_config_has_bounded_valid_encoder_settings(self):
        self.cfg.streaming.recording.enabled = True
        for name, invalid in (("max_dimension", (0, 641, 8192)),
                              ("fps", (0, 60, 1.5)), ("crf", (-1, 52, float("nan")))):
            original = self.cfg.streaming.recording[name]
            for value in invalid:
                with self.subTest(name=name, value=value):
                    self.cfg.streaming.recording[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        runtime.validate_config(self.cfg)
            self.cfg.streaming.recording[name] = original
        runtime.validate_config(self.cfg)

    def test_websocket_sends_once_at_release_after_warmup_and_before_log_io(self):
        sender = mock.Mock()
        sender.close.return_value = {"status": "closed", "sent_events": 1}
        self.cfg.streaming.midi_output.enabled = True
        self.cfg.streaming.countdown_midi = False
        seen = []
        original_event = runtime.SessionLog.event

        def log_event(log, kind, record):
            if kind == "trigger":
                sender.submit.assert_not_called()
            elif kind == "release":
                sender.submit.assert_called_once()
                seen.append(record)
            return original_event(log, kind, record)

        class ReleasingScheduler(IdleScheduler):
            def __init__(self, stepper, epoch, emit):
                super().__init__()
                sender.start.assert_called_once()
                sender.submit.assert_not_called()
                self.stepper, self.emit = stepper, emit

            def close(self, **kwargs):
                # Drain twice to check that occurrence latching cannot send twice.
                for _ in range(2):
                    for release in self.stepper.controller.release_due(
                            self.stepper.last_frame_index, now=0.5,
                            frame_time=self.stepper.last_source_time):
                        self.emit(release)

        with mock.patch("airpiano.streaming.midi_output.WebSocketMidiOutput", return_value=sender) as factory, \
             mock.patch.object(runtime.SessionLog, "event", log_event):
            summary = self.run_session([0, 1, 2], model=RecordingModel(trigger=True),
                                       scheduler_factory=ReleasingScheduler)
        self.assertEqual(summary["released_chords"], 1)
        self.assertEqual(summary["midi_output"], sender.close.return_value)
        self.assertEqual(factory.call_args.args, (self.cfg.streaming.midi_output.url,))
        self.assertEqual(factory.call_args.kwargs["capacity"], 32)
        sender.submit.assert_called_once_with(seen[0], released_at=self.capture.epoch + 0.5)
        self.assertEqual(tuple(seen[0]["pitches"]), (60,))
        sender.close.assert_called_once_with(drain=True, timeout=2.0)
        self.assertEqual(json.loads((self.output / "midi_output.json").read_text()), summary["midi_output"])

    def test_websocket_does_not_send_raw_triggers(self):
        self.cfg.streaming.midi_output.enabled = True
        self.cfg.streaming.countdown_midi = False
        sender = mock.Mock()
        sender.close.return_value = {"status": "closed", "sent_events": 0}
        with mock.patch("airpiano.streaming.midi_output.WebSocketMidiOutput", return_value=sender):
            summary = self.run_session([0, 1], model=RecordingModel(trigger=True))
        records = [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(record["kind"] == "trigger" for record in records))
        self.assertEqual(summary["released_chords"], 0)
        sender.submit.assert_not_called()

    def test_websocket_interruption_drops_pending_output_and_preserves_session(self):
        class InterruptedPreparation(Capture):
            def healthcheck(self):
                if self.epoch is not None:
                    raise KeyboardInterrupt

        self.cfg.streaming.midi_output.enabled = True
        sender = mock.Mock()
        sender.close.return_value = {"status": "closed", "sent_events": 0}
        with mock.patch("airpiano.streaming.midi_output.WebSocketMidiOutput", return_value=sender):
            summary = self.run_session([0], capture_factory=InterruptedPreparation)
        self.assertEqual(summary["status"], "interrupted")
        sender.submit.assert_not_called()
        sender.close.assert_called_once_with(drain=False, timeout=2.0)
        self.assertTrue(self.capture.closed and self.pool.closed)
        self.assertTrue((self.output / "performance.mid").is_file())

    def test_websocket_unavailable_is_soft_failure_with_transport_diagnostics(self):
        self.cfg.streaming.midi_output.enabled = True
        sender = mock.Mock()
        sender.close.return_value = {"status": "disabled", "last_error": "missing websockets"}
        def factory(*args, **kwargs):
            sender.start.side_effect = lambda: kwargs["emit"]({
                "status": "disabled", "timestamp": self.clock(), "error": "missing websockets"})
            return sender
        with mock.patch("airpiano.streaming.midi_output.WebSocketMidiOutput", side_effect=factory), \
             mock.patch("builtins.print") as output:
            summary = self.run_session([0, 1])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["processed_frames"], 2)
        self.assertEqual(summary["midi_output"]["status"], "disabled")
        self.assertTrue(any("[midi websocket] disabled: missing websockets" in call.args[0]
                            for call in output.call_args_list))
        records = [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]
        transport, = [record for record in records if record["kind"] == "midi_output"]
        self.assertIsNone(transport["session_time"])
        self.assertTrue((self.output / "performance.mid").is_file())

    def test_websocket_disabled_does_not_create_sender(self):
        with mock.patch("airpiano.streaming.midi_output.WebSocketMidiOutput") as factory:
            summary = self.run_session([0])
        factory.assert_not_called()
        self.assertEqual(summary["midi_output"]["status"], "disabled")

    def test_websocket_config_validates_url_capacity_and_time_limits(self):
        self.cfg.streaming.midi_output.enabled = True
        runtime.validate_config(self.cfg)
        valid_url = self.cfg.streaming.midi_output.url
        for url in ("http://host:8765", "ws:///", "ws://host:0", "ws://host:99999", "ws://host/#fragment"):
            with self.subTest(url=url):
                self.cfg.streaming.midi_output.url = url
                with self.assertRaisesRegex(ValueError, "midi_output.url"):
                    runtime.validate_config(self.cfg)
        self.cfg.streaming.midi_output.url = valid_url
        for name, invalid in (("capacity", (0, -1, 1.5, 1025, float("inf"))),
                              ("max_age_ms", (0, -1, float("inf"), float("nan"))),
                              ("open_timeout_seconds", (0, -1)),
                              ("close_timeout_seconds", (0, -1)),
                              ("reconnect_seconds", (0, -1))):
            original = self.cfg.streaming.midi_output[name]
            for value in invalid:
                with self.subTest(name=name, value=value):
                    self.cfg.streaming.midi_output[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        runtime.validate_config(self.cfg)
            self.cfg.streaming.midi_output[name] = original
        runtime.validate_config(self.cfg)

    def test_no_drop_mode_requires_video_and_no_deadline(self):
        self.cfg.streaming.drop_stale = False
        runtime.validate_config(self.cfg)
        self.cfg.streaming.deadline_ms = 100.0
        with self.assertRaisesRegex(ValueError, "deadline_ms=null"):
            runtime.validate_config(self.cfg)
        self.cfg.streaming.deadline_ms = None
        self.cfg.streaming.source.kind = "camera"
        with self.assertRaisesRegex(ValueError, "videos only"):
            runtime.validate_config(self.cfg)


if __name__ == "__main__":
    unittest.main()
