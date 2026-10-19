"""Tests for the dispatch-only Hydra entry point."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

import run as app


class RunTest(unittest.TestCase):
    def test_cli_resolves_baseline_config(self) -> None:
        root = Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "run.py", "--cfg", "job"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("stage: train", result.stdout)
        self.assertIn("ragged_to_canonical", result.stdout)

    def test_full_song_real_time_config_is_inherited(self) -> None:
        root = Path(__file__).parents[1]
        for preset in ("causal_chord24", "super_96_ahead100ms"):
            with self.subTest(preset=preset):
                result = subprocess.run(
                    [
                        sys.executable, "run.py", "--config-name", preset,
                        "stage=full_song_real_time", "--cfg", "job", "--resolve",
                    ],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                config = OmegaConf.create(result.stdout)
                self.assertEqual(config.stage, "full_song_real_time")
                self.assertEqual(
                    config.full_song_real_time.checkpoint,
                    f"outputs/{preset}/checkpoints/last.pt",
                )
                self.assertEqual(
                    config.full_song_real_time.output_dir,
                    f"outputs/{preset}/full_song_real_time",
                )

    def test_dispatch_passes_the_same_config_to_only_the_selected_handler(self) -> None:
        config = OmegaConf.create({"stage": "test", "sentinel": {"value": 1}})
        handlers = {name: Mock() for name in app.STAGES}

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        handlers["test"].assert_called_once_with(config)
        self.assertIs(handlers["test"].call_args.args[0], config)
        for name, handler in handlers.items():
            if name != "test":
                handler.assert_not_called()

    def test_dispatches_full_song_stage(self) -> None:
        config = OmegaConf.create({"stage": "full-song"})
        handlers = {name: Mock() for name in app.STAGES}

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        handlers["full-song"].assert_called_once_with(config)
        for name, handler in handlers.items():
            if name != "full-song":
                handler.assert_not_called()

    def test_dispatches_full_song_real_time_stage(self) -> None:
        config = OmegaConf.create({"stage": "full_song_real_time"})
        handlers = {name: Mock() for name in app.STAGES}

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        handlers["full_song_real_time"].assert_called_once_with(config)
        for name, handler in handlers.items():
            if name != "full_song_real_time":
                handler.assert_not_called()

    def test_dispatches_evaluation_stage(self) -> None:
        config = OmegaConf.create({"stage": "evaluation"})
        handlers = {name: Mock() for name in app.STAGES}

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        handlers["evaluation"].assert_called_once_with(config)
        for name, handler in handlers.items():
            if name != "evaluation":
                handler.assert_not_called()

    def test_dispatch_runs_stage_list_in_order_with_the_same_config(self) -> None:
        config = OmegaConf.create(
            {"stage": ["preprocess", "train", "test"], "sentinel": {"value": 1}}
        )
        calls: list[tuple[str, object]] = []

        def record(name: str):
            return lambda received: calls.append((name, received))

        handlers = {
            name: Mock(side_effect=record(name))
            for name in app.STAGES
        }

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        self.assertEqual(
            [name for name, _ in calls], ["preprocess", "train", "test"]
        )
        self.assertTrue(all(received is config for _, received in calls))
        handlers["preprocess"].assert_called_once_with(config)
        handlers["train"].assert_called_once_with(config)
        handlers["test"].assert_called_once_with(config)
        for name in ("evaluation", "visualize", "sonify"):
            handlers[name].assert_not_called()

    def test_dispatches_full_song_then_evaluation(self) -> None:
        config = OmegaConf.create({"stage": ["full-song", "evaluation"]})
        calls = []
        handlers = {
            name: Mock(side_effect=lambda _config, name=name: calls.append(name))
            for name in app.STAGES
        }

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        self.assertEqual(calls, ["full-song", "evaluation"])

    def test_stage_list_stops_after_a_handler_failure(self) -> None:
        config = OmegaConf.create(
            {"stage": ["preprocess", "train", "full-song"]}
        )
        handlers = {name: Mock() for name in app.STAGES}
        handlers["train"].side_effect = RuntimeError("training failed")

        with patch.object(app, "STAGES", handlers):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                app.main.__wrapped__(config)

        handlers["preprocess"].assert_called_once_with(config)
        handlers["train"].assert_called_once_with(config)
        handlers["test"].assert_not_called()
        handlers["full-song"].assert_not_called()
        handlers["visualize"].assert_not_called()
        handlers["sonify"].assert_not_called()

    def test_removed_realtime_stage_is_unknown(self) -> None:
        self.assertNotIn("realtime", app.STAGES)
        with self.assertRaises(KeyError):
            app.main.__wrapped__(OmegaConf.create({"stage": "realtime"}))

    def test_unknown_stage_fails_at_dispatch_lookup(self) -> None:
        with self.assertRaises(KeyError):
            app.main.__wrapped__(OmegaConf.create({"stage": "unknown"}))


if __name__ == "__main__":
    unittest.main()
