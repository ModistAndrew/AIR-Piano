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

    def test_dispatch_passes_the_same_config_to_only_the_selected_handler(self) -> None:
        config = OmegaConf.create({"stage": "realtime", "sentinel": {"value": 1}})
        handlers = {name: Mock() for name in app.STAGES}

        with patch.object(app, "STAGES", handlers):
            app.main.__wrapped__(config)

        handlers["realtime"].assert_called_once_with(config)
        self.assertIs(handlers["realtime"].call_args.args[0], config)
        for name, handler in handlers.items():
            if name != "realtime":
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
        for name in ("realtime", "visualize", "sonify"):
            handlers[name].assert_not_called()

    def test_stage_list_stops_after_a_handler_failure(self) -> None:
        config = OmegaConf.create(
            {"stage": ["preprocess", "train", "realtime"]}
        )
        handlers = {name: Mock() for name in app.STAGES}
        handlers["train"].side_effect = RuntimeError("training failed")

        with patch.object(app, "STAGES", handlers):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                app.main.__wrapped__(config)

        handlers["preprocess"].assert_called_once_with(config)
        handlers["train"].assert_called_once_with(config)
        handlers["test"].assert_not_called()
        handlers["realtime"].assert_not_called()
        handlers["visualize"].assert_not_called()
        handlers["sonify"].assert_not_called()

    def test_unknown_stage_fails_at_dispatch_lookup(self) -> None:
        with self.assertRaises(KeyError):
            app.main.__wrapped__(OmegaConf.create({"stage": "unknown"}))


if __name__ == "__main__":
    unittest.main()
