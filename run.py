#!/usr/bin/env python3
"""Hydra entry point for the AirPiano stage handlers."""

from __future__ import annotations

from collections.abc import Callable

import hydra
from omegaconf import DictConfig, ListConfig

from airpiano.inference import test
from airpiano.preprocessing.pipeline import preprocess
from airpiano.realtime import realtime
from airpiano.render.sonify import sonify
from airpiano.render.visualize import visualize
from airpiano.train import train


StageHandler = Callable[[DictConfig], None]

STAGES: dict[str, StageHandler] = {
    "preprocess": preprocess,
    "train": train,
    "test": test,
    "realtime": realtime,
    "visualize": visualize,
    "sonify": sonify,
}


@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    """Run one stage or a Hydra list of stages in the requested order."""

    stages = cfg.stage if isinstance(cfg.stage, ListConfig) else [cfg.stage]
    for stage in stages:
        STAGES[str(stage)](cfg)


if __name__ == "__main__":
    main()
