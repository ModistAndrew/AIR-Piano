"""Shared SKY-Piano marker parsing and coordinate conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


RAW_SCHEMA_VERSION = "airpiano-skypiano-ragged-v1"
HAND_SIDES = ("L", "R")
JOINT_MARKERS = (
    "Wrist",
    "TC",
    "TM",
    "TD",
    "TF",
    "IM",
    "IP",
    "ID",
    "IF",
    "MM",
    "MP",
    "MD",
    "MF",
    "RM",
    "RP",
    "RD",
    "RF",
    "LM",
    "LP",
    "LD",
    "LF",
)
AXIS_SIGNS = np.asarray((-1.0, 1.0, -1.0), dtype=np.float32)


def read_skypiano_markers(path: str | Path) -> np.ndarray:
    """Read one marker CSV as millimeter XYZ in OpenPose joint order."""

    table = np.atleast_1d(
        np.genfromtxt(
            Path(path),
            delimiter=",",
            names=True,
            dtype=np.float32,
            encoding="utf-8",
        )
    )
    markers = np.empty((len(table), 2, 21, 3), dtype=np.float32)
    for hand, side in enumerate(HAND_SIDES):
        for joint, marker in enumerate(JOINT_MARKERS):
            for axis, axis_name in enumerate("xyz"):
                markers[:, hand, joint, axis] = table[
                    f"{side}{marker}_{axis_name}"
                ]
    return markers


def skypiano_camera_markers(path: str | Path) -> np.ndarray:
    """Return absolute markers in meters after the SKY-to-camera rotation."""

    markers = read_skypiano_markers(path)
    markers /= np.float32(1000.0)
    markers *= AXIS_SIGNS
    return markers


def skypiano_camera_marker_data(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute and wrist-relative camera-convention marker joints."""

    markers = read_skypiano_markers(path)
    absolute = markers / np.float32(1000.0)
    absolute *= AXIS_SIGNS
    local = (markers - markers[:, :, :1]) / np.float32(1000.0)
    local *= AXIS_SIGNS
    return absolute, local
