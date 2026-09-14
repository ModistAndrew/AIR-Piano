"""Draw aligned prediction, target, and residual heatmaps from an artifact."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..utils import resolve_path
from .artifact import load_artifact


def visualize(cfg: DictConfig) -> None:
    """Recursively render every inference artifact below the configured root."""

    root = resolve_path(cfg.visualize.root)
    for input_path in sorted(root.rglob("*.npz")):
        if not input_path.is_file():
            continue
        output = visualize_artifact(input_path, input_path.with_suffix(".png"))
        print(f"wrote {output}", flush=True)


def visualize_artifact(input_path: str | Path, output_path: str | Path) -> Path:
    """Render prediction, target, and their signed difference as heatmaps."""

    artifact = load_artifact(input_path)

    # Keep Matplotlib's cache out of the repository and select a headless backend.
    matplotlib_config = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Transpose so time runs horizontally and mask padded frame/chord pairs.
    support = artifact.support.T
    prediction = np.ma.masked_where(~support, artifact.probability.T)
    target = np.ma.masked_where(~support, artifact.target.T)
    delta = np.ma.masked_where(
        ~support,
        artifact.probability.T - artifact.target.T,
    )
    colormap = matplotlib.colormaps["magma"].with_extremes(bad="#d5d8dc")
    delta_colormap = matplotlib.colormaps["coolwarm"].with_extremes(bad="#d5d8dc")
    width = max(8.0, min(18.0, artifact.probability.shape[0] / 15.0))
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(width, 15.0),
        sharex=True,
        constrained_layout=True,
    )
    for axis, values, title, cmap, vmin, vmax in (
        (axes[0], prediction, "Prediction probability", colormap, 0.0, 1.0),
        (axes[1], target, "Alignment target", colormap, 0.0, 1.0),
        (
            axes[2],
            delta,
            "Prediction − target",
            delta_colormap,
            -1.0,
            1.0,
        ),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_ylabel("Chord slot")
        figure.colorbar(image, ax=axis, pad=0.01, fraction=0.025)

    frames = artifact.probability.shape[0]
    ticks = np.linspace(0, frames - 1, min(9, frames), dtype=int)
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([f"{frame / artifact.fps:.2f}" for frame in ticks])
    axes[2].set_xlabel("Time in slice (seconds)")

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
