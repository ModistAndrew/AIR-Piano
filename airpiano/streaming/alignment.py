"""Causal alignment heatmaps for a live preview and compact session exports."""

from __future__ import annotations

import math
import os
import tempfile
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np


_NO_OBSERVATION_BGR = (220, 216, 213)


class AlignmentVisualization:
    """Retain evaluated score windows without allocating a whole-song matrix."""

    def __init__(
        self, chord_count: int, history_seconds: float = 15.0, fps: float = 30.0,
    ) -> None:
        self.chord_count = int(chord_count)
        self.history_seconds = float(history_seconds)
        self.fps = float(fps)
        self.frame_indices: list[int] = []
        self.source_times: list[float] = []
        self.window_starts: list[int] = []
        self.history_starts: list[int] = []
        self.probabilities: list[np.ndarray] = []

    def append(
        self, *, frame_index: int, source_time: float, window_start: int,
        history_start: int, scores: np.ndarray,
    ) -> None:
        self.frame_indices.append(int(frame_index))
        self.source_times.append(float(source_time))
        self.window_starts.append(int(window_start))
        self.history_starts.append(int(history_start))
        self.probabilities.append(np.array(scores, dtype=np.float32, copy=True))

    def _visible_indices(self, time_start: float, time_end: float):
        first = bisect_right(self.source_times, time_start - 1.0 / self.fps)
        last = bisect_left(self.source_times, time_end)
        for index in range(first, last):
            if self._support_end(index) > time_start:
                yield index

    def _support_end(self, index: int) -> float:
        stop = self.source_times[index] + 1.0 / self.fps
        if index + 1 < len(self.source_times):
            stop = min(stop, self.source_times[index + 1])
        return stop

    def rasterize(
        self, *, time_start: float, time_end: float, chord_start: int = 1,
        chord_end: int | None = None, width: int = 960, height: int = 360,
    ) -> np.ndarray:
        """Return ascending chord rows, max pooling intersecting pixel cells.

        Rows cover inclusive global chord IDs. The raster has at most ``height``
        rows and exactly ``width`` columns; NaN means no evaluated observation.
        Each observation covers at most one nominal frame period, clipped by the
        next actual timestamp. No intermediate observations are manufactured.
        """
        if chord_end is None:
            chord_end = self.chord_count
        chord_span = chord_end - chord_start + 1
        rows = min(height, chord_span)
        raster = np.full((rows, width), np.nan, dtype=np.float32)
        time_scale = width / (time_end - time_start)
        for index in self._visible_indices(time_start, time_end):
            source_time = self.source_times[index]
            stop = min(time_end, self._support_end(index))
            if stop <= source_time:
                continue
            left = max(0, math.floor((source_time - time_start) * time_scale))
            # Avoid extending an exact exclusive boundary by one pixel because
            # a decimal timestamp rounded one ULP above its raster boundary.
            right_edge = np.nextafter((stop - time_start) * time_scale, -math.inf)
            right = min(width, math.ceil(right_edge))
            if left >= right:
                continue
            scores = self.probabilities[index]
            first_chord = self.window_starts[index]
            start_slot = max(0, chord_start - first_chord)
            stop_slot = min(len(scores), chord_end - first_chord + 1)
            if start_slot >= stop_slot:
                continue
            chord_ids = np.arange(first_chord + start_slot, first_chord + stop_slot)
            row_indices = (chord_ids - chord_start) * rows // chord_span
            column = np.full(rows, np.nan, dtype=np.float32)
            np.fmax.at(column, row_indices, scores[start_slot:stop_slot])
            region = raster[:, left:right]
            np.fmax(region, column[:, None], out=region)
        return raster

    def _time_end(self, requested: float | None) -> float:
        observed_end = (
            self.source_times[-1] + 1.0 / self.fps if self.source_times
            else 1.0 / self.fps
        )
        return max(observed_end, observed_end if requested is None else requested)

    @staticmethod
    def _colors(values: np.ndarray) -> np.ndarray:
        levels = np.rint(np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(levels, cv2.COLORMAP_MAGMA)
        colors[np.isnan(values)] = _NO_OBSERVATION_BGR
        return colors

    def render(
        self, *, current_time: float, current_chord: int,
        width: int = 960, height: int = 360,
    ) -> np.ndarray:
        """Return a BGR panel without creating a GUI window or a dense history."""
        time_end = self._time_end(current_time)
        time_start = max(0.0, time_end - self.history_seconds)
        visible = list(self._visible_indices(time_start, time_end))
        if visible:
            chord_start = min(self.window_starts[index] for index in visible)
            chord_end = max(
                self.window_starts[index] + len(self.probabilities[index]) - 1
                for index in visible
            )
        else:
            chord_start, chord_end = 1, self.chord_count
        left, right = min(70, width // 5), min(110, width // 4)
        top, bottom = min(45, height // 4), min(55, height // 3)
        plot_width, plot_height = max(1, width - left - right), max(1, height - top - bottom)
        raster = self.rasterize(
            time_start=time_start, time_end=time_end, chord_start=chord_start,
            chord_end=chord_end, width=plot_width, height=plot_height,
        )
        panel = np.full((height, width, 3), 250, dtype=np.uint8)
        colors = cv2.resize(self._colors(raster)[::-1], (plot_width, plot_height), interpolation=cv2.INTER_NEAREST)
        panel[top:top + plot_height, left:left + plot_width] = colors
        font, ink = cv2.FONT_HERSHEY_SIMPLEX, (35, 35, 35)

        def label(text: str, x: int, y: int, scale: float = 0.42):
            cv2.putText(panel, text, (x, y), font, scale, ink, 1, cv2.LINE_AA)

        label(f"Alignment probability | current chord {current_chord}", left, 20, 0.52)
        label("Global chord ID", 5, top - 8, 0.35)
        for chord_id in sorted({chord_start, (chord_start + chord_end) // 2, chord_end}):
            fraction = (chord_id - chord_start + 0.5) / (chord_end - chord_start + 1)
            y = top + int((1.0 - fraction) * plot_height)
            label(str(chord_id), max(3, left - 42), y + 4)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            seconds = time_start + fraction * (time_end - time_start)
            x = left + round(fraction * max(0, plot_width - 1))
            label(f"{seconds:.1f}", max(0, x - 12), top + plot_height + 17)
        label("Source time (s) | max probability per pixel", left, height - 10, 0.38)
        legend_x = left + plot_width + 15
        gradient = np.linspace(1.0, 0.0, plot_height, dtype=np.float32)[:, None]
        legend_width = max(0, min(12, width - legend_x))
        bar = np.repeat(self._colors(gradient), legend_width, axis=1)
        panel[top:top + plot_height, legend_x:legend_x + legend_width] = bar
        for value in (1.0, 0.5, 0.0):
            y = top + round((1.0 - value) * max(0, plot_height - 1))
            label(f"{value:.1f}", legend_x + 17, min(height - 1, y + 4), 0.35)
        label("Gray: no observation", max(left, width - 185), 35, 0.35)
        if not visible:
            label("No observations in this interval", left + 10, top + plot_height // 2)
        return panel

    def save(self, output_dir: Path, *, end_time: float | None = None) -> dict:
        """Write the full-session raster and lossless compact evaluated windows."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path, data_path = output_dir / "alignment.png", output_dir / "alignment.npz"
        frame_count = len(self.frame_indices)
        slots = max((len(scores) for scores in self.probabilities), default=0)
        probabilities = np.full((frame_count, slots), np.nan, dtype=np.float32)
        for index, scores in enumerate(self.probabilities):
            probabilities[index, :len(scores)] = scores
        np.savez_compressed(
            data_path,
            frame_index=np.asarray(self.frame_indices, dtype=np.int64),
            source_time=np.asarray(self.source_times, dtype=np.float64),
            window_start=np.asarray(self.window_starts, dtype=np.int64),
            history_start=np.asarray(self.history_starts, dtype=np.int64),
            probabilities=probabilities,
            chord_count=np.asarray(self.chord_count, dtype=np.int64),
            fps=np.asarray(self.fps, dtype=np.float64),
        )
        cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        time_end = self._time_end(end_time)
        raster = self.rasterize(time_start=0.0, time_end=time_end, width=1800, height=720)
        figure, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
        try:
            cmap = matplotlib.colormaps["magma"].with_extremes(bad="#d5d8dc")
            image = axis.imshow(
                np.ma.masked_invalid(raster), origin="lower", aspect="auto",
                interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0,
                extent=(0.0, time_end, 0.5, self.chord_count + 0.5),
            )
            axis.set(title="Alignment probability", xlabel="Source time (seconds)", ylabel="Global chord ID")
            axis.legend(handles=[Patch(facecolor="#d5d8dc", label="No observation / not evaluated")], loc="upper left")
            figure.colorbar(image, ax=axis, label="Probability", ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
            figure.suptitle("Causal newest-frame scores; maximum probability per raster pixel", fontsize=10)
            if not frame_count:
                axis.text(0.5, 0.5, "No observations", transform=axis.transAxes, ha="center", va="center")
            figure.savefig(image_path, dpi=150, facecolor="white")
        finally:
            plt.close(figure)
        return {
            "status": "saved" if frame_count else "empty",
            "image_path": str(image_path), "data_path": str(data_path),
            "frame_count": frame_count,
        }
