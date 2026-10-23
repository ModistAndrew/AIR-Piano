"""Plot saved full_song_real_time windows, histories, triggers, and MIDI onsets."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airpiano.render.artifact import load_artifact


def visualize_trace(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    start: float | None = None,
    end: float | None = None,
) -> list[Path]:
    """Use recorded inputs and events; no inference or timing reconstruction."""
    input_path = Path(input_path)
    artifact = load_artifact(input_path, require_releases=True)
    with np.load(input_path, allow_pickle=False) as payload:
        trace = {
            key: payload[key].copy()
            for key in (
                "frame_time", "frame_chord_window_start",
                "frame_video_history_start", "window_size",
                "event_frame_index", "event_chord_id", "event_success",
                "event_match_score", "event_old_best",
            )
        }

    time = trace["frame_time"]
    frame_count, chord_count = artifact.probability.shape
    chord_ids = artifact.chord_ids
    if not np.array_equal(chord_ids, np.arange(1, chord_count + 1)):
        raise ValueError("Expected real-time artifact with 1-based chord IDs")
    window_start = trace["frame_chord_window_start"]
    history_start = trace["frame_video_history_start"]
    if any(values.shape != (frame_count,) for values in
           (time, window_start, history_start)):
        raise ValueError("Frame traces must cover every saved inference frame")
    if np.any(history_start < 0) or np.any(history_start > np.arange(frame_count)):
        raise ValueError("History indices must refer to received frames")

    left = float(time[0]) if start is None else float(start)
    right = float(time[-1] + 1 / artifact.fps) if end is None else float(end)
    visible = (time >= left) & (time <= right)
    if right <= left or not np.any(visible):
        raise ValueError("Requested time range contains no frames")

    window_size = int(trace["window_size"])
    window_end = np.minimum(window_start + window_size - 1, chord_count)
    history_time = time[history_start]
    history_length = (np.arange(frame_count) - history_start + 1) / artifact.fps
    rewind = np.flatnonzero(np.diff(history_start) < 0) + 1
    event_time = time[trace["event_frame_index"]]
    success = trace["event_success"].astype(bool)
    event_ids = trace["event_chord_id"]
    margin = trace["event_match_score"] - trace["event_old_best"]
    released = np.isfinite(artifact.release_time) & artifact.chord_mask

    # Stored targets are a reference, not an input to the plotted controller.
    valid_frames = np.flatnonzero(artifact.frame_mask)
    reference_chords = np.flatnonzero(
        artifact.chord_mask
        & np.any(artifact.target[valid_frames] > 0, axis=0)
    )
    reference_time = time[valid_frames[
        artifact.target[np.ix_(valid_frames, reference_chords)].argmax(axis=0)
    ]] if reference_chords.size else np.empty(0)
    reference_ids = chord_ids[reference_chords]

    cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    blue, green, red, purple = "#3478b5", "#168250", "#d04740", "#7c4cb2"
    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    }):
        figure, axes = plt.subplots(
            5, 1, figsize=(16, 15), sharex=True,
            gridspec_kw={"height_ratios": [2.8, 1.8, 1.0, 1.3, 2.2]},
        )
        figure.subplots_adjust(
            left=0.08, right=0.93, top=0.90, bottom=0.115, hspace=0.40,
        )
        accepted, rejected = int(success.sum()), int((~success).sum())
        figure.suptitle(
            f"{input_path.stem} | full_song_real_time\n"
            f"{frame_count / artifact.fps:.2f} s  |  {chord_count} chords  |  "
            f"window {window_size}  |  DP accepted {accepted}, rejected {rejected}"
            f"  |  MIDI released {int(released.sum())}\n"
            f"{artifact.record_id}",
            fontsize=14, y=0.98,
        )

        axis = axes[0]
        axis.fill_between(
            time, window_start, window_end, step="post",
            color=blue, alpha=0.17, label="Model chord window",
        )
        axis.step(time, window_start, where="post", color=blue, lw=0.8)
        axis.step(time, window_end, where="post", color=blue, lw=0.8)
        axis.plot(
            reference_time, reference_ids, color="#777777",
            ls="--", lw=1, label="Stored target peaks", alpha=0.8,
        )
        axis.scatter(event_time[success], event_ids[success],
                     s=15, c=green, label="DP accepted", zorder=4)
        axis.scatter(event_time[~success], event_ids[~success],
                     s=28, c=red, marker="x", linewidths=1.2,
                     label="DP rejected", zorder=5)
        axis.set(title="1. Sliding chord window and threshold-triggered events",
                 ylabel="Score chord ID")
        axis.legend(loc="upper left", ncol=4, fontsize=9)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        axis = axes[1]
        axis.fill_between(
            time, history_time, time, step="post", color=blue,
            alpha=0.14, label="Received history used by model",
        )
        axis.plot(time, time, color="#777777", lw=1, ls="--",
                  label="Current received frame")
        axis.step(time, history_time, where="post", color=purple, lw=1.1,
                  label="History start")
        axis.scatter(time[rewind], history_time[rewind],
                     color=red, marker="v", s=25, zorder=4,
                     label="History moved backward")
        axis.set(title="2. Video history used for each forward pass",
                 ylabel="Video timestamp (s)")
        axis.legend(loc="upper left", ncol=4, fontsize=9)
        axis.set_ylim(
            max(0, float(history_time[visible].min()) - 1), right + 1,
        )

        axis = axes[2]
        axis.fill_between(time, 0, history_length, step="post",
                          color=blue, alpha=0.14)
        axis.step(time, history_length, where="post", color=blue, lw=1)
        axis.set(title="3. Video history length (including current frame)",
                 ylabel="Seconds")
        axis.set_ylim(0, float(history_length[visible].max()) * 1.15)
        secondary = axis.secondary_yaxis(
            "right",
            functions=(lambda x: x * artifact.fps,
                       lambda x: x / artifact.fps),
        )
        secondary.set_ylabel("Frames")

        axis = axes[3]
        axis.axhline(0, color="#777777", lw=1, ls="--")
        axis.scatter(event_time[success], margin[success],
                     s=15, c=green, label="New historical best")
        axis.scatter(event_time[~success], margin[~success],
                     s=25, c=red, marker="x", linewidths=1.2,
                     label="No new historical best")
        axis.set(title="4. DP decision: match score minus previous historical best",
                 ylabel="DP margin")
        selected_events = (event_time >= left) & (event_time <= right)
        if np.any(selected_events):
            axis.set_ylim(
                min(0, float(margin[selected_events].min())) - 0.5,
                max(1, float(margin[selected_events].max())) + 0.5,
            )
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        axis = axes[4]
        axis.plot(reference_time, reference_ids, color="#777777",
                  ls="--", lw=1, label="Stored target peaks")
        axis.scatter(artifact.release_time[released], chord_ids[released],
                     color=purple, s=17, marker="|", linewidths=1.3,
                     label="Actual MIDI onset (release_time)", zorder=4)
        missed = ~released[reference_chords]
        axis.scatter(reference_time[missed], reference_ids[missed],
                     color="#c28721", s=18, marker="x", linewidths=1,
                     label="No saved MIDI release (at target time)", zorder=3)
        axis.set(title="5. Recorded MIDI note-on times",
                 ylabel="Score chord ID", xlabel="Time (seconds)")
        axis.legend(loc="upper left", ncol=3, fontsize=9)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Keep local chord detail legible for time-range zooms.
        if start is not None or end is not None:
            in_reference = (reference_time >= left) & (reference_time <= right)
            in_release = released & (artifact.release_time >= left) & (
                artifact.release_time <= right
            )
            ranges = np.concatenate((
                window_start[visible], window_end[visible],
                event_ids[selected_events], reference_ids[in_reference],
                chord_ids[in_release],
            ))
            for index in (0, 4):
                axes[index].set_ylim(
                    max(0, int(ranges.min()) - 5),
                    min(chord_count + 1, int(ranges.max()) + 5),
                )
        for axis in axes:
            axis.grid(alpha=0.15)
            axis.set_xlim(left, right)
        figure.text(
            0.08, 0.035,
            "Panels 1-4 use incoming frame time. Window/history traces are the "
            "inputs used before that frame's events; updates affect the next pass.\n"
            "DP accepted/rejected describes alignment only: rejected triggers "
            "can still release MIDI. Panel 5 uses saved actual release times.\n"
            "Dashed reference = peak of each chord's saved target. "
            "Counts in the heading cover the whole recording.",
            fontsize=9, color="#444444", va="bottom",
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if start is None and end is None else f"_{left:g}-{right:g}s"
        paths = [
            output_dir / f"{input_path.stem}.trace{suffix}.{extension}"
            for extension in ("png", "svg")
        ]
        for path in paths:
            figure.savefig(path, dpi=150, facecolor="white")
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    for path in args.artifacts:
        for output in visualize_trace(
            path, args.output_dir or path.parent / "trace_plots",
            start=args.start, end=args.end,
        ):
            print(output, flush=True)


if __name__ == "__main__":
    main()
