"""Plot saved full-song windows/history with DP events or sonified MIDI onsets."""

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
from airpiano.render.sonify import decode_chord_onsets


def visualize_trace(
    input_path: Path,
    output_dir: Path,
    start: float | None = None,
    end: float | None = None,
    *,
    threshold: float = 0.5,
    onset_mode: str = "first_threshold",
    target_radius_seconds: float = 0.2,
) -> list[Path]:
    """Plot logged inputs; baseline MIDI onsets use the existing sonify decoder."""

    cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with np.load(input_path, allow_pickle=False) as data:
        fps = float(data["fps"])
        frame_time = data["frame_time"]
        window_start = data["frame_chord_window_start"]
        window_size = int(data["window_size"])
        history_start = data["frame_video_history_start"]
        real_time = "event_success" in data
        # Display one-based score occurrences for both inference stages.
        chord_ids = np.arange(1, len(data["chord_ids"]) + 1)
        if real_time:
            event_frames = data["event_frame_index"]
            event_ids = data["event_chord_id"]
            success = data["event_success"]
            margin = data["event_match_score"] - data["event_old_best"]
        target = data["target"]
        valid_frames = np.flatnonzero(data["frame_mask"])
        valid_chords = np.flatnonzero(data["chord_mask"])
        record_id = str(data["record_id"])

    if real_time:
        event_time = frame_time[event_frames]
        event_summary = f"{success.sum()} accepted / {(~success).sum()} rejected triggers"
    else:
        artifact = load_artifact(input_path)
        onset_frames, onset_scores = decode_chord_onsets(
            artifact, threshold, onset_mode, target_radius_seconds,
        )
        detected = onset_frames >= 0
        event_time = (artifact.frame_start + onset_frames[detected]) / fps
        event_ids = chord_ids[detected]
        missed_count = (~detected & artifact.chord_mask).sum()
        event_summary = f"{detected.sum()} MIDI chord onsets / {missed_count} missed chords"
    history_time = frame_time[history_start]
    history_seconds = (np.arange(len(frame_time)) - history_start + 1) / fps
    window_end = np.minimum(window_start + window_size - 1, len(chord_ids))
    rewind = np.flatnonzero(np.diff(history_start) < 0) + 1
    left = frame_time[0] if start is None else start
    right = frame_time[-1] + 1 / fps if end is None else end
    visible = (frame_time >= left) & (frame_time < right)
    if not visible.any():
        raise ValueError("requested time range contains no received frames")

    accepted_color = "#14805e"
    rejected_color = "#c94735"
    window_color = "#3677ad"
    figure, axes = plt.subplots(
        4, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.7, 1.1, 1.3]},
    )
    figure.subplots_adjust(left=0.08, right=0.92, top=0.89, bottom=0.10, hspace=0.36)
    title = input_path.stem.replace("_", " ").title()
    zoom = "" if start is None and end is None else f" | {left:g}-{right:g} s detail"
    stage = "full_song_real_time" if real_time else "full_song"
    figure.suptitle(f"{title}: {stage}{zoom}", fontsize=18, y=0.98)
    figure.text(
        0.5, 0.945,
        f"{len(frame_time) / fps:.2f} s | {len(chord_ids)} chords | "
        f"{window_size}-chord window | {event_summary}",
        ha="center", fontsize=11,
    )
    figure.text(0.5, 0.922, record_id, ha="center", fontsize=9, color="#555555")

    axis = axes[0]
    axis.fill_between(
        frame_time, window_start, window_end, step="post",
        color=window_color, alpha=0.16, label="Model chord window",
    )
    axis.step(frame_time, window_start, where="post", color=window_color, lw=1)
    axis.step(frame_time, window_end, where="post", color=window_color, lw=1)
    positive = np.any(target[valid_frames][:, valid_chords] > 0, axis=0)
    target_chords = valid_chords[positive]
    target_frames = valid_frames[np.argmax(target[valid_frames][:, target_chords], axis=0)]
    axis.plot(
        frame_time[target_frames], chord_ids[target_chords],
        color="#737b84", lw=1.1, ls="--", label="Stored target peaks", zorder=2,
    )
    if real_time:
        for mask, color, marker, label, size in (
            (success, accepted_color, "o", "DP accepted", 13),
            (~success, rejected_color, "x", "DP rejected", 27),
        ):
            axis.scatter(
                event_time[mask], event_ids[mask], c=color, marker=marker,
                s=size, linewidths=0.8, label=label, zorder=4,
            )
    else:
        axis.scatter(
            event_time, event_ids, c=accepted_color, s=13,
            label="MIDI chord onset", zorder=4,
        )
        missing = ~detected[target_chords]
        axis.scatter(
            frame_time[target_frames[missing]], chord_ids[target_chords][missing],
            c=rejected_color, marker="x", s=25, linewidths=0.8,
            label="No MIDI onset (at target peak)", zorder=4,
        )
    axis.set_ylabel("Absolute chord ID")
    axis.set_title("Chord window used for each forward pass", loc="left", fontsize=11)
    if start is None and end is None:
        axis.set_ylim(0, chord_ids[-1] + 1)
    else:
        shown_targets = (frame_time[target_frames] >= left) & (frame_time[target_frames] < right)
        shown_ids = np.concatenate((
            window_start[visible], window_end[visible], chord_ids[target_chords][shown_targets],
        ))
        axis.set_ylim(max(0, shown_ids.min() - 5), min(chord_ids[-1] + 1, shown_ids.max() + 5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axis.legend(loc="upper left", ncol=4, fontsize=9, framealpha=0.95)

    axis = axes[1]
    axis.fill_between(
        frame_time, history_time, frame_time, color=window_color,
        alpha=0.13, label="Frames supplied to the model",
    )
    axis.step(
        frame_time, history_time, where="post", color="#7557a3",
        lw=1.2, label="History start",
    )
    axis.plot(frame_time, frame_time, color="#66717e", lw=1, label="Current received frame")
    axis.scatter(
        frame_time[rewind], history_time[rewind], marker="v", s=25,
        c=rejected_color, label="History start moved backward", zorder=4,
    )
    axis.set_ylabel("Video timestamp (s)")
    axis.set_title("Video history used for each forward pass", loc="left", fontsize=11)
    axis.set_ylim(max(0, history_time[visible].min() - 1), frame_time[visible].max() + 1)
    axis.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.95)

    axis = axes[2]
    axis.step(frame_time, history_seconds, where="post", color=window_color, lw=1.25)
    axis.fill_between(frame_time, 0, history_seconds, step="post", color=window_color, alpha=0.12)
    axis.set_ylabel("History length (s)")
    axis.set_ylim(0, history_seconds[visible].max() * 1.15)
    axis.set_title(
        "History size including the current frame (received frame count / FPS)",
        loc="left", fontsize=11,
    )
    count_axis = axis.secondary_yaxis(
        "right", functions=(lambda seconds: seconds * fps, lambda count: count / fps),
    )
    count_axis.set_ylabel("Frames")
    count_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    axis = axes[3]
    if real_time:
        axis.axhline(0, color="#66717e", lw=1, ls="--")
        for mask, color, marker, size in (
            (success, accepted_color, "o", 13),
            (~success, rejected_color, "x", 27),
        ):
            axis.scatter(event_time[mask], margin[mask], c=color, marker=marker, s=size, linewidths=0.8)
        visible_events = (event_time >= left) & (event_time < right)
        visible_margin = margin[visible_events]
        axis.set_ylim(min(-0.5, visible_margin.min(initial=0) - 0.5), 1.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.set_ylabel("DP match - prior best")
        axis.set_title("Trigger decision: accepted only when the DP margin is positive", loc="left", fontsize=11)
    else:
        axis.axhline(threshold, color="#66717e", lw=1, ls="--", label=f"Threshold {threshold:g}")
        axis.scatter(event_time, onset_scores[detected], c=accepted_color, s=13)
        axis.set_ylim(max(0, threshold - 0.05), 1.02)
        axis.set_ylabel("Model onset score")
        axis.set_title(
            f"Score used by the MIDI decoder ({onset_mode}); plotted at scheduled MIDI onset",
            loc="left", fontsize=11,
        )
        axis.legend(loc="upper left", fontsize=9)
    axis.set_xlabel("Incoming video frame time (seconds)")

    for axis in axes:
        axis.set_xlim(left, right)
        axis.grid(axis="y", color="#dce1e6", lw=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    if real_time:
        note = (
            "Triggers are placed at their source-frame time; all crossed the model threshold. "
            "Green/red indicate DP acceptance/rejection.\n"
            "Window/history changes from those events apply to the next forward pass. "
            "The dashed score reference uses saved target peaks."
        )
    else:
        note = (
            "Green dots are scheduled MIDI onsets; red crosses mark chords with no MIDI onset "
            "(placed at their stored target peaks).\n"
            "Window/history traces are the actual inputs to each forward pass. "
            "Full-song MIDI decoding has no DP acceptance/rejection."
        )
    figure.text(0.08, 0.035, note, fontsize=9, color="#555555")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if start is None and end is None else f"_{left:g}-{right:g}s"
    outputs = []
    for extension in ("png", "svg"):
        output = output_dir / f"{input_path.stem}.trace{suffix}.{extension}"
        figure.savefig(output, dpi=170, facecolor="white")
        outputs.append(output)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float, help="First source-frame time to display (seconds).")
    parser.add_argument("--end", type=float, help="Last source-frame time to display (seconds).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Baseline MIDI decoding threshold.")
    parser.add_argument("--onset-mode", choices=("first_threshold", "peak"), default="first_threshold")
    parser.add_argument("--target-radius-seconds", type=float, default=0.2)
    args = parser.parse_args()
    for path in args.artifacts:
        for output in visualize_trace(
            path, args.output_dir or path.parent, args.start, args.end,
            threshold=args.threshold, onset_mode=args.onset_mode,
            target_radius_seconds=args.target_radius_seconds,
        ):
            print(f"wrote {output}")


if __name__ == "__main__":
    main()
