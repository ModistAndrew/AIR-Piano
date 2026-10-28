"""Compare the requested Bach aligned-tracking and tracking-failure intervals."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airpiano.render.artifact import load_artifact


def load_case(root: Path, person: str, start: float, end: float) -> dict:
    path = root / f"{person}_tabletop_aligned" / "bach_prelude.npz"
    artifact = load_artifact(path, require_releases=True)
    with np.load(path, allow_pickle=False) as payload:
        trace = {key: payload[key].copy() for key in (
            "frame_time", "frame_chord_window_start", "frame_video_history_start",
            "window_size", "event_frame_index", "event_chord_id", "event_time",
            "event_success", "event_match_score", "event_old_best",
        )}
    valid_frames = np.flatnonzero(artifact.frame_mask)
    values = artifact.target[valid_frames]
    reference = (artifact.frame_start + valid_frames[
        values.argmax(axis=0)
    ]) / artifact.fps
    valid_chords = artifact.chord_mask & np.any(values > 0, axis=0)
    time = trace["frame_time"]
    event_x = time[trace["event_frame_index"]]
    selected_events = (event_x >= start) & (event_x <= end)
    accepted = trace["event_success"]
    errors = trace["event_time"] - reference[trace["event_chord_id"] - 1]
    cohort = valid_chords & (reference >= start) & (reference <= end)
    release_errors = artifact.release_time - reference
    timely = cohort & np.isfinite(release_errors) & (
        np.abs(release_errors) <= 16 / artifact.fps
    )
    history_length = (
        np.arange(len(time)) - trace["frame_video_history_start"] + 1
    ) / artifact.fps
    visible = (time >= start) & (time <= end)
    return {
        "path": path, "artifact": artifact, "trace": trace,
        "start": start, "end": end, "time": time, "reference": reference,
        "valid_chords": valid_chords, "event_x": event_x,
        "event_errors": errors, "selected_events": selected_events,
        "cohort": cohort, "history_length": history_length, "visible": visible,
        "stats": {
            "record_id": artifact.record_id, "image_interval": [start, end],
            "accepted_events": int((selected_events & accepted).sum()),
            "rejected_events": int((selected_events & ~accepted).sum()),
            "median_accepted_onset_error_seconds": float(
                np.median(errors[selected_events & accepted])
            ),
            "history_seconds_min": float(history_length[visible].min()),
            "history_seconds_max": float(history_length[visible].max()),
            "timely_chords_at_16_frames": int(timely.sum()),
            "target_chords_in_interval": int(cohort.sum()),
            "target_chords_never_released": int(
                (cohort & ~np.isfinite(artifact.release_time)).sum()
            ),
        },
    }


def render_comparison(root: Path, output_dir: Path) -> list[Path]:
    cases = [
        load_case(root, "huiran", 56, 62),
        load_case(root, "venkat", 39, 48),
    ]
    cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    green, red, blue = "#39cf85", "#f06a66", "#548fb6"
    with plt.rc_context({
        "font.size": 11, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    }):
        figure, axes = plt.subplots(2, 1, figsize=(13, 8.5))
        figure.subplots_adjust(
            left=0.075, right=0.88, top=0.92, bottom=0.075, hspace=0.27,
        )
        cmap = matplotlib.colormaps["magma"].with_extremes(bad="#d5d8dc")
        for axis, case in zip(axes, cases):
            time, trace, artifact = case["time"], case["trace"], case["artifact"]
            ids = artifact.chord_ids
            visible = case["visible"]
            frame_times = time[visible]
            window_start = trace["frame_chord_window_start"][visible]
            window_end = window_start + int(trace["window_size"]) - 1
            observed = (
                (ids[None, :] >= window_start[:, None])
                & (ids[None, :] <= window_end[:, None])
                & artifact.frame_mask[visible, None]
                & artifact.chord_mask[None, :]
            )
            scores = np.ma.array(artifact.probability[visible], mask=~observed)
            half_frame = 0.5 / artifact.fps
            time_edges = np.r_[frame_times - half_frame, frame_times[-1] + half_frame]
            chord_edges = np.r_[ids - 0.5, ids[-1] + 0.5]
            background = axis.pcolormesh(
                time_edges, chord_edges, scores.T,
                cmap=cmap, vmin=0, vmax=1, shading="flat",
                rasterized=True,
            )
            # Share cell edges so the outline exactly follows the heatmap mask.
            axis.stairs(
                window_start - 0.5, time_edges, baseline=None,
                color=blue, lw=1.1, label="Score window",
            )
            axis.stairs(
                window_end + 0.5, time_edges, baseline=None, color=blue, lw=1.1,
            )
            valid = case["valid_chords"]
            axis.scatter(
                case["reference"][valid], ids[valid],
                facecolors="#6ec7ef", edgecolors="#17364a",
                marker="o", s=22, linewidths=0.6,
                label="Target onset", zorder=3,
            )
            selected = case["selected_events"]
            accepted = trace["event_success"]
            accept_mask = selected & accepted
            reject_mask = selected & ~accepted
            outline = [
                path_effects.Stroke(linewidth=2.2, foreground="#172126"),
                path_effects.Normal(),
            ]
            axis.scatter(
                case["event_x"][accept_mask], trace["event_chord_id"][accept_mask],
                facecolors="none", edgecolors=green, marker="o", s=36,
                linewidths=1.4, path_effects=outline,
                label="DP accepted", zorder=4,
            )
            axis.scatter(
                case["event_x"][reject_mask], trace["event_chord_id"][reject_mask],
                color=red, marker="x", s=30, linewidths=1.4,
                path_effects=outline, label="DP rejected", zorder=5,
            )
            visible_ids = np.concatenate((
                window_start, window_end, ids[case["cohort"]],
                trace["event_chord_id"][selected],
            ))
            lower = np.floor(visible_ids.min() / 10) * 10
            upper = np.ceil(visible_ids.max() / 10) * 10
            axis.set(
                xlim=(case["start"], case["end"]), ylim=(lower - 2, upper + 2),
                xlabel="Frame time (s)", ylabel="Chord ID",
            )
            axis.set_yticks(np.arange(lower, upper + 1, 10))

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.49, 0.992),
            ncol=4, frameon=False,
        )
        color_axis = figure.add_axes([0.91, 0.25, 0.014, 0.50])
        colorbar = figure.colorbar(background, cax=color_axis, ticks=[0, 0.5, 1])
        colorbar.set_label("Predicted probability")

        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"bach_success_vs_failure.{ext}"
                 for ext in ("png", "svg", "pdf")]
        for path in paths:
            figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)

    failure = cases[1]
    evidence = {
        "skip_score": -1,
        "cases": [
            {**case["stats"], "source_npz": str(case["path"].resolve())}
            for case in cases
        ],
        "failure_events": [
            {
                "chord_id": int(failure["trace"]["event_chord_id"][i]),
                "frame_time": float(failure["event_x"][i]),
                "estimated_onset": float(failure["trace"]["event_time"][i]),
                "target_onset": float(failure["reference"][
                    failure["trace"]["event_chord_id"][i] - 1
                ]),
                "onset_error_seconds": float(failure["event_errors"][i]),
                "accepted": bool(failure["trace"]["event_success"][i]),
                "dp_margin": float(
                    failure["trace"]["event_match_score"][i]
                    - failure["trace"]["event_old_best"][i]
                ),
            }
            for i in np.flatnonzero(failure["selected_events"])
        ],
    }
    (output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/airpiano/full_song_real_time/air_tabletop/AIR"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path("outputs/airpiano") / (
        "bach_case_comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    for path in render_comparison(args.root, output):
        print(path.resolve(), flush=True)
