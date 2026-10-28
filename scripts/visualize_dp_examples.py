"""Render selected DP rejection examples and their complete saved traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airpiano.evaluation import calculate_alignment_metrics
from airpiano.render.artifact import load_artifact
from scripts.visualize_real_time import visualize_trace


def render_examples(selection_path: Path, output_dir: Path) -> None:
    selections = json.loads(selection_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "airpiano-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    green, red = "#168250", "#d04740"
    figure, axes = plt.subplots(
        len(selections), 2, figsize=(15, 4 * len(selections) + 1.5),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.07, right=0.97, bottom=0.12, top=0.88,
        hspace=0.62, wspace=0.20,
    )
    figure.suptitle(
        "DP keeps large timing outliers out of the anchor estimates\n"
        "Selected intervals from three recordings with accurate overall MIDI timing",
        fontsize=16, y=0.98,
    )
    report = []
    for row, case in enumerate(selections):
        path = Path(case["path"])
        artifact = load_artifact(path, require_releases=True)
        with np.load(path, allow_pickle=False) as payload:
            trace = {key: payload[key].copy() for key in (
                "event_time", "event_frame_index", "event_chord_id",
                "event_success", "event_chord_window_start",
                "event_match_score", "event_old_best", "window_size",
            )}
        valid_frames = np.flatnonzero(artifact.frame_mask)
        target = artifact.target[valid_frames]
        reference = (artifact.frame_start + valid_frames[
            target.argmax(axis=0)
        ]) / artifact.fps
        valid_chords = artifact.chord_mask & np.any(target > 0, axis=0)
        ids = trace["event_chord_id"]
        times = trace["event_time"]
        success = trace["event_success"]
        errors = times - reference[ids - 1]
        visible = (times >= case["start"]) & (times <= case["end"])
        reference_visible = valid_chords & (reference >= case["start"]) & (
            reference <= case["end"]
        )
        focus = case["focus_event_index"]
        assert not success[focus]
        assert visible[focus]
        # Use the existing evaluator; six frames at the saved 30 fps = 200 ms.
        assert artifact.fps == 30
        metrics = calculate_alignment_metrics(
            artifact, tolerances_frames=(4, 6, 8),
        )
        actual_start = int(trace["event_chord_window_start"][focus])
        previous_start = 1 if focus == 0 else int(
            trace["event_chord_window_start"][focus - 1]
        )
        if_accepted_start = int(np.clip(
            ids[focus] - int(trace["window_size"]) // 2,
            1, len(artifact.chord_ids) - int(trace["window_size"]) + 1,
        ))
        timing = "early" if errors[focus] < 0 else "late"
        label = f'ID {ids[focus]}: {abs(errors[focus]):.2f} s {timing}'
        axis = axes[row, 0]
        axis.plot(reference[reference_visible],
                  artifact.chord_ids[reference_visible],
                  color="#777777", ls="--", lw=1.2, label="Target onset")
        for mask, color, marker, name in (
            (visible & success, green, "o", "DP accepted"),
            (visible & ~success, red, "x", "DP rejected"),
        ):
            axis.scatter(times[mask], ids[mask], color=color,
                         marker=marker, s=28, label=name, zorder=3)
        axis.scatter([times[focus]], [ids[focus]], s=100,
                     facecolors="none", edgecolors=red, zorder=4)
        axis.annotate(
            label, xy=(times[focus], ids[focus]),
            xytext=(0.04, 0.88 if errors[focus] < 0 else 0.12),
            textcoords="axes fraction", fontsize=10, color=red,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": red},
        )
        axis.set_title(case["label"], fontsize=12, loc="left", pad=12)
        axis.set_ylabel("Score chord ID")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        if row == 0:
            axis.legend(loc="lower right", fontsize=9)

        axis = axes[row, 1]
        axis.axhspan(-0.2, 0.2, color=green, alpha=0.10,
                     label="Within +/-200 ms")
        axis.axhline(0, color="#777777", lw=0.8)
        for mask, color, marker in (
            (visible & success, green, "o"),
            (visible & ~success, red, "x"),
        ):
            axis.scatter(times[mask], errors[mask],
                         color=color, marker=marker, s=28, zorder=3)
        extent = max(0.4, float(np.abs(errors[visible]).max())) * 1.25
        axis.set_ylim(-extent, extent)
        axis.set_ylabel("Estimated onset - target (s)")
        axis.set_title(
            f'Whole recording: {metrics["c_ar"]["6"]:.1%} within +/-200 ms\n'
            f'{metrics["detected_chord_count"]}/{metrics["ground_truth_chord_count"]}'
            f' MIDI chords released; MAE {metrics["onset_mae_seconds"] * 1000:.0f} ms',
            fontsize=11, loc="left", pad=12,
        )
        axis.text(
            0.03, 0.93 if errors[focus] < 0 else 0.07,
            f"At highlighted rejection: window start {previous_start} -> {actual_start}\n"
            f"Accepting that event would recenter at {if_accepted_start}.",
            transform=axis.transAxes, fontsize=9, va="top" if errors[focus] < 0 else "bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        for axis in axes[row]:
            axis.set_xlim(case["start"], case["end"])
            axis.set_xlabel("Estimated onset time (seconds)")
            axis.grid(alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)

        trace_dir = output_dir / case["slug"]
        paths = visualize_trace(path, trace_dir)
        paths += visualize_trace(
            path, trace_dir, start=case["start"], end=case["end"],
        )
        report.append({
            **case, "record_id": artifact.record_id,
            "metrics": metrics,
            "accepted_events": int(success.sum()),
            "rejected_events": int((~success).sum()),
            "focus_event": {
                "chord_id": int(ids[focus]),
                "frame_time": float(trace["event_frame_index"][focus] / artifact.fps),
                "estimated_onset": float(times[focus]),
                "target_onset": float(reference[ids[focus] - 1]),
                "onset_error_seconds": float(errors[focus]),
                "dp_margin": float(trace["event_match_score"][focus] - trace["event_old_best"][focus]),
                "window_before": previous_start, "window_after": actual_start,
                "window_if_accepted": if_accepted_start,
            },
            "interval_events": [
                {"index": int(i), "chord_id": int(ids[i]),
                 "estimated_onset": float(times[i]), "accepted": bool(success[i]),
                 "onset_error_seconds": float(errors[i])}
                for i in np.flatnonzero(visible)
            ],
            "trace_plots": [str(p) for p in paths],
        })
        print(case["label"], metrics["c_ar"]["6"], flush=True)

    figure.text(
        0.07, 0.025,
        "Green/red marks are DP decisions, not correctness labels. Some well-timed events are also rejected.\n"
        "Plot x = saved event.time (image timestamp + 100 ms). Timing accuracy above uses actual MIDI release_time across the whole recording.\n"
        "Rejected events do not become anchors, but can still emit MIDI. The hypothetical window value is a single update, not a no-DP replay.\n"
        "These traces show rejection and continued tracking; an end-to-end benefit over disabling DP requires a separate ablation.",
        fontsize=10, color="#444444",
    )
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"dp_success_examples.{extension}",
                       dpi=160, facecolor="white")
    plt.close(figure)
    (output_dir / "evidence.json").write_text(
        json.dumps(report, indent=2) + "\n",
    )
    print(output_dir / "dp_success_examples.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_examples(args.selection, args.output_dir)
