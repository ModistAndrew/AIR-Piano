"""Measure GT time-shift sensitivity without changing saved releases or labels.

Positive shifts move physical GT onsets later. Original evaluated chord sets
are retained, including boundary chords; missing releases remain failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airpiano.evaluation import (
    _alignment_observations,
    _metrics_from_observations,
    aggregate_alignment_metrics,
)
from airpiano.render.artifact import load_artifact
from airpiano.utils import write_json


TOLERANCES = (2, 4, 8, 16)


def check_equal(actual, expected):
    """Require all original aggregate metrics to be reproduced."""
    for key, value in expected.items():
        if isinstance(value, dict):
            check_equal(actual[key], value)
        elif value is None:
            assert actual[key] is None, key
        else:
            np.testing.assert_allclose(actual[key], value, rtol=1e-10, atol=1e-10,
                                       err_msg=key)


def load_records(path):
    source = json.loads(path.read_text())
    records, baseline = [], []
    for entry in source["artifacts"]:
        artifact = load_artifact(Path(source["artifact_root"]) / entry["artifact_path"],
                                 require_releases=True)
        correct, observations = _alignment_observations(artifact)
        metrics = _metrics_from_observations(artifact, correct, observations, TOLERANCES)
        check_equal(metrics, entry["metrics"])
        baseline.append(metrics)
        slots = np.array(list(observations), dtype=int)
        errors = np.array([np.nan if e is None else e for e in observations.values()])
        frames = np.flatnonzero(artifact.frame_mask)
        target = artifact.physical_target if artifact.physical_target is not None else artifact.target
        gt = artifact.frame_start + frames[target[frames][:, slots].argmax(axis=0)]
        records.append({
            "record_id": artifact.record_id,
            "fps": artifact.fps,
            "slots": slots,
            "gt_seconds": gt / artifact.fps,
            "errors": errors,
            "release_seconds": artifact.release_time[slots],
            "predicted_seconds": artifact.predicted_onset[slots],
            "onset_prediction_offset": artifact.onset_prediction_offset,
        })
    micro, macro = aggregate_alignment_metrics(baseline, TOLERANCES)
    check_equal(micro, source["micro"])
    check_equal(macro, source["macro"])
    print(f"{path.stem}: reproduced every per-record and aggregate baseline metric", flush=True)
    return records, source


def summarize(records, shift_ms):
    errors = [r["errors"] - shift_ms * r["fps"] / 1000 for r in records]
    detected = [e[np.isfinite(e)] for e in errors]
    all_frames = np.concatenate(detected)
    all_ms = np.concatenate([e * 1000 / r["fps"] for e, r in zip(detected, records)])
    total = sum(len(e) for e in errors)
    counts = {str(t): sum(int((np.abs(e) <= t).sum()) for e in detected) for t in TOLERANCES}
    return {
        "shift_ms": float(shift_ms),
        "micro": {
            "c_ar": {str(t): counts[str(t)] / total for t in TOLERANCES},
            "correct_chord_counts": counts,
            "ground_truth_chord_count": total,
            "detected_chord_count": len(all_frames),
            "chord_coverage": len(all_frames) / total,
            "onset_mae_ms": float(np.abs(all_ms).mean()),
            "onset_mean_delta_ms": float(all_ms.mean()),
            "onset_median_delta_ms": float(np.median(all_ms)),
        },
        "macro": {
            "c_ar": {str(t): float(np.mean([(np.abs(e) <= t).sum() / len(raw)
                                            for e, raw in zip(detected, errors)]))
                     for t in TOLERANCES},
            "chord_coverage": float(np.mean([len(e) / len(raw) for e, raw in zip(detected, errors)])),
            "onset_mae_ms": float(np.mean([np.abs(e).mean() * 1000 / r["fps"]
                                           for e, r in zip(detected, records)])),
        },
        "detected_error_quantiles_ms": {
            str(q): float(np.percentile(all_ms, q)) for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
        },
        "detected_fraction_late": float(np.mean(all_ms > 0)),
        "detected_fraction_abs_error_over_1s": float(np.mean(np.abs(all_ms) > 1000)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(datasets, result, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    colors = ("#247ba0", "#db5e36")
    for (name, records), color in zip(datasets.items(), colors):
        errors = np.concatenate([r["errors"] * 1000 / r["fps"] for r in records])
        finite = errors[np.isfinite(errors)]
        bins = np.arange(-500, 1010, 1000 / 30)
        axes[0].hist(finite, bins=bins, weights=np.full(len(finite), 100 / len(finite)),
                     histtype="step", linewidth=1.8, label=name, color=color)
        for axis, tolerance in zip(axes[1:], (2, 4)):
            sweep = result["datasets"][name]["sweep"]
            axis.plot([x["shift_ms"] for x in sweep],
                      [100 * x["micro"]["c_ar"][str(tolerance)] for x in sweep],
                      label=name, color=color, linewidth=2)
            axis.axvline(100, color="#777777", linestyle=":", linewidth=1)
            axis.set(title=f"C-AR {tolerance}: ±{1000*tolerance/30:.1f} ms at 30 fps",
                     xlabel="GT shift (ms; positive = later)", ylabel="All GT chords correct (%)",
                     ylim=(0, 100))
    axes[0].set(title="Release time − physical GT onset",
                xlabel="Error (ms; positive = late release)", ylabel="Detected chords per bin (%)")
    axes[0].text(.98, .97, "Tails beyond plot omitted", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9)
    for axis in axes:
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
        axis.legend(fontsize=8)
    fig.suptitle("Fixed-release GT shift sensitivity · baseline verified against saved JSON", fontsize=13)
    fig.savefig(output / "gt_shift_comparison.png", dpi=170)
    fig.savefig(output / "gt_shift_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Whole-frame shifts avoid putting the half-frame release errors exactly on
    # integer tolerance boundaries, where roundoff can change inclusive counts.
    shifts = np.arange(-15, 16) * (1000 / 30)
    result = {
        "method": "Fixed-release sensitivity analysis, not inference or training rerun. "
                  "GT_shifted = physical_GT + shift_ms/1000; error_shifted = error - shift. "
                  "Fractional release frames, original chord membership and missing releases preserved. "
                  "Sweep uses whole-frame shifts at 30 fps, from -500 to +500 ms. "
                  "Best sweep shifts are descriptive fits to this evaluation set, not held-out calibration. "
                  "Frame chord_acc is not recomputed for shifted conditions.",
        "datasets": {},
    }
    datasets, sweep_rows, record_rows, chord_rows = {}, [], [], []
    for path in args.evaluations:
        name = path.stem
        records, source = load_records(path)
        datasets[name] = records
        sweep = [summarize(records, shift) for shift in shifts]
        baseline = summarize(records, 0)
        selected = [summarize(records, s) for s in (-200, -100, 0, 100, 200, 300)]
        for averaging in ("micro", "macro"):
            check_equal(baseline[averaging]["c_ar"], source[averaging]["c_ar"])
        result["datasets"][name] = {
            "source_json": str(path.resolve()),
            "baseline_verified": True,
            "fps_values": sorted({r["fps"] for r in records}),
            "baseline": baseline, "selected_shifts": selected, "sweep": sweep,
            "best_micro_c_ar_shifts": {
                str(t): max(sweep, key=lambda x: (x["micro"]["c_ar"][str(t)], -abs(x["shift_ms"])))
                for t in TOLERANCES
            },
        }
        for row in sweep:
            sweep_rows.append({"dataset": name, "shift_ms": row["shift_ms"],
                               **{f"{avg}_c_ar_{t}": row[avg]["c_ar"][str(t)]
                                  for avg in ("micro", "macro") for t in TOLERANCES},
                               "micro_mae_ms": row["micro"]["onset_mae_ms"]})
        for record in records:
            before, after = summarize([record], 0), summarize([record], 100)
            record_rows.append({"dataset": name, "record_id": record["record_id"],
                                **{k: v for k, v in before["micro"].items() if not isinstance(v, dict)},
                                **{f"c_ar_{t}_shift_{shift}_ms": item["micro"]["c_ar"][str(t)]
                                   for shift, item in ((0, before), (100, after)) for t in TOLERANCES}})
            for i, slot in enumerate(record["slots"]):
                chord_rows.append({"dataset": name, "record_id": record["record_id"],
                                   "chord_slot": int(slot), "gt_seconds": record["gt_seconds"][i],
                                   "release_seconds": record["release_seconds"][i],
                                   "predicted_onset_seconds": record["predicted_seconds"][i],
                                   "error_ms": record["errors"][i] * 1000 / record["fps"]})
        print(name, json.dumps({"baseline": baseline["micro"],
                               "plus_100ms": summarize(records, 100)["micro"]}), flush=True)
    write_json(args.output_dir / "gt_shift_analysis.json", result)
    write_csv(args.output_dir / "shift_sweep.csv", sweep_rows)
    write_csv(args.output_dir / "per_record.csv", record_rows)
    write_csv(args.output_dir / "per_chord.csv", chord_rows)
    plot_results(datasets, result, args.output_dir)
    print(f"Wrote analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
