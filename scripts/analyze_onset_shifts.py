"""Audit saved release timing and fixed-offset sensitivity without rerunning inference.

Positive errors/biases mean late releases. A positive correction is subtracted
from release times (advances them). Corrections are retrospective diagnostics,
not measurements of physical A/V synchronization or deployable latency savings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


DATASETS = (
    "air_piano", "air_tabletop", "air_tabletop_valid", "miditest",
    "pianomotion10m30s", "pianomotion10m60s", "pianomotion10m120s",
)
# Include half-frame offsets to capture inclusive tolerance-boundary maxima.
BIAS_MS = np.unique(np.concatenate((np.arange(-500, 501, dtype=float), np.arange(-30, 31) * (1000 / 60))))


def rate(errors, fps, total, bias_ms=0.0):
    return float(np.count_nonzero(np.abs(errors - bias_ms / 1000) <= 4 / fps + 1e-10) / total) if total else None


def stats(values):
    if not len(values):
        return {}
    q = np.percentile(values, [10, 25, 50, 75, 90]) * 1000
    return dict(zip(("p10_ms", "p25_ms", "median_ms", "p75_ms", "p90_ms"), map(float, q)),
                mean_ms=float(np.mean(values) * 1000), mae_ms=float(np.mean(np.abs(values)) * 1000),
                centered_mae_ms=float(np.mean(np.abs(values - np.median(values))) * 1000),
                mad_ms=float(np.median(np.abs(values - np.median(values))) * 1000))


def sweep(errors, fps, total):
    return np.array([rate(errors, fps, total, b) for b in BIAS_MS])


def best_bias(curve, median_ms):
    candidates = np.flatnonzero(curve == curve.max())
    index = candidates[np.argmin(np.abs(BIAS_MS[candidates] - median_ms))]
    return float(BIAS_MS[index]), float(curve[index])


def read_record(dataset, manifest, item):
    path = Path(manifest["artifact_root"]) / item["artifact_path"]
    with np.load(path, allow_pickle=False) as z:
        fps, start = float(z["fps"]), int(z["frame_start"])
        frames = np.flatnonzero(z["frame_mask"])
        slots = np.flatnonzero(z["chord_mask"])
        target_name = "physical_target" if "physical_target" in z else "target"
        target = z[target_name][np.ix_(frames, slots)]
        slots = slots[np.any(target > 0, axis=0)]
        target = z[target_name][np.ix_(frames, slots)]
        reference = (start + frames[np.argmax(target, axis=0)]) / fps
        release, prediction = z["release_time"][slots], z["predicted_onset"][slots]
        detected = np.isfinite(release)
        errors = release[detected] - reference[detected]
        prediction_errors = prediction[detected] - reference[detected]
        offset = float(z["onset_prediction_offset"])
        trigger_errors = prediction_errors - offset
        # Compare in frames exactly as evaluation.py does, before unit conversion.
        frame_errors = release[detected] * fps - reference[detected] * fps
        cached = item["metrics"]
        assert len(errors) == cached["detected_chord_count"], str(path)
        assert len(slots) == cached["ground_truth_chord_count"], str(path)
        for tolerance in (4, 8, 16):
            assert int(np.count_nonzero(np.abs(frame_errors) <= tolerance)) == cached["correct_chord_counts"][str(tolerance)], str(path)
        assert np.isclose(np.mean(errors), cached["onset_mean_delta_seconds"], atol=1e-8), str(path)
        midpoint = (start + len(z["frame_mask"]) / 2) / fps
        first = reference[detected] < midpoint
        second = ~first
        first_bias = float(np.median(errors[first]) * 1000) if np.any(first) else None
        second_gt_count = int(np.count_nonzero(reference >= midpoint))
        row = dict(dataset=dataset, record_id=item["record_id"], fps=fps,
                   target_source=target_name, ground_truth_count=len(slots), detected_count=len(errors),
                   coverage=len(errors) / len(slots), **stats(errors),
                   positive_fraction=float(np.mean(errors > 0)),
                   abs_error_over_500ms_fraction=float(np.mean(np.abs(errors) > .5)),
                   c_ar_4=cached["c_ar"]["4"], c_ar_8=cached["c_ar"]["8"], c_ar_16=cached["c_ar"]["16"],
                   c_ar_4_advance_100ms=rate(errors, fps, len(slots), 100),
                   c_ar_4_delay_100ms=rate(errors, fps, len(slots), -100),
                   c_ar_4_without_hold_fixed_triggers=rate(trigger_errors + float(z["processing_latency"]), fps, len(slots)),
                   c_ar_4_piece_median_oracle=rate(errors, fps, len(slots), float(np.median(errors) * 1000)),
                   predicted_onset_median_ms=float(np.median(prediction_errors) * 1000),
                   trigger_median_ms=float(np.median(trigger_errors) * 1000),
                   onset_prediction_offset_ms=offset * 1000,
                   processing_latency_ms=float(z["processing_latency"]) * 1000,
                   scheduler_median_ms=float(np.median(release[detected] - prediction[detected]) * 1000),
                   first_half_median_ms=first_bias,
                   second_half_median_ms=float(np.median(errors[second]) * 1000) if np.any(second) else None,
                   second_half_ground_truth_count=second_gt_count,
                   second_half_c_ar_4=rate(errors[second], fps, second_gt_count),
                   second_half_c_ar_4_first_half_calibration=rate(errors[second], fps, second_gt_count, first_bias) if first_bias is not None else None,
                   artifact_path=str(path))
        row["half_median_change_ms"] = row["second_half_median_ms"] - first_bias if first_bias is not None and row["second_half_median_ms"] is not None else None
        curve = sweep(errors, fps, len(slots))
        row["best_bias_ms_within_500ms"], row["best_c_ar_4_within_500ms"] = best_bias(curve, row["median_ms"])
        return row, dict(errors=errors, reference=reference[detected], slots=slots[detected],
                         all_reference=reference, fps=fps, total=len(slots), curve=curve,
                         prediction_errors=prediction_errors, trigger_errors=trigger_errors)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output, rows, groups, summaries):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/airpiano-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(DATASETS, ("#2878b5", "#d05b36", "#bc8a19", "#8b52ad", "#54a56b", "#21896d", "#145c49")))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for name, data in groups.items():
        errors = np.concatenate([a["errors"] for a in data])
        axes[0, 0].hist(errors * 1000, bins=np.arange(-601, 602, 100 / 3), weights=np.ones(len(errors)) / len(errors), histtype="step", label=name, color=colors[name], linewidth=1.5)
        curve = np.average([a["curve"] for a in data], axis=0, weights=[a["total"] for a in data])
        axes[0, 1].plot(BIAS_MS, curve * 100, label=name, color=colors[name])
    axes[0, 0].set(xlabel="Release − reference onset (ms); tails outside plot omitted", ylabel="Fraction of detected chords", title="Onset error distributions")
    axes[0, 1].set(xlabel="Bias subtracted from releases (ms); positive = earlier", ylabel="C-AR@4 (%)", title="Retrospective fixed-offset sensitivity")
    for ax in axes[0]:
        ax.axvline(0, color="black", lw=.7, alpha=.5)
        ax.legend(fontsize=8)
    values = [[r["median_ms"] for r in rows if r["dataset"] == name] for name in groups]
    axes[1, 0].boxplot(values, labels=list(groups), vert=False, showfliers=False)
    for i, (name, vals) in enumerate(zip(groups, values), 1):
        axes[1, 0].scatter(vals, np.full(len(vals), i), s=9, alpha=.6, c=colors[name])
    axes[1, 0].set(xlabel="Per-piece median release error (ms)", xlim=(-500, 500), title="Typical offset differs across domains (extremes clipped)")
    x = np.arange(len(summaries)); width = .25
    for dx, key, label in ((-width, "c_ar_4", "Original"), (0, "c_ar_4_advance_100ms", "Advance 100 ms"), (width, "c_ar_4_leave_one_piece_out", "Median of other pieces")):
        axes[1, 1].bar(x + dx, [100 * s[key] for s in summaries], width, label=label)
    axes[1, 1].set_xticks(x, [s["dataset"] for s in summaries], rotation=30, ha="right", fontsize=8)
    axes[1, 1].set(ylabel="C-AR@4 (%)", title="Correction effects; missed chords remain failures", ylim=(0, 100))
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output / "overview.png", dpi=180)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)

    for name in groups:
        records = [r for r in rows if r["dataset"] == name]
        ncols = 4 if len(records) > 10 else 2
        nrows = (len(records) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.4), squeeze=False, layout="constrained")
        for ax, row, data in zip(axes.flat, records, groups[name]):
            ax.scatter(data["reference"], data["errors"] * 1000, s=4, alpha=.65, color=colors[name])
            ax.axhline(0, color="black", lw=.6)
            ax.axhspan(-4000 / data["fps"], 4000 / data["fps"], color="#aaaaaa", alpha=.18)
            ax.axhline(row["median_ms"], color="#c33", lw=.8)
            ax.set_title(row["record_id"].split("/", 1)[1].replace("_aligned", "") + f"\nmedian {row['median_ms']:.0f} ms | coverage {row['coverage']:.0%}", fontsize=8)
            ax.set_xlabel("Reference onset (s)", fontsize=7)
            ax.set_ylabel("Error (ms)", fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(records):]:
            ax.axis("off")
        fig.savefig(output / f"{name}_pieces.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/airpiano/full_song_real_time"))
    parser.add_argument("--output", type=Path, default=Path("outputs/airpiano/onset_shift_analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, groups, summaries, chord_rows = [], {}, [], []
    for name in DATASETS:
        manifest = json.loads((args.root / f"{name}.json").read_text())
        records, data = [], []
        for item in manifest["artifacts"]:
            row, values = read_record(name, manifest, item)
            records.append(row); data.append(values)
            for slot, ref, err in zip(values["slots"], values["reference"], values["errors"]):
                chord_rows.append(dict(dataset=name, record_id=row["record_id"], chord_slot=int(slot), reference_seconds=float(ref), release_error_ms=float(err * 1000)))
        groups[name] = data
        errors = np.concatenate([a["errors"] for a in data])
        total = sum(a["total"] for a in data)
        fps = np.concatenate([np.full(len(a["errors"]), a["fps"]) for a in data])
        curve = np.average([a["curve"] for a in data], axis=0, weights=[a["total"] for a in data])
        global_bias = float(np.median(errors) * 1000)
        for index, (row, a) in enumerate(zip(records, data)):
            other_errors = np.concatenate([b["errors"] for j, b in enumerate(data) if j != index])
            bias = float(np.median(other_errors) * 1000)
            row["leave_one_piece_out_bias_ms"] = bias
            row["c_ar_4_leave_one_piece_out"] = rate(a["errors"], a["fps"], a["total"], bias)
        summary = dict(dataset=name, piece_count=len(data), ground_truth_count=total, detected_count=len(errors), coverage=len(errors) / total,
                       **stats(errors), c_ar_4=manifest["micro"]["c_ar"]["4"],
                       c_ar_4_advance_100ms=rate(errors, fps, total, 100),
                       c_ar_4_delay_100ms=rate(errors, fps, total, -100),
                       c_ar_4_without_hold_fixed_triggers=sum(r["c_ar_4_without_hold_fixed_triggers"] * r["ground_truth_count"] for r in records) / total,
                       c_ar_4_dataset_median_oracle=rate(errors, fps, total, global_bias),
                       c_ar_4_piece_median_oracle=sum(r["c_ar_4_piece_median_oracle"] * r["ground_truth_count"] for r in records) / total,
                       c_ar_4_leave_one_piece_out=sum(r["c_ar_4_leave_one_piece_out"] * r["ground_truth_count"] for r in records) / total,
                       macro_c_ar_4=manifest["macro"]["c_ar"]["4"], macro_mean_ms=manifest["macro"]["onset_mean_delta_seconds"] * 1000,
                       piece_median_p10_ms=float(np.percentile([r["median_ms"] for r in records], 10)),
                       piece_median_p90_ms=float(np.percentile([r["median_ms"] for r in records], 90)),
                       median_within_piece_iqr_ms=float(np.median([r["p75_ms"] - r["p25_ms"] for r in records])),
                       positive_fraction=float(np.mean(errors > 0)), abs_error_over_500ms_fraction=float(np.mean(np.abs(errors) > .5)),
                       predicted_onset_median_ms=float(np.median(np.concatenate([a["prediction_errors"] for a in data])) * 1000),
                       trigger_median_ms=float(np.median(np.concatenate([a["trigger_errors"] for a in data])) * 1000))
        summary["best_bias_ms_within_500ms"], summary["best_c_ar_4_within_500ms"] = best_bias(curve, global_bias)
        temporal = [r for r in records if r["second_half_c_ar_4_first_half_calibration"] is not None]
        temporal_n = sum(r["second_half_ground_truth_count"] for r in temporal)
        for key in ("second_half_c_ar_4", "second_half_c_ar_4_first_half_calibration"):
            summary[key] = sum(r[key] * r["second_half_ground_truth_count"] for r in temporal) / temporal_n
        assert np.isclose(summary["mean_ms"], manifest["micro"]["onset_mean_delta_seconds"] * 1000, atol=1e-7)
        summaries.append(summary); rows.extend(records)
        print(name, json.dumps(summary), flush=True)
    supplemental = {"pianomotion_duration_pairs": [], "mean_outlier_sensitivity": [], "half_median_stability": []}
    for name in DATASETS:
        selected = [r for r in rows if r["dataset"] == name]
        signed = [r["mean_ms"] * r["detected_count"] for r in selected]
        largest = int(np.argmax(np.abs(signed)))
        rest_n = sum(r["detected_count"] for i, r in enumerate(selected) if i != largest)
        supplemental["mean_outlier_sensitivity"].append(dict(
            dataset=name, largest_absolute_signed_contributor=selected[largest]["record_id"],
            contributor_fraction_of_signed_sum=float(signed[largest] / sum(signed)),
            pooled_mean_ms_without_that_piece=float((sum(signed) - signed[largest]) / rest_n)))
        changes = [abs(r["half_median_change_ms"]) for r in selected if r["half_median_change_ms"] is not None]
        supplemental["half_median_stability"].append(dict(
            dataset=name, evaluated_pieces=len(changes),
            within_one_frame=sum(x <= 1000 / 30 + 1e-7 for x in changes),
            within_two_frames=sum(x <= 2000 / 30 + 1e-7 for x in changes)))
    for left, right in (("pianomotion10m30s", "pianomotion10m60s"), ("pianomotion10m30s", "pianomotion10m120s"), ("pianomotion10m60s", "pianomotion10m120s")):
        a = {r["record_id"].split("/", 1)[1]: r["median_ms"] for r in rows if r["dataset"] == left}
        b = {r["record_id"].split("/", 1)[1]: r["median_ms"] for r in rows if r["dataset"] == right}
        pairs = np.array([(a[k], b[k]) for k in sorted(a.keys() & b.keys()) if abs(a[k]) <= 500 and abs(b[k]) <= 500])
        supplemental["pianomotion_duration_pairs"].append(dict(
            datasets=[left, right], selection="Shared source ID; absolute piece median <=500ms in both crops", count=len(pairs),
            pearson_correlation=float(np.corrcoef(pairs.T)[0, 1]),
            median_absolute_difference_ms=float(np.median(np.abs(pairs[:, 0] - pairs[:, 1]))),
            p90_absolute_difference_ms=float(np.percentile(np.abs(pairs[:, 0] - pairs[:, 1]), 90))))
    (args.output / "supplemental.json").write_text(json.dumps(supplemental, indent=2))
    write_csv(args.output / "per_piece.csv", rows)
    write_csv(args.output / "per_chord.csv", chord_rows)
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "shift_sweep.csv", [dict(bias_subtracted_ms=float(b), **{name: float(np.average([a["curve"][i] for a in data], weights=[a["total"] for a in data])) for name, data in groups.items()}) for i, b in enumerate(BIAS_MS)])
    (args.output / "analysis.json").write_text(json.dumps(dict(
        methodology=dict(error="release_time minus physical_target peak (or target if absent)",
                         sign="Positive is late; positive bias is SUBTRACTED to advance releases",
                         c_ar_4="All reference chords in denominator; missed releases remain failures",
                         grid="Integer milliseconds and 30 Hz half-frame offsets from -500 to +500 inclusive",
                         oracle="Dataset/per-piece median and best offset use evaluation labels; diagnostic only",
                         loo="For each piece, use pooled median of detected errors from other pieces in same dataset",
                         temporal="Fit median on first half of wall-clock duration, evaluate only second half",
                         no_hold="Fixed recorded triggers plus saved processing_latency; removing the hold changes the controller feedback in a real rerun, so this is only a counterfactual",
                         boundary="Use 1e-10 seconds epsilon for inclusive tolerance boundaries in offset experiments; baseline counts verified exactly in frames",
                         verification="All per-piece counts at 4/8/16 frames and signed means reproduce input JSON"),
        summaries=summaries, pieces=rows), indent=2))
    make_plots(args.output, rows, groups, summaries)


if __name__ == "__main__":
    main()
