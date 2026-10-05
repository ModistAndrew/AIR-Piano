#!/usr/bin/env python3
"""Measure recovery after one forced forward score-window jump."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence

from airpiano import data
from airpiano.evaluation import (
    aggregate_alignment_metrics,
    calculate_alignment_metrics,
)
from airpiano.model.loss import alignment_probabilities
from airpiano.render.artifact import InferenceArtifact, load_artifact
from airpiano.train import _autocast, _forward
from airpiano.utils import resolve_path, write_json


@dataclass
class ControllerState:
    chord_start: int = 0
    history_start: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--injection-fraction", type=float, default=0.5)
    parser.add_argument("--shifts", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--stable-frames", type=int, default=30)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normal_window_starts(probability: np.ndarray) -> np.ndarray:
    represented = probability > 0.0
    if not bool(represented.any(axis=1).all()):
        raise ValueError("normal artifact has a frame with no represented chord")
    return represented.argmax(axis=1).astype(np.int64)


def choose_injection_frame(
    starts: np.ndarray,
    *,
    final_chord_start: int,
    maximum_shift: int,
    fraction: float,
    stable_frames: int,
) -> int:
    frames = np.arange(len(starts))
    eligible = frames[
        (frames > 0)
        & (frames <= len(starts) - stable_frames - 1)
        & (starts + maximum_shift <= final_chord_start)
    ]
    if not len(eligible):
        raise ValueError("song has no valid frame for the requested window jump")
    desired = int(round(fraction * len(starts)))
    return int(eligible[np.argmin(np.abs(eligible - desired))])


@torch.no_grad()
def controller_step(
    model: torch.nn.Module,
    song: data.Song,
    chord_tokens: torch.Tensor,
    states: list[ControllerState],
    frame_index: int,
    *,
    window_width: int,
    final_chord_start: int,
    device: torch.device,
    bf16: bool,
    alignment_threshold: float,
    advance_after_chords: int,
    alignment_loss: str,
) -> list[np.ndarray]:
    hand_histories = [
        song.hands[state.history_start : frame_index + 1]
        for state in states
    ]
    presence_histories = [
        song.presence[state.history_start : frame_index + 1]
        for state in states
    ]
    chord_windows = [
        chord_tokens[
            state.chord_start : min(
                state.chord_start + window_width, len(chord_tokens)
            )
        ]
        for state in states
    ]
    frame_lengths = torch.tensor(
        [len(value) for value in hand_histories], dtype=torch.long
    )
    chord_lengths = torch.tensor(
        [len(value) for value in chord_windows], dtype=torch.long
    )
    batch = {
        "hands": pad_sequence(hand_histories, batch_first=True),
        "hand_presence": pad_sequence(presence_histories, batch_first=True),
        "chords": pad_sequence(chord_windows, batch_first=True),
        "frame_mask": (
            torch.arange(int(frame_lengths.max()))[None, :]
            < frame_lengths[:, None]
        ),
        "chord_mask": (
            torch.arange(int(chord_lengths.max()))[None, :]
            < chord_lengths[:, None]
        ),
    }
    with _autocast(device, bf16):
        logits = _forward(model, batch, device)
    probabilities = alignment_probabilities(
        logits,
        batch["chord_mask"].to(device),
        alignment_loss,
    ).cpu()

    rows = []
    for index, state in enumerate(states):
        history_length = int(frame_lengths[index])
        chord_count = int(chord_lengths[index])
        history_probability = probabilities[
            index, :history_length, :chord_count
        ]
        rows.append(history_probability[-1].numpy().copy())
        if state.chord_start >= final_chord_start:
            continue
        trailing_peaks = history_probability[
            :, -advance_after_chords:
        ].amax(dim=0)
        if bool((trailing_peaks >= alignment_threshold).any()):
            predicted_start = int(torch.argmax(history_probability[:, 1]))
            state.history_start += predicted_start
            state.chord_start += 1
    return rows


def sliced_artifact(
    normal: InferenceArtifact,
    probability: np.ndarray,
    injection_frame: int,
) -> InferenceArtifact:
    return InferenceArtifact(
        probability=probability,
        target=normal.target[injection_frame:].copy(),
        frame_mask=normal.frame_mask[injection_frame:].copy(),
        chord_mask=normal.chord_mask.copy(),
        chord_pitches=normal.chord_pitches.copy(),
        chord_ids=normal.chord_ids.copy(),
        fps=normal.fps,
        record_id=normal.record_id,
        frame_start=normal.frame_start + injection_frame,
    )


def recovery_metrics(
    shifted_starts: np.ndarray,
    normal_starts: np.ndarray,
    *,
    fps: float,
    stable_frames: int,
) -> dict[str, object]:
    offsets = shifted_starts - normal_starts
    exact = offsets == 0
    recovery_index = None
    if len(exact) >= stable_frames:
        stable = np.convolve(
            exact.astype(np.int64),
            np.ones(stable_frames, dtype=np.int64),
            mode="valid",
        )
        matches = np.flatnonzero(stable == stable_frames)
        if len(matches):
            recovery_index = int(matches[0])
    permanent_index = None
    nonzero = np.flatnonzero(~exact)
    candidate = int(nonzero[-1] + 1) if len(nonzero) else 0
    if len(exact) - candidate >= stable_frames:
        permanent_index = candidate
    return {
        "recovered_30_frames": recovery_index is not None,
        "recovery_frame": recovery_index,
        "recovery_seconds": (
            recovery_index / fps if recovery_index is not None else None
        ),
        "recovery_baseline_advances": (
            int(normal_starts[recovery_index] - normal_starts[0])
            if recovery_index is not None
            else None
        ),
        "permanently_recovered": permanent_index is not None,
        "permanent_recovery_frame": permanent_index,
        "permanent_recovery_seconds": (
            permanent_index / fps if permanent_index is not None else None
        ),
        "exact_window_fraction": float(exact.mean()),
        "mean_absolute_window_offset": float(np.abs(offsets).mean()),
        "terminal_window_offset": int(offsets[-1]),
        "maximum_window_offset": int(offsets.max()),
        "minimum_window_offset": int(offsets.min()),
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_results(
    results: list[dict[str, object]],
    *,
    shifts: tuple[int, ...],
    tolerances: tuple[int, ...],
) -> dict[str, object]:
    baseline_metrics = [item["baseline_post_metrics"] for item in results]
    baseline_micro, baseline_macro = aggregate_alignment_metrics(
        baseline_metrics, tolerances
    )
    summary: dict[str, object] = {
        "song_count": len(results),
        "baseline_post": {"micro": baseline_micro, "macro": baseline_macro},
        "shifts": {},
    }
    for shift in shifts:
        key = str(shift)
        metrics = [item["shifts"][key]["post_metrics"] for item in results]
        recovery = [item["shifts"][key]["recovery"] for item in results]
        micro, macro = aggregate_alignment_metrics(metrics, tolerances)
        recovered_seconds = [
            float(item["recovery_seconds"])
            for item in recovery
            if item["recovery_seconds"] is not None
        ]
        permanent_seconds = [
            float(item["permanent_recovery_seconds"])
            for item in recovery
            if item["permanent_recovery_seconds"] is not None
        ]
        summary["shifts"][key] = {
            "post": {"micro": micro, "macro": macro},
            "recovery": {
                "recovered_30_frames_count": sum(
                    bool(item["recovered_30_frames"]) for item in recovery
                ),
                "recovered_30_frames_rate": float(
                    np.mean(
                        [bool(item["recovered_30_frames"]) for item in recovery]
                    )
                ),
                "median_recovery_seconds": (
                    float(np.median(recovered_seconds))
                    if recovered_seconds
                    else None
                ),
                "permanently_recovered_count": sum(
                    bool(item["permanently_recovered"]) for item in recovery
                ),
                "permanently_recovered_rate": float(
                    np.mean(
                        [bool(item["permanently_recovered"]) for item in recovery]
                    )
                ),
                "median_permanent_recovery_seconds": (
                    float(np.median(permanent_seconds))
                    if permanent_seconds
                    else None
                ),
                "mean_exact_window_fraction": float(
                    np.mean([item["exact_window_fraction"] for item in recovery])
                ),
                "mean_absolute_window_offset": float(
                    np.mean(
                        [item["mean_absolute_window_offset"] for item in recovery]
                    )
                ),
                "terminal_offset_zero_rate": float(
                    np.mean(
                        [item["terminal_window_offset"] == 0 for item in recovery]
                    )
                ),
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    normal_root = args.normal_root.resolve()
    output_root = args.output_root.resolve()
    shifts = tuple(sorted(set(int(value) for value in args.shifts)))
    if not shifts or shifts[0] <= 0:
        raise ValueError("--shifts must contain positive forward offsets")

    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "configs")
    ):
        cfg = compose(config_name=args.config_name)
    components = data.build_data_components(cfg)
    device = torch.device(args.device)
    checkpoint = resolve_path(cfg.full_song.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = instantiate(cfg.model).to(device)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.eval()

    midi_root = resolve_path(cfg.data.midi_root)
    hand_root = resolve_path(cfg.data.hand_root)
    beat_root = (
        resolve_path(cfg.data.beat_root)
        if cfg.data.get("beat_root") is not None
        else None
    )
    normal_paths = sorted(normal_root.rglob("*.npz"))
    if args.limit is not None:
        normal_paths = normal_paths[: args.limit]
    if not normal_paths:
        raise ValueError(f"no artifacts under {normal_root}")

    tolerances = tuple(int(value) for value in cfg.evaluation.tolerances_frames)
    onset_threshold = float(cfg.evaluation.onset_threshold)
    window_width = int(components["slicer"].segment_frames)
    stable_frames = int(args.stable_frames)
    per_record_root = output_root / args.config_name / "per_record"
    started = time.time()
    results: list[dict[str, object]] = []
    for index, normal_path in enumerate(normal_paths, 1):
        relative = normal_path.relative_to(normal_root)
        record_id = relative.with_suffix("").as_posix()
        result_path = per_record_root / relative.with_suffix(".json")
        if result_path.is_file():
            result = load_json(result_path)
            results.append(result)
            print(
                f"[{index:02d}/{len(normal_paths)}] reuse {record_id}",
                flush=True,
            )
            continue

        normal = load_artifact(normal_path)
        pair = data.FilePair(
            record_id=record_id,
            midi_path=midi_root / f"{record_id}.mid",
            hand_path=hand_root / f"{record_id}.npz",
            beat_path=(beat_root / f"{record_id}.txt" if beat_root else None),
        )
        song = components["file_loader"].reader.read(
            pair,
            components["midi_loader"],
            components["hand_loader"],
        )
        chord_positions = tuple(range(len(song.chords)))
        score = components["slicer"].slice(song, 0, 0, chord_positions)
        chord_tokens = score["chords"]
        if normal.probability.shape != (len(song.hands), len(song.chords)):
            raise ValueError(
                f"shape mismatch for {record_id}: "
                f"artifact={normal.probability.shape}, "
                f"loaded={(len(song.hands), len(song.chords))}"
            )

        starts = normal_window_starts(normal.probability)
        final_chord_start = max(0, len(song.chords) - window_width)
        injection_frame = choose_injection_frame(
            starts,
            final_chord_start=final_chord_start,
            maximum_shift=max(shifts),
            fraction=float(args.injection_fraction),
            stable_frames=stable_frames,
        )
        state = ControllerState()
        for frame_index in range(injection_frame):
            if state.chord_start != int(starts[frame_index]):
                raise RuntimeError(
                    f"normal trajectory mismatch for {record_id} at frame "
                    f"{frame_index}: rerun={state.chord_start}, "
                    f"artifact={starts[frame_index]}"
                )
            controller_step(
                model,
                song,
                chord_tokens,
                [state],
                frame_index,
                window_width=window_width,
                final_chord_start=final_chord_start,
                device=device,
                bf16=bool(cfg.train.bf16),
                alignment_threshold=float(cfg.full_song.alignment_threshold),
                advance_after_chords=int(cfg.full_song.advance_after_chords),
                alignment_loss=str(cfg.get("alignment_loss", "bce")),
            )
        if state.chord_start != int(starts[injection_frame]):
            raise RuntimeError(
                f"normal trajectory mismatch for {record_id} at injection: "
                f"rerun={state.chord_start}, artifact={starts[injection_frame]}"
            )

        branch_states = [
            ControllerState(
                chord_start=state.chord_start + shift,
                history_start=state.history_start,
            )
            for shift in shifts
        ]
        post_frame_count = len(song.hands) - injection_frame
        branch_probabilities = [
            np.zeros(
                (post_frame_count, len(song.chords)), dtype=np.float32
            )
            for _ in shifts
        ]
        branch_starts = [
            np.zeros(post_frame_count, dtype=np.int64) for _ in shifts
        ]
        for post_index, frame_index in enumerate(
            range(injection_frame, len(song.hands))
        ):
            for branch_index, branch_state in enumerate(branch_states):
                branch_starts[branch_index][post_index] = (
                    branch_state.chord_start
                )
            rows = controller_step(
                model,
                song,
                chord_tokens,
                branch_states,
                frame_index,
                window_width=window_width,
                final_chord_start=final_chord_start,
                device=device,
                bf16=bool(cfg.train.bf16),
                alignment_threshold=float(cfg.full_song.alignment_threshold),
                advance_after_chords=int(cfg.full_song.advance_after_chords),
                alignment_loss=str(cfg.get("alignment_loss", "bce")),
            )
            for branch_index, row in enumerate(rows):
                branch_start = int(branch_starts[branch_index][post_index])
                branch_probabilities[branch_index][
                    post_index, branch_start : branch_start + len(row)
                ] = row

        baseline_post = sliced_artifact(
            normal,
            normal.probability[injection_frame:].copy(),
            injection_frame,
        )
        baseline_metrics = calculate_alignment_metrics(
            baseline_post,
            onset_threshold=onset_threshold,
            tolerances_frames=tolerances,
        )
        shift_results = {}
        for branch_index, shift in enumerate(shifts):
            shifted_post = sliced_artifact(
                normal,
                branch_probabilities[branch_index],
                injection_frame,
            )
            shift_results[str(shift)] = {
                "post_metrics": calculate_alignment_metrics(
                    shifted_post,
                    onset_threshold=onset_threshold,
                    tolerances_frames=tolerances,
                ),
                "recovery": recovery_metrics(
                    branch_starts[branch_index],
                    starts[injection_frame:],
                    fps=normal.fps,
                    stable_frames=stable_frames,
                ),
            }
        result = {
            "record_id": record_id,
            "frame_count": len(song.hands),
            "chord_count": len(song.chords),
            "fps": normal.fps,
            "injection_frame": injection_frame,
            "injection_seconds": injection_frame / normal.fps,
            "normal_window_start_at_injection": int(starts[injection_frame]),
            "baseline_post_metrics": baseline_metrics,
            "shifts": shift_results,
        }
        write_json(result_path, result)
        results.append(result)
        details = " ".join(
            f"+{shift}:recover="
            f"{shift_results[str(shift)]['recovery']['recovery_seconds']}"
            for shift in shifts
        )
        print(
            f"[{index:02d}/{len(normal_paths)}] {record_id} "
            f"inject={injection_frame} {details} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        del song, normal, branch_probabilities
        torch.cuda.empty_cache()

    summary = aggregate_results(
        results, shifts=shifts, tolerances=tolerances
    )
    summary.update(
        {
            "config_name": args.config_name,
            "checkpoint": str(checkpoint),
            "normal_root": str(normal_root),
            "injection_fraction": float(args.injection_fraction),
            "stable_frames": stable_frames,
            "forward_shifts": list(shifts),
        }
    )
    summary_path = output_root / f"{args.config_name}.summary.json"
    write_json(summary_path, summary)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
