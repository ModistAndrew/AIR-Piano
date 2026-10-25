#!/usr/bin/env python3
"""Export synchronized media, skeletons, score inputs, and alignment images.

Run in the project environment, for example::

    python scripts/inspect_segment.py \
        --record-id PianoMotion10M120s/864712/BV168411r7fB

The segment index selects an accepted, unaugmented dataset window. Its width
counts chords, not video frames. Outputs are staged and published together
only after validation. Choose a new --output-dir to retain previous exports.
Requires the project's Python dependencies, FFmpeg with libvpx-vp9, ffprobe,
completed raw/canonical hand caches, and a trained checkpoint. No inference
or preprocessing of the full dataset is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from airpiano.data.dataset import AlignmentDataset, build_data_components, collate_alignment_batch
from airpiano.data.midi import MidoNoteReader
from airpiano.data.types import FilePair
from airpiano.inference import load_model_checkpoint, write_inference_artifact
from airpiano.model.loss import alignment_probabilities, masked_bce_loss
from scripts.segment_hands import render_hand_videos
from scripts.segment_score import export_score_bundle, render_alignment_images
from airpiano.train import _autocast, _forward, seed_everything
from airpiano.utils import resolve_path, write_json


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def probe(path, *, count_frames=False):
    command = ["ffprobe", "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command.extend(["-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(subprocess.check_output(command))


def load_segment(config_path, record_id, segment_index):
    """Use the production loaders and slicer for exactly one explicit record."""
    record = Path(record_id)
    if record.is_absolute() or ".." in record.parts or not record_id.strip():
        raise ValueError("--record-id must be a dataset-relative recording stem")
    if segment_index < 0:
        raise ValueError("--segment-index must be nonnegative")
    config_path = resolve_path(config_path)
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem)
    components = build_data_components(cfg)
    sources = {
        "config": config_path,
        "video": resolve_path(cfg.data.video_root) / f"{record_id}.mp4",
        "audio": resolve_path(cfg.data.video_root).parent / "Audio" / f"{record_id}.m4a",
        "midi": resolve_path(cfg.data.midi_root) / f"{record_id}.mid",
        "beats": resolve_path(cfg.data.beat_root) / f"{record_id}.txt",
        "canonical_hands": resolve_path(cfg.data.hand_root) / f"{record_id}.npz",
        "raw_hands": resolve_path(cfg.data.raw_wilor_root) / record_id,
    }
    pair = FilePair(record_id=record_id, midi_path=sources["midi"],
                    hand_path=sources["canonical_hands"], beat_path=sources["beats"])
    song = components["file_loader"].reader.read(
        pair, components["midi_loader"], components["hand_loader"]
    )
    dataset = AlignmentDataset(
        songs=[song], fps=float(cfg.dataset.fps), slicer=components["slicer"],
        target_generator=components["target_generator"],
        tokenizer=components["tokenizer"], augmenter=None,
    )
    if segment_index >= len(dataset):
        raise IndexError(f"Segment {segment_index} requested; record has {len(dataset)} accepted segments")
    sample = dataset[segment_index]
    _, (_, _, chord_positions) = dataset.samples[segment_index]
    chords = tuple(song.chords[index] for index in chord_positions)
    return cfg, song, chords, sample, sources


def infer_segment(cfg, sample, chords, checkpoint, output):
    device = torch.device(str(cfg.device))
    seed_everything(int(cfg.seed))
    model = load_model_checkpoint(checkpoint, device)
    model.eval()
    batch = collate_alignment_batch([sample])
    with torch.inference_mode():
        with _autocast(device, bool(cfg.train.bf16)):
            logits = _forward(model, batch, device)
        probability = alignment_probabilities(logits)[0].cpu().numpy()
        loss = float(masked_bce_loss(
            logits, batch["target"].to(device), batch["frame_mask"].to(device),
            batch["chord_mask"].to(device),
        ))
    write_inference_artifact(
        output / "inference.npz", probability=probability,
        target=sample["target"].numpy(), frame_mask=batch["frame_mask"][0].numpy(),
        chord_mask=batch["chord_mask"][0].numpy(),
        chord_pitches=sample["chords"][:, :128].numpy(),
        chord_ids=sample["chord_ids"].numpy(), record_id=sample["record_id"],
        frame_start=sample["frame_start"], fps=sample["fps"], loss=loss,
        extra_arrays={
            "tokens": sample["chords"].numpy(),
            "hands_model": sample["hands"].numpy(),
            "hand_presence": sample["hand_presence"].numpy(),
            "physical_onsets_seconds": np.array([chord.time for chord in chords]),
            "duration_ticks": np.array([chord.duration_ticks for chord in chords]),
        },
    )
    del model
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = payload["config"]
    info = {
        **file_identity(checkpoint), "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"), "device": str(device),
        "bf16": bool(cfg.train.bf16) and device.type == "cuda",
        "dataset_config_matches": saved_config["dataset"] == OmegaConf.to_container(cfg.dataset, resolve=True),
        "model_config_matches": saved_config["model"] == OmegaConf.to_container(cfg.model, resolve=True),
        "probability": "sigmoid(logits.float()); no real-time controller",
        "loss_bce": loss, "shape": list(probability.shape),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
    }
    del payload
    return info


def export_source_media(video, audio, sample, output):
    video_info, audio_info = probe(video), probe(audio)
    stream = next(item for item in video_info["streams"] if item["codec_type"] == "video")
    audio_stream = next(item for item in audio_info["streams"] if item["codec_type"] == "audio")
    fps, frame_start = float(sample["fps"]), int(sample["frame_start"])
    frames = len(sample["hands"])
    if not math.isclose(float(Fraction(stream["avg_frame_rate"])), fps, rel_tol=1e-6):
        raise ValueError("Source and canonical FPS must match for frame-exact export")
    start, stop = frame_start / fps, (frame_start + frames) / fps
    sample_rate = int(audio_stream["sample_rate"])
    start_sample, stop_sample = round(start * sample_rate), round(stop * sample_rate)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", str(audio),
        "-map", "0:a:0", "-af",
        f"atrim=start_sample={start_sample}:end_sample={stop_sample},asetpts=PTS-STARTPTS",
        "-c:a", "pcm_s16le", str(output / "audio.wav"),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-y",
        "-i", str(video), "-i", str(output / "audio.wav"),
        "-map", "0:v:0", "-map", "1:a:0", "-vf",
        f"trim=start_frame={frame_start}:end_frame={frame_start + frames},setpts=PTS-STARTPTS",
        # trim already determines the exact frame count; -frames:v can stop
        # FFmpeg before the separate audio input has been fully consumed.
        "-r", str(fps), "-t", f"{frames / fps:.12f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-threads", "4",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(output / "video.mp4"),
    ], check=True)
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "sample_rate": sample_rate, "channels": int(audio_stream["channels"]),
            "audio_samples": stop_sample - start_sample,
            "source_start_sample": start_sample, "source_stop_sample": stop_sample}


def text_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t"))


def validate_bundle(output, sample, chords, song, media, beat_path):
    """Check written media and text against the exact dataset tensors/events."""
    frames, fps = len(sample["hands"]), float(sample["fps"])
    start, stop = sample["frame_start"] / fps, (sample["frame_start"] + frames) / fps
    videos = {}
    for filename in ("video.mp4", "hands_front.webm", "hands_side.webm"):
        info = probe(output / filename, count_frames=True)
        stream = next(item for item in info["streams"] if item["codec_type"] == "video")
        if int(stream["nb_read_frames"]) != frames:
            raise ValueError(f"Incorrect frame count: {filename}")
        if not math.isclose(float(Fraction(stream["avg_frame_rate"])), fps, rel_tol=1e-6):
            raise ValueError(f"Incorrect FPS: {filename}")
        if (stream["width"], stream["height"]) != (media["width"], media["height"]):
            raise ValueError(f"Incorrect dimensions: {filename}")
        videos[filename] = {"frames": frames, "fps": fps,
                            "width": stream["width"], "height": stream["height"]}
        if filename == "video.mp4":
            audio_stream = next(item for item in info["streams"] if item["codec_type"] == "audio")
            audio_start = float(audio_stream.get("start_time", 0))
            audio_duration = float(audio_stream["duration"])
            if abs(audio_start) > 1 / fps or abs(audio_duration - frames / fps) > 1 / fps:
                raise ValueError("Muxed MP4 audio does not match the segment timing")
            videos[filename]["audio_start_seconds"] = audio_start
            videos[filename]["audio_duration_seconds"] = audio_duration
    # The alpha renderer separately decodes every WebM alpha plane. Decode the
    # conventional clip's complete video/audio streams here as well.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output / "video.mp4"),
                    "-f", "null", "-"], check=True)
    with wave.open(str(output / "audio.wav"), "rb") as handle:
        if (handle.getnframes(), handle.getframerate(), handle.getnchannels()) != (
                media["audio_samples"], media["sample_rate"], media["channels"]):
            raise ValueError("Exported audio does not cover the exact interval")
    with np.load(output / "inference.npz", allow_pickle=False) as arrays:
        for key, tensor in (("tokens", "chords"), ("target", "target"),
                            ("hand_presence", "hand_presence"), ("hands_model", "hands")):
            np.testing.assert_array_equal(arrays[key], sample[tensor].numpy())
        probabilities = arrays["probability"]
        if (probabilities.shape != (frames, len(chords)) or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError("Invalid model probability matrix")
    with np.load(output / "hand_geometry.npz", allow_pickle=False) as geometry:
        presence = sample["hand_presence"].numpy()
        np.testing.assert_array_equal(geometry["presence"], presence)
        np.testing.assert_array_equal(geometry["frame_indices"],
                                      np.arange(sample["frame_start"], sample["frame_start"] + frames))
        anchors = np.broadcast_to(geometry["anchors"], (frames, 2, 2))
        np.testing.assert_array_equal(geometry["side_xy"][:, :, 0][presence], anchors[presence])
        if (not np.isnan(geometry["side_xy"][~presence]).all()
                or not np.isnan(geometry["front_xy"][~presence]).all()):
            raise ValueError("Absent hands must have no skeleton geometry")
    beat_rows = text_rows(output / "beats.txt")
    source_beats = np.loadtxt(beat_path, ndmin=2)
    selected_beats = source_beats[(source_beats[:, 0] >= start) & (source_beats[:, 0] < stop)]
    np.testing.assert_allclose([float(row["source_seconds"]) for row in beat_rows],
                               selected_beats[:, 0], atol=1e-9, rtol=0)
    token_rows = text_rows(output / "score_tokens.txt")
    np.testing.assert_array_equal(
        np.array([json.loads(row["token"]) for row in token_rows]), sample["chords"].numpy()
    )
    onset_rows = text_rows(output / "chord_onsets.txt")
    if len(onset_rows) != len(chords):
        raise ValueError("Onset export must include every score chord")
    for row, chord in zip(onset_rows, chords):
        mean = float(np.mean(json.loads(row["note_source_onsets"])))
        if abs(mean - chord.time) > 1e-10 or abs(float(row["mean_source_seconds"]) - mean) > 1e-10:
            raise ValueError("Chord onset text disagrees with physical note onsets")
    expected = sorted((max(note.onset, start) - start, min(note.offset, stop) - start,
                       note.pitch, note.velocity) for note in song.notes
                      if note.onset < stop and note.offset > start)
    actual = sorted((note.onset, note.offset, note.pitch, note.velocity)
                    for note in MidoNoteReader().read(output / "gt_notes.mid"))
    if len(actual) != len(expected):
        raise ValueError("MIDI note count changed during export")
    for before, after in zip(expected, actual):
        if before[2:] != after[2:] or max(abs(before[i] - after[i]) for i in (0, 1)) > 1 / 60_000 + 1e-8:
            raise ValueError("MIDI physical timing, pitch, or velocity changed")
    return {"status": "passed", "videos": videos, "audio_samples": media["audio_samples"],
            "token_shape": list(sample["chords"].shape), "alignment_shape": [frames, len(chords)],
            "physical_notes_checked": len(actual), "physical_onsets_checked": len(onset_rows),
            "beat_markers_checked": len(beat_rows), "fixed_wrist_geometry_checked": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio-root", type=Path, help="Defaults to the Audio sibling of data.video_root")
    parser.add_argument("--device", help="Override the configured inference device")
    args = parser.parse_args(argv)
    cfg, song, chords, sample, sources = load_segment(args.config, args.record_id, args.segment_index)
    if args.device:
        cfg.device = args.device
    if args.audio_root:
        sources["audio"] = resolve_path(args.audio_root) / f"{args.record_id}.m4a"
    checkpoint = resolve_path(args.checkpoint or cfg.test.checkpoint)
    output = resolve_path(args.output_dir or PROJECT_ROOT / "assets/inspect" /
                          "_".join(Path(args.record_id).parts) / f"segment_{args.segment_index:06d}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose another --output-dir: {output}")
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    frames, fps = len(sample["hands"]), float(sample["fps"])
    frame_start = int(sample["frame_start"])
    print(f"[segment] {args.record_id}: {frames} frames, {len(chords)} chords, "
          f"{frame_start / fps:.3f}-{(frame_start + frames) / fps:.3f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        work = Path(temporary) / "bundle"
        work.mkdir()
        print("[inference] running the exact unaugmented sample", flush=True)
        checkpoint_info = infer_segment(cfg, sample, chords, checkpoint, work)
        print("[media] exporting source video and sample-exact audio", flush=True)
        media = export_source_media(sources["video"], sources["audio"], sample, work)
        print("[score] exporting notes, beats, tokens, onsets, and alignment images", flush=True)
        score = export_score_bundle(song=song, chords=chords, sample=sample,
                                    beat_path=sources["beats"], output_dir=work)
        alignment = render_alignment_images(work / "inference.npz", work)
        print("[hands] rendering transparent front/side videos with fixed wrist markers", flush=True)
        hands = render_hand_videos(
            raw_dir=sources["raw_hands"], canonical_path=sources["canonical_hands"],
            frame_start=frame_start, frame_stop=frame_start + frames, fps=fps,
            width=media["width"], height=media["height"], output_dir=work,
        )
        print("[validate] checking media, physical events, and exact model tensors", flush=True)
        validation = validate_bundle(work, sample, chords, song, media, sources["beats"])
        sources_info = {name: file_identity(path if path.is_file() else path / "manifest.json")
                        for name, path in sources.items()}
        inventory = {path.name: {key: value for key, value in file_identity(path).items() if key != "path"}
                     for path in sorted(work.iterdir()) if path.is_file()}
        manifest = {
            "schema_version": "airpiano-segment-inspection-v1", "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(), "record_id": args.record_id,
            "segment_index": args.segment_index, "output_dir": str(output),
            "augmentation": None, "config": OmegaConf.to_container(cfg, resolve=True),
            "sources": sources_info, "checkpoint": checkpoint_info,
            "segment": {"frame_start": frame_start, "frame_stop": frame_start + frames,
                        "frame_count": frames, "fps": fps, "start_seconds": frame_start / fps,
                        "stop_seconds": (frame_start + frames) / fps, "duration_seconds": frames / fps,
                        "chord_ids": sample["chord_ids"].tolist(),
                        "hand_presence_counts": sample["hand_presence"].sum(dim=0).tolist()},
            "media": media, "score": score, "alignment": alignment,
            "hands": hands, "validation": validation, "files": inventory,
        }
        write_json(work / "manifest.json", manifest)
        work.rename(output)
    print(f"[complete] {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
