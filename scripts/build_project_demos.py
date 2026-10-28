"""Build the public demo derivatives without changing saved inference events.

Requires Python with mido, numpy and scipy, FFmpeg/FFprobe, FluidSynth and a
local piano SoundFont. Source media, synthesis WAVs and audit records stay
outside docs/. No network access or publication is performed by this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import warnings

import mido
import numpy as np
from scipy.io import wavfile


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO.parent
SAMPLE_RATE = 48_000
GAIN = 1.0
BUILD_VERSION = "airpiano-web-media-v1"
EXAMPLES = [
    ("jinxuan-beethoven-8-3", "Jinxuan", "beethoven_8_3", "Beethoven — Sonata No. 8, III"),
    ("jinxuan-summer", "Jinxuan", "summer", "Summer"),
    ("huiran-bach-prelude", "Huiran", "bach_prelude", "Bach — Prelude"),
    ("huiran-schubert-serenade", "Huiran", "schubert_serenade", "Schubert — Serenade"),
    ("venkat-minuet", "Venkat", "minuet", "Minuet"),
    ("venkat-chopin-waltz", "Venkat", "chopin_waltz", "Chopin — Waltz"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], log: Path | None = None) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log:
        log.write_bytes(result.stdout + b"\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def probe(path: Path, ffprobe: str) -> dict:
    return json.loads(run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def validate_prediction(midi_path: Path, artifact_path: Path, video: dict) -> dict:
    midi = mido.MidiFile(midi_path)
    if midi.ticks_per_beat != 480:
        raise ValueError("Expected the saved 480-tick constant-tempo prediction export")
    tick, actual = 0, []
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "set_tempo" and message.tempo != 500_000:
            raise ValueError("Unexpected MIDI tempo; refuse implicit time conversion")
        if message.type in ("note_on", "note_off"):
            is_on = message.type == "note_on" and message.velocity > 0
            actual.append((tick, 1 if is_on else 0, message.note, message.velocity if is_on else 0))
    expected = []
    with np.load(artifact_path, allow_pickle=False) as artifact:
        fps = float(artifact["fps"])
        if int(artifact["frame_start"]) != 0 or abs(float(artifact["frame_time"][0])) > 1e-9:
            raise ValueError("Source artifact does not start at video zero")
        frame_count = int(artifact["frame_mask"].sum())
        stream = next(s for s in video["streams"] if s["codec_type"] == "video")
        if int(stream["nb_frames"]) != frame_count or fps != 30:
            raise ValueError("Aligned video and inference frame sequence differ")
        frame_interval = np.diff(artifact["frame_time"][:frame_count])
        if not np.allclose(frame_interval, 1 / fps, atol=1e-8):
            raise ValueError("Unexpected inference frame clock")
        duration = float(artifact["note_duration"])
        velocity = int(artifact["note_velocity"])
        for slot in np.flatnonzero(artifact["chord_mask"]):
            onset = float(artifact["release_time"][slot])
            if not math.isfinite(onset):
                continue
            for pitch in np.flatnonzero(artifact["chord_pitches"][slot]):
                expected.extend([(round(onset * 960), 1, int(pitch), velocity),
                                 (round((onset + duration) * 960), 0, int(pitch), 0)])
        latency = float(artifact["processing_latency"])
    if sorted(actual) != sorted(expected):
        raise ValueError(f"Saved MIDI does not reproduce NPZ releases exactly: {midi_path}")
    onsets = sorted(set(t / 960 for t, on, _, _ in actual if on))
    last_note_off = max(t / 960 for t, on, _, _ in actual if not on)
    return {"exact_npz_event_match": True, "note_count": sum(e[1] for e in actual),
            "distinct_onset_count": len(onsets), "first_onset": onsets[0],
            "middle_onset": onsets[len(onsets) // 2], "last_onset": onsets[-1],
            "last_note_off": last_note_off, "saved_note_duration": duration,
            "saved_velocity": velocity, "processing_latency_already_in_midi": latency,
            "video_frame_count": frame_count, "fps": fps, "frame_start": 0,
            "sample_onsets": [onsets[0], onsets[len(onsets) // 2], onsets[-1]]}


def audio_qa(wav_path: Path, mp4_path: Path, onset_check: dict, ffmpeg: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        rate, reference = wavfile.read(wav_path)
    if rate != SAMPLE_RATE or reference.dtype.kind != "f":
        raise ValueError("Expected floating-point 48 kHz synthesis")
    peak = float(np.max(np.abs(reference)))
    if peak >= .98:
        raise ValueError(f"Synthesis gain needs a common reduction: peak={peak}")
    envelope = np.max(np.abs(reference), axis=1)
    first_audible = int(np.flatnonzero(envelope > 1e-5)[0]) / rate
    attack_offset = first_audible - onset_check["first_onset"]
    if not (-.005 <= attack_offset <= .050):
        raise ValueError(f"Unexpected synthesis start offset: {attack_offset}")
    decoded_bytes = run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(mp4_path),
                         "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
                         "-ar", str(rate), "-ac", "2", "pipe:1"])
    decoded = np.frombuffer(decoded_bytes, dtype="<f4").reshape(-1, 2)
    decoded_peak = float(np.max(np.abs(decoded)))
    if decoded_peak >= .999:
        raise ValueError(f"Decoded AAC clipping risk: peak={decoded_peak}")
    ref = reference.mean(axis=1)
    output = decoded.mean(axis=1)
    checks = []
    length = 8192
    for onset in onset_check["sample_onsets"]:
        start = int((onset + .08) * rate)
        start = min(start, len(output) - length - 96, len(ref) - length - 96)
        a = ref[start:start + length].astype(np.float64)
        a -= a.mean()
        scores = []
        for lag in range(-96, 97):
            b = output[start + lag:start + lag + length].astype(np.float64)
            b -= b.mean()
            score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))
            scores.append(score)
        index = int(np.argmax(scores))
        best_lag = index - 96
        if scores[index] < .95 or abs(best_lag) > 48:
            raise ValueError(f"AAC waveform differs from synthesized source at {onset}s: {max(scores)}, lag={best_lag}")
        checks.append({"predicted_onset": onset, "correlation": scores[index],
                       "best_lag_samples": best_lag, "best_lag_ms": best_lag * 1000 / rate})
    return {"synth_peak": peak, "clipping": False, "first_audible_seconds": first_audible,
            "first_note_attack_offset_ms": attack_offset * 1000,
            "decoded_audio_matches_synthesis": checks,
            "decoded_peak": decoded_peak}


def build_one(example: tuple, args: argparse.Namespace, provenance: dict) -> tuple[dict, dict]:
    ident, performer, piece, title = example
    group = f"{performer.lower()}_tabletop_aligned"
    source_video = args.workspace / "REAL" / group / f"{piece}.mp4"
    source_midi = args.workspace / "REAL/airpiano_air_tabletop/AIR" / group / f"{piece}.mid"
    source_npz = source_midi.with_suffix(".npz")
    if source_midi.name.endswith(".gt.mid"):
        raise ValueError("Ground truth MIDI is never a demo input")
    source_probe = probe(source_video, args.ffprobe)
    video_stream = next(s for s in source_probe["streams"] if s["codec_type"] == "video")
    source_duration = float(video_stream["duration"])
    if abs(float(video_stream["start_time"])) > 1e-9:
        raise ValueError("Unexpected source-video start time")
    prediction = validate_prediction(source_midi, source_npz, source_probe)
    output_duration = max(source_duration, prediction["last_note_off"] + 1)
    target = args.output / f"{ident}.mp4"
    thumbnail = target.with_suffix(".jpg")
    downloadable = target.with_suffix(".mid")
    wav = args.cache / f"{ident}.wav"
    cache_report = args.cache / f"{ident}.json"
    hashes = {"video": sha256(source_video), "prediction_midi": sha256(source_midi), "artifact_npz": sha256(source_npz)}
    fingerprint = hashlib.sha256(json.dumps({"version": BUILD_VERSION, "hashes": hashes,
        "soundfont": provenance["soundfont_sha256"], "gain": GAIN}, sort_keys=True).encode()).hexdigest()
    prior = json.loads(cache_report.read_text()) if cache_report.exists() else {}
    reuse = not args.force and prior.get("fingerprint") == fingerprint and wav.exists() and target.exists()
    print(f"[{ident}] validated {prediction['note_count']} prediction notes; {'reusing' if reuse else 'rendering'}", flush=True)
    if not reuse:
        run([args.fluidsynth, "-n", "-i", "-F", str(wav), "-T", "wav", "-O", "float",
             "-r", str(SAMPLE_RATE), "-g", str(GAIN), "-R", "0", "-C", "0",
             "-o", "player.timing-source=sample", str(args.soundfont), str(source_midi)],
            args.cache / f"{ident}.synth.log")
        pad = max(0, output_duration - source_duration)
        video_filter = "scale=1280:720:flags=lanczos,setsar=1,fps=30"
        if pad:
            video_filter += f",tpad=stop_mode=clone:stop_duration={pad:.9f}"
        run([args.ffmpeg, "-hide_banner", "-y", "-i", str(source_video), "-i", str(wav),
             "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1", "-map_chapters", "-1",
             "-vf", video_filter, "-af", "apad", "-t", f"{output_duration:.9f}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-maxrate", "2000k",
             "-bufsize", "4000k", "-pix_fmt", "yuv420p", "-threads", "4",
             "-c:a", "aac", "-b:a", "128k", "-ar", str(SAMPLE_RATE), "-ac", "2",
             "-movflags", "+faststart", str(target)], args.cache / f"{ident}.encode.log")
    shutil.copyfile(source_midi, downloadable)
    run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "5", "-i", str(target),
         "-frames:v", "1", "-q:v", "3", str(thumbnail)])
    target_probe = probe(target, args.ffprobe)
    streams = target_probe["streams"]
    output_video = next(s for s in streams if s["codec_type"] == "video")
    output_audio = next(s for s in streams if s["codec_type"] == "audio")
    assert len(streams) == 2 and output_video["codec_name"] == "h264" and output_audio["codec_name"] == "aac"
    assert (output_video["width"], output_video["height"], output_video["pix_fmt"]) == (1280, 720, "yuv420p")
    assert output_video["avg_frame_rate"] == "30/1"
    assert abs(float(output_video["start_time"])) < .001 and abs(float(output_audio["start_time"])) < .001
    assert abs(float(target_probe["format"]["duration"]) - output_duration) <= .05
    assert target.stat().st_size < 100 * 1024**2
    assert sha256(downloadable) == hashes["prediction_midi"]
    audio_checks = audio_qa(wav, target, prediction, args.ffmpeg)
    report = {"id": ident, "fingerprint": fingerprint, "source_video": str(source_video),
        "source_midi": str(source_midi), "source_npz": str(source_npz), "source_hashes": hashes,
        "source_duration_seconds": source_duration, "requested_output_duration_seconds": output_duration,
        "last_frame_hold_seconds": max(0, output_duration-source_duration), "prediction": prediction,
        "audio": audio_checks, "output_bytes": target.stat().st_size, "output_sha256": sha256(target),
        "source_audio_removed": True, "mux_stream_map": ["0:v:0", "1:a:0"],
        "output_streams": [{k: s.get(k) for k in ["codec_type", "codec_name", "start_time", "duration", "nb_frames"]} for s in streams]}
    cache_report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    manifest_entry = {"id": ident, "title": title, "performer": performer, "piece": piece,
        "video": f"assets/demos/{ident}.mp4", "poster": f"assets/demos/{ident}.jpg",
        "midi": f"assets/demos/{ident}.mid", "duration_seconds": float(target_probe["format"]["duration"]),
        "source_duration_seconds": source_duration}
    print(f"[{ident}] ready: {target.stat().st_size / 1024**2:.1f} MiB; audio/source match verified", flush=True)
    return manifest_entry, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--fluidsynth", required=True)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.cache = args.workspace / ".cache/website-media"
    args.output = REPO / "docs/assets/demos"
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"build_version": BUILD_VERSION,
        "fluidsynth_version": run([args.fluidsynth, "--version"]).decode(errors="replace").strip(),
        "fluidsynth_release": "https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1",
        "fluidsynth_download": "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip",
        "fluidsynth_archive_sha256": "fab7a2e4b85675b66970f97a39bbc239729c5e0f237198b5922a6a73cbc8677c",
        "ffmpeg_version": run([args.ffmpeg, "-version"]).decode(errors="replace").splitlines()[0],
        "soundfont_path": str(args.soundfont), "soundfont_sha256": sha256(args.soundfont),
        "sample_rate": SAMPLE_RATE, "synthesis_gain": GAIN, "reverb": False, "chorus": False,
        "timing_source": "sample", "extra_time_shift_seconds": 0,
        "video_encoding": {"codec": "H.264", "size": "1280x720", "fps": 30, "crf": 22, "maxrate_kbps": 2000},
        "audio_encoding": {"codec": "AAC", "bitrate_kbps": 128}, "midi_source": "prediction only; never *.gt.mid"}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        outputs = list(executor.map(lambda item: build_one(item, args, provenance), EXAMPLES))
    manifest = {"demos": [item[0] for item in outputs]}
    write_json_atomic(REPO / "docs/demos.json", manifest)
    total_bytes = sum(path.stat().st_size for path in (REPO / "docs").rglob("*") if path.is_file())
    assert total_bytes < 1024**3
    report = {"provenance": provenance, "demos": [item[1] for item in outputs], "docs_total_bytes": total_bytes}
    report_dir = args.workspace / "output/website-media"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "build_report.json"
    write_json_atomic(report_path, report)
    print(f"Six full demos ready; website total {total_bytes / 1024**2:.1f} MiB. Audit: {report_path}", flush=True)


if __name__ == "__main__":
    main()
