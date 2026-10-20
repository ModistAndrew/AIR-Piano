#!/usr/bin/env python3
"""Build one repeatable 30s, 60s, and 120s clip per PianoMotion10M split entry."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DURATIONS = (30, 60, 120)
TOLERANCE = 0.1


def read_split(path):
    records, seen = [], set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = raw.strip()
        if not record or record.startswith("#"):
            continue
        parts = record.split("/")
        if (len(parts) != 3 or parts[0] != "PianoMotion10M"
                or any(part in ("", ".", "..") for part in parts)
                or any(character.isspace() for character in record)
                or Path(parts[-1]).suffix):
            raise ValueError(f"{path}:{number}: invalid source ID: {record}")
        if record in seen:
            raise ValueError(f"{path}:{number}: duplicate source ID: {record}")
        seen.add(record)
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no source records")
    return records


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}): {result.stderr[-1500:]}")
    return result.stdout


def probe(path):
    payload = json.loads(run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,start_time,duration:format=start_time,duration",
        "-of", "json", str(path),
    ]))
    streams = payload.get("streams", [])
    container = payload.get("format", {})
    duration = container.get("duration")
    if duration in (None, "N/A"):
        duration = max((float(stream["duration"]) for stream in streams
                        if stream.get("duration") not in (None, "N/A")), default=0)
    duration = float(duration)
    starts = [float(item["start_time"]) for item in [container, *streams]
              if item.get("start_time") not in (None, "N/A")]
    if not math.isfinite(duration) or duration <= 0 or not all(map(math.isfinite, starts)):
        raise ValueError(f"{path}: invalid media timing")
    return {
        "duration": duration, "start": min(starts, default=0),
        "video": [s.get("codec_name") for s in streams if s.get("codec_type") == "video"],
        "audio": [s.get("codec_name") for s in streams if s.get("codec_type") == "audio"],
    }


def start_seconds(source_id, duration, common_duration):
    digest = hashlib.sha256(f"42:PianoMotion10M{duration}s:{source_id}".encode()).digest()
    fraction = int.from_bytes(digest[:16], "big") / (2**128 - 1)
    return float(f"{fraction * max(0.0, common_duration - duration):.9f}")


def valid_pair(video, audio, duration):
    try:
        if not video.is_file() or not audio.is_file():
            return False
        v, a = probe(video), probe(audio)
        return (v["video"] == ["h264"] and not v["audio"]
                and a["audio"] == ["aac"] and not a["video"]
                and all(abs(p["duration"] - duration) <= TOLERANCE
                        and abs(p["start"]) <= TOLERANCE for p in (v, a))
                and abs(v["duration"] - a["duration"]) <= TOLERANCE)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False


def build_pair(source_video, source_audio, video, audio, start, duration):
    if valid_pair(video, audio, duration):
        return "skipped"
    video.parent.mkdir(parents=True, exist_ok=True)
    audio.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=video.parent) as vd, \
            tempfile.TemporaryDirectory(prefix=".clip-", dir=audio.parent) as ad:
        temporary_video, temporary_audio = Path(vd) / video.name, Path(ad) / audio.name
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        run(command + [
            "-i", str(source_video), "-map", "0:v:0", "-vf",
            f"trim=start={start:.9f}:end={start + duration:.9f},setpts=PTS-STARTPTS",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-vsync", "0", "-movflags", "+faststart",
            str(temporary_video),
        ])
        run(command + [
            "-i", str(source_audio), "-ss", f"{start:.9f}", "-t", f"{duration:.9f}",
            "-map", "0:a:0", "-vn", "-c:a", "copy", "-avoid_negative_ts", "make_zero",
            str(temporary_audio),
        ])
        if not valid_pair(temporary_video, temporary_audio, duration):
            raise RuntimeError("Cut outputs failed stream or timing checks")
        try:
            os.replace(temporary_video, video)
            os.replace(temporary_audio, audio)
        except BaseException:
            # Leave no half-published pair that a later run could mistake for complete.
            video.unlink(missing_ok=True)
            audio.unlink(missing_ok=True)
            raise
    return "processed"


def save_progress(root, indices, report):
    contents = {root / "Metadata/simple_duration_report.json": json.dumps(report, indent=2) + "\n"}
    for duration, rows in indices.items():
        path = root / f"Metadata/PianoMotion10M{duration}s/clips.jsonl"
        contents[path] = "".join(json.dumps(row) + "\n" for row in rows)
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def record_failure(report, source_id, duration, error):
    report["failed"] += 1
    report["failures"].append({"source_id": source_id, "duration": duration, "error": str(error)})
    print(f"[failed] {source_id} ({duration}s): {error}", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("split_file", type=Path)
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    report = {"status": "running", "input_records": 0, "processed": 0, "skipped": 0,
              "excluded": 0, "failed": 0, "exclusions": [], "failures": []}
    indices, source_id, duration, code = {}, None, None, 0
    try:
        if not root.is_dir():
            raise NotADirectoryError(root)
        records = read_split(args.split_file)
        for program in ("ffmpeg", "ffprobe"):
            if shutil.which(program) is None:
                raise RuntimeError(f"Required program not found: {program}")
        report["input_records"] = len(records)
        indices = {duration: [] for duration in DURATIONS}
        for source_id in records:
            duration = None
            stem = Path(source_id).relative_to("PianoMotion10M")
            source_video = (root / "Video" / source_id).with_suffix(".mp4")
            source_audio = (root / "Audio" / source_id).with_suffix(".m4a")
            try:
                if not source_video.is_file() or not source_audio.is_file():
                    raise FileNotFoundError(f"Source video/audio pair missing for {source_id}")
                v, a = probe(source_video), probe(source_audio)
                if len(v["video"]) != 1 or a["video"] or a["audio"] != ["aac"]:
                    raise ValueError("Expected one video stream and separate AAC audio")
                common = min(v["duration"], a["duration"])
                for duration in DURATIONS:
                    if common < duration:
                        report["excluded"] += 1
                        report["exclusions"].append({"source_id": source_id, "duration": duration,
                                                     "error": "Source audio/video is too short"})
                        continue
                    record_id = f"PianoMotion10M{duration}s/{stem.as_posix()}"
                    video = (root / "Video" / record_id).with_suffix(".mp4")
                    audio = (root / "Audio" / record_id).with_suffix(".m4a")
                    start = start_seconds(source_id, duration, common)
                    try:
                        result = build_pair(source_video, source_audio, video, audio, start, duration)
                        indices[duration].append({
                            "source_id": source_id, "record_id": record_id,
                            "start_seconds": start, "end_seconds": start + duration,
                            "video_path": video.relative_to(root).as_posix(),
                            "audio_path": audio.relative_to(root).as_posix(),
                        })
                        report[result] += 1
                        print(f"[{result}] {record_id}", flush=True)
                    except Exception as error:
                        record_failure(report, source_id, duration, error)
            except Exception as error:
                record_failure(report, source_id, duration, error)
            source_id, duration = None, None
            save_progress(root, indices, report)
        report["status"] = "completed_with_failures" if report["failed"] else "complete"
        code = 1 if report["failed"] else 0
    except KeyboardInterrupt:
        report["status"], code = "interrupted", 130
        if source_id is not None:
            record_failure(report, source_id, duration, "Interrupted")
    except Exception as error:
        report.update(status="setup_failed", setup_error=str(error))
        print(f"[setup failed] {error}", file=sys.stderr)
        code = 2
    finally:
        if root.is_dir():
            try:
                save_progress(root, indices, report)
            except OSError as error:
                print(f"[report failed] {error}", file=sys.stderr)
                code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
