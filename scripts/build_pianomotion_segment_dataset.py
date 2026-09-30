"""Build the PianoMotion10MSeg audio/video dataset.

All released clip-local MP3s are moved out of the full-song PianoMotion10M
namespace.  Video is materialized only for clips whose parent song occurs in a
selected song-level split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = "PianoMotion10M"
TARGET_DATASET = "PianoMotion10MSeg"
DEFAULT_DATA_ROOT = Path("/storage/jinxuan/data/AirPianoFull")
DEFAULT_RECORDS_PATH = Path(
    "/storage2/jinxuan/data/AirPianoFull/"
    "processed_pianomotionnew_wilor_2p5d_30fps/"
    "PianoMotion10M/records.json"
)
DEFAULT_VALID_SPLIT = (
    PROJECT_ROOT / "assets" / "splits" / "extended_valid.txt"
)
DEFAULT_SPLIT_OUTPUT = PROJECT_ROOT / "assets" / "splits"


class SegmentDatasetError(RuntimeError):
    """Raised when the segment dataset cannot be built safely."""


@dataclass(frozen=True)
class Segment:
    subject: str
    video_id: str
    sequence: str
    source_frame_start: int
    source_frame_end: int
    source_fps: float
    annotation_interval_corrected: bool
    included: bool
    exclusion_reason: str | None

    @property
    def song_id(self) -> str:
        return f"{self.subject}/{self.video_id}"

    @property
    def record_id(self) -> str:
        return f"{TARGET_DATASET}/{self.song_id}/{self.sequence}"

    @property
    def requested_frames(self) -> int:
        return self.source_frame_end - self.source_frame_start


@dataclass(frozen=True)
class VideoProbe:
    codec: str
    width: int
    height: int
    average_frame_rate: str
    frame_count: int | None
    duration_seconds: float
    audio_streams: int


@dataclass(frozen=True)
class VideoResult:
    status: str
    actual_frames: int | None
    requested_frames: int
    width: int | None
    height: int | None
    average_frame_rate: str | None
    duration_seconds: float | None
    error: str | None = None


def load_segments(records_path: Path) -> list[Segment]:
    with records_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    segments = [
        Segment(
            subject=str(record["subject"]),
            video_id=str(record["video_id"]),
            sequence=str(record["sequence"]),
            source_frame_start=int(record["source_frame_start"]),
            source_frame_end=int(record["source_frame_end"]),
            source_fps=float(record["source_fps"]),
            annotation_interval_corrected=bool(
                record["annotation_interval_corrected"]
            ),
            included=bool(record["included"]),
            exclusion_reason=record.get("exclusion_reason"),
        )
        for record in records
    ]
    segments.sort(key=lambda item: (item.subject, item.video_id, item.sequence))
    identities = [segment.record_id for segment in segments]
    if len(identities) != len(set(identities)):
        raise SegmentDatasetError(f"{records_path}: duplicate segment identity")
    for segment in segments:
        if segment.source_frame_start < 0:
            raise SegmentDatasetError(
                f"{segment.record_id}: source_frame_start must be nonnegative"
            )
        if segment.source_frame_end <= segment.source_frame_start:
            raise SegmentDatasetError(
                f"{segment.record_id}: invalid source frame interval"
            )
        if not math.isfinite(segment.source_fps) or segment.source_fps <= 0:
            raise SegmentDatasetError(
                f"{segment.record_id}: source_fps must be positive"
            )
    return segments


def read_song_split(path: Path) -> list[str]:
    prefix = f"{SOURCE_DATASET}/"
    song_ids: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(prefix):
            raise SegmentDatasetError(
                f"{path}: expected {SOURCE_DATASET!r} record, got {line!r}"
            )
        song_id = line.removeprefix(prefix)
        if song_id in seen:
            raise SegmentDatasetError(f"{path}: duplicate song {line!r}")
        seen.add(song_id)
        song_ids.append(song_id)
    return song_ids


def select_valid_segments(
    segments: Iterable[Segment], song_ids: Iterable[str]
) -> list[Segment]:
    by_song: dict[str, list[Segment]] = {}
    for segment in segments:
        by_song.setdefault(segment.song_id, []).append(segment)
    selected: list[Segment] = []
    for song_id in song_ids:
        song_segments = by_song.get(song_id)
        if song_segments is None:
            raise SegmentDatasetError(
                f"validation song has no segment metadata: {SOURCE_DATASET}/{song_id}"
            )
        selected.extend(song_segments)
    return selected


def source_audio_path(data_root: Path, segment: Segment) -> Path:
    return (
        data_root
        / "Audio"
        / SOURCE_DATASET
        / segment.subject
        / segment.video_id
        / f"{segment.sequence}.mp3"
    )


def target_audio_path(data_root: Path, segment: Segment) -> Path:
    return (
        data_root
        / "Audio"
        / TARGET_DATASET
        / segment.subject
        / segment.video_id
        / f"{segment.sequence}.mp3"
    )


def source_video_path(data_root: Path, segment: Segment) -> Path:
    return (
        data_root
        / "Video"
        / SOURCE_DATASET
        / segment.subject
        / f"{segment.video_id}.mp4"
    )


def target_video_path(data_root: Path, segment: Segment) -> Path:
    return (
        data_root
        / "Video"
        / TARGET_DATASET
        / segment.subject
        / segment.video_id
        / f"{segment.sequence}.mp4"
    )


def validate_audio_inventory(data_root: Path, segments: Iterable[Segment]) -> None:
    expected = {source_audio_path(data_root, segment) for segment in segments}
    target_expected = {target_audio_path(data_root, segment) for segment in segments}
    source_root = data_root / "Audio" / SOURCE_DATASET
    target_root = data_root / "Audio" / TARGET_DATASET
    actual_source = set(source_root.glob("*/*/*_seq_*.mp3"))
    actual_target = (
        set(target_root.glob("*/*/*_seq_*.mp3")) if target_root.exists() else set()
    )
    unknown_source = actual_source - expected
    unknown_target = actual_target - target_expected
    if unknown_source or unknown_target:
        unknown = sorted(unknown_source | unknown_target)
        preview = ", ".join(str(path) for path in unknown[:5])
        raise SegmentDatasetError(f"unindexed segment MP3(s): {preview}")
    missing = [
        segment.record_id
        for segment in segments
        if not source_audio_path(data_root, segment).is_file()
        and not target_audio_path(data_root, segment).is_file()
    ]
    if missing:
        raise SegmentDatasetError(
            f"missing {len(missing)} segment MP3(s), first: {missing[0]}"
        )


def _move_without_overwrite(source: Path, target: Path) -> str:
    source_exists = source.is_file()
    target_exists = target.is_file()
    if source_exists and target_exists:
        if source.samefile(target):
            source.unlink()
            return "resumed"
        raise SegmentDatasetError(f"refusing to overwrite existing file: {target}")
    if not source_exists and target_exists:
        return "already_moved"
    if not source_exists:
        raise FileNotFoundError(source)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise SegmentDatasetError(
            f"refusing to overwrite concurrently created file: {target}"
        ) from exc
    source.unlink()
    return "moved"


def move_audio_segments(data_root: Path, segments: list[Segment]) -> dict[str, int]:
    counts = {"moved": 0, "resumed": 0, "already_moved": 0}
    source_directories: set[Path] = set()
    for index, segment in enumerate(segments, start=1):
        source = source_audio_path(data_root, segment)
        target = target_audio_path(data_root, segment)
        source_directories.add(source.parent)
        counts[_move_without_overwrite(source, target)] += 1
        if index % 250 == 0 or index == len(segments):
            print(f"audio: {index}/{len(segments)}", flush=True)
    for directory in sorted(source_directories, reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return counts


def _run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise SegmentDatasetError(
            f"{description} failed ({result.returncode}):\n{result.stderr[-3000:]}"
        )
    return result


def probe_video(path: Path, ffprobe: str, *, count_frames: bool) -> VideoProbe:
    command = [ffprobe, "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            (
                "stream=codec_type,codec_name,width,height,avg_frame_rate,"
                "nb_frames,nb_read_frames,duration"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(_run(command, f"probe {path}").stdout)
    streams = payload["streams"]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise SegmentDatasetError(
            f"{path}: expected one video stream, found {len(videos)}"
        )
    video = videos[0]
    raw_count = video.get("nb_read_frames") or video.get("nb_frames")
    frame_count = (
        None if raw_count in (None, "N/A") else int(str(raw_count))
    )
    duration = video.get("duration") or payload["format"].get("duration")
    return VideoProbe(
        codec=str(video["codec_name"]),
        width=int(video["width"]),
        height=int(video["height"]),
        average_frame_rate=str(video["avg_frame_rate"]),
        frame_count=frame_count,
        duration_seconds=float(duration),
        audio_streams=sum(
            stream.get("codec_type") == "audio" for stream in streams
        ),
    )


def _validate_output_video(
    path: Path,
    segment: Segment,
    source_probe: VideoProbe,
    ffprobe: str,
) -> VideoResult:
    output = probe_video(path, ffprobe, count_frames=True)
    if output.codec != "h264":
        raise SegmentDatasetError(f"{path}: expected H.264, got {output.codec}")
    if (output.width, output.height) != (source_probe.width, source_probe.height):
        raise SegmentDatasetError(
            f"{path}: output geometry {(output.width, output.height)} does not "
            f"match source {(source_probe.width, source_probe.height)}"
        )
    if output.audio_streams:
        raise SegmentDatasetError(f"{path}: segment video must not contain audio")
    if output.frame_count is None or output.frame_count < 1:
        raise SegmentDatasetError(f"{path}: could not verify a positive frame count")
    if output.frame_count > segment.requested_frames:
        raise SegmentDatasetError(
            f"{path}: produced {output.frame_count} frames, expected at most "
            f"{segment.requested_frames}"
        )
    status = (
        "ready"
        if output.frame_count == segment.requested_frames
        else "truncated_source"
    )
    return VideoResult(
        status=status,
        actual_frames=output.frame_count,
        requested_frames=segment.requested_frames,
        width=output.width,
        height=output.height,
        average_frame_rate=output.average_frame_rate,
        duration_seconds=output.duration_seconds,
    )


def encode_video_segment(
    data_root: Path,
    segment: Segment,
    source_probe: VideoProbe,
    *,
    ffmpeg: str,
    ffprobe: str,
    preset: str,
    crf: int,
) -> VideoResult:
    source = source_video_path(data_root, segment)
    target = target_video_path(data_root, segment)
    if target.is_file():
        return _validate_output_video(target, segment, source_probe, ffprobe)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.partial-{os.getpid()}-{threading.get_ident()}-"
        f"{uuid.uuid4().hex}.mp4"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        (
            f"trim=start_frame={segment.source_frame_start}:"
            f"end_frame={segment.source_frame_end},setpts=PTS-STARTPTS"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        _run(command, f"encode {segment.record_id}")
        result = _validate_output_video(temporary, segment, source_probe, ffprobe)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise SegmentDatasetError(
                f"refusing to overwrite concurrently created file: {target}"
            ) from exc
        temporary.unlink()
        return result
    finally:
        temporary.unlink(missing_ok=True)


def build_validation_videos(
    data_root: Path,
    segments: list[Segment],
    *,
    ffmpeg: str,
    ffprobe: str,
    preset: str,
    crf: int,
    jobs: int,
) -> dict[str, VideoResult]:
    results: dict[str, VideoResult] = {}
    source_probes: dict[str, VideoProbe] = {}
    pending: list[Segment] = []
    for segment in segments:
        source = source_video_path(data_root, segment)
        if not source.is_file():
            results[segment.record_id] = VideoResult(
                status="missing_source_video",
                actual_frames=None,
                requested_frames=segment.requested_frames,
                width=None,
                height=None,
                average_frame_rate=None,
                duration_seconds=None,
            )
            continue
        if segment.song_id not in source_probes:
            source_probes[segment.song_id] = probe_video(
                source, ffprobe, count_frames=False
            )
        pending.append(segment)

    completed = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                encode_video_segment,
                data_root,
                segment,
                source_probes[segment.song_id],
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                preset=preset,
                crf=crf,
            ): segment
            for segment in pending
        }
        for future in as_completed(futures):
            segment = futures[future]
            try:
                results[segment.record_id] = future.result()
            except Exception as exc:
                results[segment.record_id] = VideoResult(
                    status="error",
                    actual_frames=None,
                    requested_frames=segment.requested_frames,
                    width=None,
                    height=None,
                    average_frame_rate=None,
                    duration_seconds=None,
                    error=str(exc),
                )
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(f"video: {completed}/{len(pending)}", flush=True)
    return results


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    data_root: Path,
    segments: list[Segment],
    valid_segments: list[Segment],
    video_results: dict[str, VideoResult],
    *,
    split_output_root: Path,
    metadata_output_root: Path,
) -> dict[str, int]:
    valid_ids = {segment.record_id for segment in valid_segments}
    ready_statuses = {"ready", "truncated_source"}
    ready_ids = {
        record_id
        for record_id, result in video_results.items()
        if result.status in ready_statuses
    }
    missing_ids = {
        record_id
        for record_id, result in video_results.items()
        if result.status == "missing_source_video"
    }
    error_ids = {
        record_id
        for record_id, result in video_results.items()
        if result.status == "error"
    }

    split_values = {
        "all": [segment.record_id for segment in segments],
        "valid": [segment.record_id for segment in valid_segments],
        "video_ready": sorted(ready_ids),
        "missing_video": sorted(missing_ids),
        "video_failed": sorted(error_ids),
    }
    for name, values in split_values.items():
        _atomic_write(
            split_output_root / f"pianomotion10mseg_{name}.txt",
            "".join(f"{value}\n" for value in values),
        )

    rows: list[dict[str, Any]] = []
    for segment in segments:
        video = video_results.get(segment.record_id)
        rows.append(
            {
                **asdict(segment),
                "record_id": segment.record_id,
                "song_id": f"{SOURCE_DATASET}/{segment.song_id}",
                "audio_path": target_audio_path(data_root, segment).relative_to(
                    data_root
                ).as_posix(),
                "video_path": (
                    target_video_path(data_root, segment).relative_to(data_root).as_posix()
                    if segment.record_id in valid_ids
                    else None
                ),
                "validation": segment.record_id in valid_ids,
                "requested_frames": segment.requested_frames,
                "video": None if video is None else asdict(video),
            }
        )
    _atomic_write(
        metadata_output_root / "segments.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    counts = {name: len(values) for name, values in split_values.items()}
    counts["truncated_source"] = sum(
        result.status == "truncated_source" for result in video_results.values()
    )
    _atomic_write(
        metadata_output_root / "summary.json",
        json.dumps(
            {
                "dataset": TARGET_DATASET,
                "source_dataset": SOURCE_DATASET,
                "counts": counts,
                "audio_relocated": True,
                "video_frame_policy": (
                    "exact source frame interval; preserve source timestamps; "
                    "no frame-rate conversion"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return counts


def _resolve_program(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SegmentDatasetError(f"required executable not found: {name}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--valid-split", type=Path, default=DEFAULT_VALID_SPLIT)
    parser.add_argument("--split-output-root", type=Path, default=DEFAULT_SPLIT_OUTPUT)
    parser.add_argument("--metadata-output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--video-jobs", type=int, default=2)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    parser.add_argument("--allow-missing-source-video", action="store_true")
    parser.add_argument("--skip-audio-move", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.video_jobs < 1:
        parser.error("--video-jobs must be positive")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be in [0, 51]")
    metadata_output_root = args.metadata_output_root or (
        args.data_root / "Metadata" / TARGET_DATASET
    )

    segments = load_segments(args.records)
    song_ids = read_song_split(args.valid_split)
    valid_segments = select_valid_segments(segments, song_ids)
    validate_audio_inventory(args.data_root, segments)
    missing_sources = {
        segment.song_id
        for segment in valid_segments
        if not source_video_path(args.data_root, segment).is_file()
    }
    missing_segments = sum(
        segment.song_id in missing_sources for segment in valid_segments
    )
    print(f"all segments: {len(segments)}")
    print(f"validation songs: {len(song_ids)}")
    print(f"validation segments: {len(valid_segments)}")
    print(f"missing source videos: {len(missing_sources)} songs, {missing_segments} segments")
    if missing_sources and not args.allow_missing_source_video:
        raise SegmentDatasetError(
            "source video is missing for validation songs; pass "
            "--allow-missing-source-video to build the available subset"
        )
    if args.dry_run:
        return

    if not args.skip_audio_move:
        audio_counts = move_audio_segments(args.data_root, segments)
        print(f"audio relocation: {audio_counts}")

    video_results: dict[str, VideoResult] = {}
    if not args.skip_video:
        free_gib = shutil.disk_usage(args.data_root).free / 2**30
        if free_gib < args.minimum_free_gib:
            raise SegmentDatasetError(
                f"only {free_gib:.1f} GiB free under {args.data_root}; "
                f"requires at least {args.minimum_free_gib:.1f} GiB"
            )
        ffmpeg = _resolve_program(args.ffmpeg)
        ffprobe = _resolve_program(args.ffprobe)
        video_results = build_validation_videos(
            args.data_root,
            valid_segments,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            preset=args.preset,
            crf=args.crf,
            jobs=args.video_jobs,
        )
    counts = write_outputs(
        args.data_root,
        segments,
        valid_segments,
        video_results,
        split_output_root=args.split_output_root,
        metadata_output_root=metadata_output_root,
    )
    print(f"outputs: {counts}")
    failures = [
        (record_id, result.error)
        for record_id, result in video_results.items()
        if result.status == "error"
    ]
    if failures:
        record_id, error = failures[0]
        raise SegmentDatasetError(
            f"{len(failures)} video clip(s) failed; first {record_id}: {error}"
        )


if __name__ == "__main__":
    main()
