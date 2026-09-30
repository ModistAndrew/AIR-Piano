"""Tests for the PianoMotion10MSeg dataset builder."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.build_pianomotion_segment_dataset import (
    Segment,
    encode_video_segment,
    load_segments,
    move_audio_segments,
    probe_video,
    read_song_split,
    select_valid_segments,
    source_audio_path,
    target_audio_path,
)


def _segment(sequence: str, *, start: int = 2, end: int = 7) -> Segment:
    return Segment(
        subject="subject",
        video_id="video",
        sequence=sequence,
        source_frame_start=start,
        source_frame_end=end,
        source_fps=5.0,
        annotation_interval_corrected=False,
        included=True,
        exclusion_reason=None,
    )


class SegmentDatasetTest(unittest.TestCase):
    def test_load_and_expand_song_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.json"
            records.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "subject": "subject",
                                "video_id": "video",
                                "sequence": "video_seq_0001",
                                "source_frame_start": 4,
                                "source_frame_end": 9,
                                "source_fps": 5,
                                "annotation_interval_corrected": False,
                                "included": False,
                                "exclusion_reason": "visibility",
                            },
                            {
                                "subject": "subject",
                                "video_id": "video",
                                "sequence": "video_seq_0000",
                                "source_frame_start": 0,
                                "source_frame_end": 5,
                                "source_fps": 5,
                                "annotation_interval_corrected": False,
                                "included": True,
                                "exclusion_reason": None,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            split = root / "valid.txt"
            split.write_text(
                "PianoMotion10M/subject/video\n", encoding="utf-8"
            )

            segments = load_segments(records)
            selected = select_valid_segments(segments, read_song_split(split))

            self.assertEqual(
                [segment.sequence for segment in selected],
                ["video_seq_0000", "video_seq_0001"],
            )
            self.assertEqual(
                [segment.record_id for segment in selected],
                [
                    "PianoMotion10MSeg/subject/video/video_seq_0000",
                    "PianoMotion10MSeg/subject/video/video_seq_0001",
                ],
            )

    def test_audio_move_preserves_full_song_audio_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = [_segment("video_seq_0000"), _segment("video_seq_0001")]
            full_audio = root / "Audio/PianoMotion10M/subject/video.m4a"
            full_audio.parent.mkdir(parents=True)
            full_audio.write_bytes(b"full-song")
            for index, segment in enumerate(segments):
                path = source_audio_path(root, segment)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"segment-{index}".encode())

            counts = move_audio_segments(root, segments)

            self.assertEqual(counts, {"moved": 2, "resumed": 0, "already_moved": 0})
            self.assertEqual(full_audio.read_bytes(), b"full-song")
            self.assertFalse(source_audio_path(root, segments[0]).exists())
            self.assertEqual(
                target_audio_path(root, segments[0]).read_bytes(), b"segment-0"
            )
            self.assertEqual(
                move_audio_segments(root, segments),
                {"moved": 0, "resumed": 0, "already_moved": 2},
            )

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg and FFprobe are required",
    )
    def test_video_clip_uses_exact_frame_count_without_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segment = _segment("video_seq_0000", start=2, end=7)
            source = root / "Video/PianoMotion10M/subject/video.mp4"
            source.parent.mkdir(parents=True)
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=64x48:rate=5:duration=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            ffprobe = shutil.which("ffprobe") or "ffprobe"
            source_probe = probe_video(source, ffprobe, count_frames=True)

            result = encode_video_segment(
                root,
                segment,
                source_probe,
                ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
                ffprobe=ffprobe,
                preset="ultrafast",
                crf=18,
            )

            self.assertEqual(result.status, "ready")
            self.assertEqual(result.actual_frames, 5)
            self.assertEqual((result.width, result.height), (64, 48))
            output = root / "Video/PianoMotion10MSeg/subject/video/video_seq_0000.mp4"
            self.assertTrue(output.is_file())
            self.assertEqual(
                probe_video(output, ffprobe, count_frames=True).audio_streams, 0
            )


if __name__ == "__main__":
    unittest.main()
