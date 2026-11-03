"""Physical release accuracy and complete-recording validation checks."""
from __future__ import annotations

import unittest

from airpiano import data
from scripts.validate_streaming import release_metrics, validate_complete


def _chord(slot: int, onset: float) -> data.Chord:
    note = data.Note(slot, onset, onset + 0.2, 60 + slot, 64, slot * 24, slot * 24 + 12)
    return data.Chord(slot, (note,), onset, slot * 24, 0.2, 12)


class StreamingValidationTests(unittest.TestCase):
    def test_physical_time_and_missing_chord_denominator(self) -> None:
        chords = (_chord(0, 0.5), _chord(1, 1.0))
        metrics = release_metrics(chords, [
            {"kind": "trigger", "chord_id": 1, "release_time": 0.0},
            {"kind": "release", "chord_id": 1, "release_time": 0.5},
        ], frame_count=60)
        self.assertEqual(metrics["ground_truth_chord_count"], 2)
        self.assertEqual(metrics["detected_chord_count"], 1)
        self.assertEqual(metrics["correct_chord_counts"]["8"], 1)
        self.assertEqual(metrics["c_ar"]["8"], 0.5)
        self.assertEqual(metrics["onset_mae_seconds"], 0.0)

    def test_fractional_release_is_not_rounded_into_tolerance(self) -> None:
        metrics = release_metrics((_chord(0, 0.5),), [
            {"kind": "release", "chord_id": 1, "release_time": 0.5 + 8.1 / 30},
        ], frame_count=60)
        self.assertEqual(metrics["c_ar"]["8"], 0.0)
        self.assertEqual(metrics["c_ar"]["16"], 1.0)

    def test_duplicate_and_unknown_release_ids_rejected(self) -> None:
        event = {"kind": "release", "chord_id": 1, "release_time": 0.5}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            release_metrics((_chord(0, 0.5),), [event, event], frame_count=60)
        with self.assertRaisesRegex(ValueError, "unknown"):
            release_metrics((_chord(0, 0.5),), [dict(event, chord_id=2)], frame_count=60)

    def test_complete_capture_preserves_dropped_positions(self) -> None:
        summary = {"status": "completed", "source_frame_count": 60,
                   "processed_frames": 58, "dropped_frames": 2}
        validate_complete(summary, 60)
        with self.assertRaisesRegex(ValueError, "Incomplete source"):
            validate_complete(summary, 61)
        with self.assertRaisesRegex(ValueError, "Incomplete streaming"):
            validate_complete(dict(summary, status="interrupted"), 60)
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_complete(dict(summary, dropped_frames=0), 60)


if __name__ == "__main__":
    unittest.main()
