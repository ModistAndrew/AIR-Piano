#!/usr/bin/env python3
"""Find exact and composition-level MIDI overlap between two split files.

The smaller split is used as the query set. Every cross-split pair receives a
cheap, transposition-invariant screening score; the strongest candidates are
then aligned with Parangonar's flexible DTW implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from parangonar.dp.dtw import FlexDynamicTimeWarping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MIDI_ROOT = Path("/storage/jinxuan/data/AirPianoFull/MIDI")

CHORD_TOLERANCE_SECONDS = 0.05
LOCAL_MATCH_THRESHOLD = 0.65
TRANSPOSE_MIN = -12
TRANSPOSE_MAX = 12
DEFAULT_SCREEN_CANDIDATES = 60
DEFAULT_LIKELY_THRESHOLD = 75.0
DEFAULT_POSSIBLE_THRESHOLD = 60.0
DEFAULT_SHOW_UNMATCHED = 10


@dataclass(frozen=True)
class SplitEntry:
    """One extensionless record identifier and its resolved MIDI path."""

    record_id: str
    path: Path


@dataclass(frozen=True)
class MidiProfile:
    """Compact symbolic features used for screening and alignment."""

    record_id: str
    path: Path
    file_sha256: str
    onsets: np.ndarray
    ends: np.ndarray
    pitches: tuple[np.ndarray, ...]
    pitch_classes: np.ndarray
    melody: np.ndarray
    bass: np.ndarray
    rhythm: np.ndarray
    pitch_histogram: np.ndarray
    pitch_class_histogram: np.ndarray
    melody_ngrams: frozenset[tuple[int, ...]]
    melody_contour_ngrams: frozenset[tuple[int, ...]]
    bass_ngrams: frozenset[tuple[int, ...]]
    chord_ngrams: frozenset[tuple[tuple[int, ...], ...]]
    rhythm_ngrams: frozenset[tuple[int, ...]]
    canonical_hash: str
    transposition_invariant_hash: str

    @property
    def event_count(self) -> int:
        return len(self.pitches)


@dataclass(frozen=True)
class LoadedSplit:
    """A resolved split and all profiles that were successfully parsed."""

    name: str
    path: Path
    requested_count: int
    profiles: tuple[MidiProfile, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class ScreenResult:
    """Cheap pairwise similarity measures used to select DTW candidates."""

    score: float
    melody: float
    bass: float
    chord: float
    rhythm: float
    histogram: float
    shifts: tuple[int, ...]
    exact_file: bool
    exact_chords: bool
    transposed_exact_chords: bool


@dataclass(frozen=True)
class AlignmentResult:
    """Detailed scores and coverage from one flexible DTW alignment."""

    score: float
    chord_score: float
    melody_score: float
    rhythm_score: float
    query_coverage: float
    reference_coverage: float
    shorter_coverage: float
    matched_query_events: int
    matched_reference_events: int
    matched_events: int
    transposition: int
    query_start: float
    query_end: float
    reference_start: float
    reference_end: float


@dataclass(frozen=True)
class PairResult:
    """One screened and aligned query/reference candidate pair."""

    query: MidiProfile
    reference: MidiProfile
    screen: ScreenResult
    alignment: AlignmentResult
    classification: str


class SplitError(ValueError):
    """Raised when a split cannot be resolved or completely loaded."""


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _ngrams(values: Sequence[object], size: int) -> frozenset[tuple]:
    if len(values) < size:
        return frozenset()
    return frozenset(
        tuple(values[index : index + size])
        for index in range(len(values) - size + 1)
    )


def _contour_bucket(interval: int) -> int:
    if interval <= -7:
        return -4
    if interval <= -3:
        return -3
    if interval < 0:
        return -2
    if interval == 0:
        return 0
    if interval < 3:
        return 2
    if interval < 7:
        return 3
    return 4


def _hash_tokens(tokens: object) -> str:
    return hashlib.sha256(repr(tokens).encode("utf-8")).hexdigest()


def _relative_rhythm(onsets: np.ndarray) -> np.ndarray:
    intervals = np.diff(onsets)
    positive = intervals[intervals > 1e-6]
    median_interval = float(np.median(positive)) if len(positive) else 1.0
    event_intervals = np.concatenate(([median_interval], intervals))
    return np.clip(
        np.log2(np.maximum(event_intervals, 1e-4) / median_interval),
        -3.0,
        3.0,
    ).astype(np.float32)


def read_profile(path: Path, record_id: str, reader: object) -> MidiProfile:
    """Parse one MIDI and construct grouped-onset symbolic features."""

    notes = reader.read(path)
    if not notes:
        raise ValueError("no positive-duration notes")

    pitch_groups: list[np.ndarray] = []
    onsets: list[float] = []
    ends: list[float] = []
    note_index = 0
    while note_index < len(notes):
        anchor = float(notes[note_index].onset)
        group = []
        while (
            note_index < len(notes)
            and float(notes[note_index].onset)
            <= anchor + CHORD_TOLERANCE_SECONDS
        ):
            group.append(notes[note_index])
            note_index += 1
        pitch_groups.append(
            np.asarray(sorted({int(note.pitch) for note in group}), dtype=np.int16)
        )
        onsets.append(anchor)
        ends.append(max(float(note.offset) for note in group))

    onset_array = np.asarray(onsets, dtype=np.float64)
    end_array = np.asarray(ends, dtype=np.float64)
    melody = np.asarray([pitches[-1] for pitches in pitch_groups], dtype=np.int16)
    bass = np.asarray([pitches[0] for pitches in pitch_groups], dtype=np.int16)
    rhythm = _relative_rhythm(onset_array)

    pitch_classes = np.zeros((len(pitch_groups), 12), dtype=np.float32)
    pitch_histogram = np.zeros(128, dtype=np.float64)
    pitch_class_histogram = np.zeros(12, dtype=np.float64)
    for index, pitches in enumerate(pitch_groups):
        pitch_classes[index, np.unique(pitches % 12)] = 1.0
        pitch_histogram[pitches] += 1.0
        pitch_class_histogram[pitches % 12] += 1.0

    melody_intervals = np.clip(np.diff(melody), -24, 24).astype(int).tolist()
    bass_intervals = np.clip(np.diff(bass), -24, 24).astype(int).tolist()
    contour = [_contour_bucket(interval) for interval in melody_intervals]
    chord_shapes = [
        tuple(int(pitch - pitches[0]) for pitch in pitches)
        for pitches in pitch_groups
    ]
    rhythm_tokens = np.rint(rhythm * 4.0).astype(int).tolist()
    canonical = tuple(tuple(int(pitch) for pitch in pitches) for pitches in pitch_groups)
    first_melody_pitch = int(melody[0])
    transposition_invariant = tuple(
        tuple(int(pitch) - first_melody_pitch for pitch in pitches)
        for pitches in pitch_groups
    )

    return MidiProfile(
        record_id=record_id,
        path=path,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        onsets=onset_array,
        ends=end_array,
        pitches=tuple(pitch_groups),
        pitch_classes=pitch_classes,
        melody=melody,
        bass=bass,
        rhythm=rhythm,
        pitch_histogram=_normalise(pitch_histogram),
        pitch_class_histogram=_normalise(pitch_class_histogram),
        melody_ngrams=_ngrams(melody_intervals, 5),
        melody_contour_ngrams=_ngrams(contour, 7),
        bass_ngrams=_ngrams(bass_intervals, 5),
        chord_ngrams=_ngrams(chord_shapes, 3),
        rhythm_ngrams=_ngrams(rhythm_tokens, 5),
        canonical_hash=_hash_tokens(canonical),
        transposition_invariant_hash=_hash_tokens(transposition_invariant),
    )


def _set_containment(left: frozenset, right: frozenset) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not left_norm or not right_norm:
        return 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), 0.0, 1.0))


def _shift_pitch_histogram(histogram: np.ndarray, semitones: int) -> np.ndarray:
    shifted = np.zeros_like(histogram)
    if semitones >= 0:
        shifted[semitones:] = histogram[: 128 - semitones]
    else:
        shifted[:semitones] = histogram[-semitones:]
    return shifted


def shift_scores(query: MidiProfile, reference: MidiProfile) -> list[tuple[float, int]]:
    """Rank shifts that transpose the reference into the query's pitch space."""

    scores = []
    for shift in range(TRANSPOSE_MIN, TRANSPOSE_MAX + 1):
        pitch_class_score = _cosine_similarity(
            query.pitch_class_histogram,
            np.roll(reference.pitch_class_histogram, shift),
        )
        absolute_score = _cosine_similarity(
            query.pitch_histogram,
            _shift_pitch_histogram(reference.pitch_histogram, shift),
        )
        scores.append((0.7 * pitch_class_score + 0.3 * absolute_score, shift))
    scores.sort(key=lambda value: (value[0], -abs(value[1]), -value[1]), reverse=True)
    return scores


def screen_pair(query: MidiProfile, reference: MidiProfile) -> ScreenResult:
    melody_exact = _set_containment(query.melody_ngrams, reference.melody_ngrams)
    melody_contour = _set_containment(
        query.melody_contour_ngrams,
        reference.melody_contour_ngrams,
    )
    melody = max(melody_exact, 0.7 * melody_contour)
    bass = _set_containment(query.bass_ngrams, reference.bass_ngrams)
    chord = _set_containment(query.chord_ngrams, reference.chord_ngrams)
    rhythm = _set_containment(query.rhythm_ngrams, reference.rhythm_ngrams)
    ranked_shifts = shift_scores(query, reference)
    histogram = ranked_shifts[0][0]
    total = (
        0.35 * melody
        + 0.15 * bass
        + 0.20 * chord
        + 0.15 * rhythm
        + 0.15 * histogram
    )
    return ScreenResult(
        score=100.0 * total,
        melody=100.0 * melody,
        bass=100.0 * bass,
        chord=100.0 * chord,
        rhythm=100.0 * rhythm,
        histogram=100.0 * histogram,
        shifts=tuple(shift for _, shift in ranked_shifts[:3]),
        exact_file=query.file_sha256 == reference.file_sha256,
        exact_chords=query.canonical_hash == reference.canonical_hash,
        transposed_exact_chords=(
            query.transposition_invariant_hash
            == reference.transposition_invariant_hash
        ),
    )


def select_candidate_indices(
    screened: Sequence[ScreenResult],
    limit: int,
) -> list[int]:
    """Retain candidates strong under the combined or a component score."""

    component_count = max(1, limit // 4)
    candidates: set[int] = set()
    for field, count in (
        ("score", limit),
        ("melody", component_count),
        ("bass", component_count),
        ("chord", component_count),
        ("rhythm", component_count),
        ("histogram", component_count),
    ):
        candidates.update(
            sorted(
                range(len(screened)),
                key=lambda index: (
                    float(getattr(screened[index], field)),
                    -index,
                ),
                reverse=True,
            )[:count]
        )
    candidates.update(
        index
        for index, result in enumerate(screened)
        if result.exact_file
        or result.exact_chords
        or result.transposed_exact_chords
    )
    return sorted(
        candidates,
        key=lambda index: (
            screened[index].exact_file,
            screened[index].exact_chords,
            screened[index].transposed_exact_chords,
            max(
                screened[index].score,
                0.8 * screened[index].melody,
                0.8 * screened[index].chord,
                0.8 * screened[index].rhythm,
            ),
            -index,
        ),
        reverse=True,
    )[:limit]


def _absolute_pitch_matrix(profile: MidiProfile, shift: int = 0) -> np.ndarray:
    matrix = np.zeros((profile.event_count, 128), dtype=np.float32)
    for index, pitches in enumerate(profile.pitches):
        shifted = pitches.astype(np.int16) + shift
        valid = shifted[(shifted >= 0) & (shifted < 128)]
        matrix[index, valid] = 1.0
    return matrix


def _f1_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersections = left @ right.T
    denominator = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
    return np.divide(
        2.0 * intersections,
        denominator,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=denominator > 0,
    )


def _flexible_path(
    cost: np.ndarray,
    aligner: FlexDynamicTimeWarping,
    *,
    cover_rows: bool,
) -> np.ndarray:
    """Return a flexible path constrained to cover the complete query axis.

    FlexDTW normally permits either input to be a local excerpt. The boundary
    penalties retain its flexible location on the reference axis while forcing
    the first and last query events into the path.
    """

    if cover_rows and cost.shape[1] == 1:
        return np.column_stack(
            (
                np.arange(cost.shape[0], dtype=np.int64),
                np.zeros(cost.shape[0], dtype=np.int64),
            )
        )
    if not cover_rows and cost.shape[0] == 1:
        return np.column_stack(
            (
                np.zeros(cost.shape[1], dtype=np.int64),
                np.arange(cost.shape[1], dtype=np.int64),
            )
        )
    if cost.shape[0] == 1 or cost.shape[1] == 1:
        raise AssertionError("unexpected singleton reference axis")

    constrained = cost.copy()
    boundary_penalty = float(cost.max()) + 1_000_000.0
    if cover_rows:
        # Start on the top edge and finish on the bottom edge.
        constrained[1:, 0] = boundary_penalty
        constrained[:-1, -1] = boundary_penalty
    else:
        # Start on the left edge and finish on the right edge.
        constrained[0, 1:] = boundary_penalty
        constrained[-1, :-1] = boundary_penalty
    path = aligner.from_distance_matrix(constrained)[0]
    return np.asarray(path, dtype=np.int64)


def align_pair(
    query: MidiProfile,
    reference: MidiProfile,
    shift: int,
    aligner: FlexDynamicTimeWarping,
) -> AlignmentResult:
    """Align a reference-to-query transposition and score the resulting path."""

    query_pitch_matrix = _absolute_pitch_matrix(query)
    reference_pitch_matrix = _absolute_pitch_matrix(reference, shift)
    absolute_pitch_score = _f1_matrix(query_pitch_matrix, reference_pitch_matrix)
    pitch_class_score = _f1_matrix(
        query.pitch_classes,
        np.roll(reference.pitch_classes, shift, axis=1),
    )
    chord_score = 0.75 * absolute_pitch_score + 0.25 * pitch_class_score

    melody_distance = np.abs(
        query.melody.astype(np.float32)[:, None]
        - (reference.melody.astype(np.float32)[None, :] + shift)
    )
    bass_distance = np.abs(
        query.bass.astype(np.float32)[:, None]
        - (reference.bass.astype(np.float32)[None, :] + shift)
    )
    melody_score = (
        0.75 * np.exp(-melody_distance / 2.0)
        + 0.25 * np.exp(-bass_distance / 3.0)
    )
    rhythm_distance = np.abs(query.rhythm[:, None] - reference.rhythm[None, :])
    rhythm_score = np.exp(-rhythm_distance)
    combined_score = (
        0.5 * chord_score + 0.3 * melody_score + 0.2 * rhythm_score
    ).astype(np.float32)

    query_is_shorter = query.event_count <= reference.event_count
    cost = 1.0 - combined_score
    oriented_cost = cost if query_is_shorter else cost.T
    oriented_path = _flexible_path(
        oriented_cost,
        aligner,
        cover_rows=query_is_shorter,
    )
    if query_is_shorter:
        query_indices = oriented_path[:, 0]
        reference_indices = oriented_path[:, 1]
    else:
        query_indices = oriented_path[:, 1]
        reference_indices = oriented_path[:, 0]

    local_combined = combined_score[query_indices, reference_indices]
    selected_positions = []
    for query_index in np.unique(query_indices):
        positions = np.flatnonzero(query_indices == query_index)
        selected_positions.append(int(positions[np.argmax(local_combined[positions])]))
    selected = np.asarray(selected_positions, dtype=np.int64)
    selected_query = query_indices[selected]
    selected_reference = reference_indices[selected]

    best_query_scores = np.zeros(query.event_count, dtype=np.float32)
    best_reference_scores = np.zeros(reference.event_count, dtype=np.float32)
    np.maximum.at(best_query_scores, query_indices, local_combined)
    np.maximum.at(best_reference_scores, reference_indices, local_combined)
    matched_query = best_query_scores >= LOCAL_MATCH_THRESHOLD
    matched_reference = best_reference_scores >= LOCAL_MATCH_THRESHOLD
    matched_path = local_combined >= LOCAL_MATCH_THRESHOLD
    if not np.any(matched_path):
        matched_path = np.ones_like(local_combined, dtype=bool)

    range_query = query_indices[matched_path]
    range_reference = reference_indices[matched_path]
    matched_query_count = int(matched_query.sum())
    matched_reference_count = int(matched_reference.sum())
    return AlignmentResult(
        score=100.0 * float(combined_score[selected_query, selected_reference].mean()),
        chord_score=100.0 * float(
            chord_score[selected_query, selected_reference].mean()
        ),
        melody_score=100.0 * float(
            melody_score[selected_query, selected_reference].mean()
        ),
        rhythm_score=100.0 * float(
            rhythm_score[selected_query, selected_reference].mean()
        ),
        query_coverage=float(matched_query.mean()),
        reference_coverage=float(matched_reference.mean()),
        shorter_coverage=(
            float(matched_query.mean())
            if query_is_shorter
            else float(matched_reference.mean())
        ),
        matched_query_events=matched_query_count,
        matched_reference_events=matched_reference_count,
        matched_events=min(matched_query_count, matched_reference_count),
        transposition=shift,
        query_start=float(query.onsets[range_query].min()),
        query_end=float(query.ends[range_query].max()),
        reference_start=float(reference.onsets[range_reference].min()),
        reference_end=float(reference.ends[range_reference].max()),
    )


def best_alignment(
    query: MidiProfile,
    reference: MidiProfile,
    shifts: Sequence[int],
    aligner: FlexDynamicTimeWarping,
) -> AlignmentResult:
    alignments = [align_pair(query, reference, shift, aligner) for shift in shifts]
    return max(
        alignments,
        key=lambda result: (
            result.score,
            result.query_coverage,
            result.shorter_coverage,
            -abs(result.transposition),
        ),
    )


def classify(
    screen: ScreenResult,
    alignment: AlignmentResult,
    *,
    likely_threshold: float,
    possible_threshold: float,
) -> str:
    if screen.exact_file:
        return "likely"
    if (
        alignment.score >= likely_threshold
        and alignment.query_coverage >= 0.50
        and alignment.matched_events >= 40
    ):
        return "likely"
    if (
        alignment.score >= possible_threshold
        and alignment.query_coverage >= 0.30
        and alignment.matched_events >= 20
    ):
        return "possible"
    return "none"


def resolve_split_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise SplitError(f"split file not found: {path}")
    return path.resolve()


def read_split_entries(split_path: Path, midi_root: Path) -> list[SplitEntry]:
    record_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not record_ids:
        raise SplitError(f"split is empty: {split_path}")
    entries = []
    for record_id in record_ids:
        relative = Path(record_id)
        midi_relative = (
            relative
            if relative.suffix.lower() in {".mid", ".midi"}
            else Path(f"{record_id}.mid")
        )
        entries.append(SplitEntry(record_id=record_id, path=midi_root / midi_relative))
    return entries


def load_split(
    value: str | Path,
    midi_root: Path,
    reader: object,
    *,
    skip_missing: bool,
    profile_cache: dict[Path, MidiProfile] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LoadedSplit:
    split_path = resolve_split_path(value)
    entries = read_split_entries(split_path, midi_root)
    missing = [entry for entry in entries if not entry.path.is_file()]
    if missing and not skip_missing:
        paths = "\n".join(f"  {entry.path}" for entry in missing)
        raise SplitError(
            f"{split_path} references {len(missing)} missing MIDI file(s):\n{paths}"
        )

    cache = profile_cache if profile_cache is not None else {}
    profiles: list[MidiProfile] = []
    skipped = [f"missing: {entry.path}" for entry in missing]
    available = [entry for entry in entries if entry.path.is_file()]
    for index, entry in enumerate(available, start=1):
        cache_key = entry.path.resolve()
        try:
            if cache_key in cache:
                profile = replace(cache[cache_key], record_id=entry.record_id)
            else:
                profile = read_profile(entry.path, entry.record_id, reader)
                cache[cache_key] = profile
            profiles.append(profile)
        except Exception as error:
            message = f"unreadable: {entry.path}: {error}"
            if not skip_missing:
                raise SplitError(message) from error
            skipped.append(message)
        if progress and (index % 100 == 0 or index == len(available)):
            progress(f"[parse] {split_path.stem}: {index}/{len(available)}")

    if not profiles:
        raise SplitError(f"split has no usable MIDI files: {split_path}")
    return LoadedSplit(
        name=split_path.stem,
        path=split_path,
        requested_count=len(entries),
        profiles=tuple(profiles),
        skipped=tuple(skipped),
    )


def compare_splits(
    query: LoadedSplit,
    reference: LoadedSplit,
    *,
    screen_candidates: int,
    likely_threshold: float,
    possible_threshold: float,
    progress: Callable[[str], None] | None = None,
) -> list[PairResult]:
    aligner = FlexDynamicTimeWarping(buffer=1)
    results: list[PairResult] = []
    for query_index, query_profile in enumerate(query.profiles, start=1):
        if progress:
            progress(
                f"[compare] {query.name}: {query_index}/{len(query.profiles)} "
                f"({query_profile.record_id})"
            )
        screened = [
            screen_pair(query_profile, reference_profile)
            for reference_profile in reference.profiles
        ]
        candidate_indices = select_candidate_indices(screened, screen_candidates)
        for reference_index in candidate_indices:
            reference_profile = reference.profiles[reference_index]
            screen = screened[reference_index]
            alignment = best_alignment(
                query_profile,
                reference_profile,
                screen.shifts,
                aligner,
            )
            results.append(
                PairResult(
                    query=query_profile,
                    reference=reference_profile,
                    screen=screen,
                    alignment=alignment,
                    classification=classify(
                        screen,
                        alignment,
                        likely_threshold=likely_threshold,
                        possible_threshold=possible_threshold,
                    ),
                )
            )
    return results


def _format_pair(result: PairResult, query_is_left: bool) -> str:
    if query_is_left:
        left = result.query
        right = result.reference
        left_coverage = result.alignment.query_coverage
        right_coverage = result.alignment.reference_coverage
        shift_text = f"{right.record_id} -> {left.record_id}"
        left_range = (result.alignment.query_start, result.alignment.query_end)
        right_range = (
            result.alignment.reference_start,
            result.alignment.reference_end,
        )
    else:
        left = result.reference
        right = result.query
        left_coverage = result.alignment.reference_coverage
        right_coverage = result.alignment.query_coverage
        shift_text = f"{left.record_id} -> {right.record_id}"
        left_range = (
            result.alignment.reference_start,
            result.alignment.reference_end,
        )
        right_range = (result.alignment.query_start, result.alignment.query_end)

    exact = []
    if result.screen.exact_file:
        exact.append("byte-identical")
    if result.screen.exact_chords:
        exact.append("identical chord sequence")
    elif result.screen.transposed_exact_chords:
        exact.append("transposed-identical chord sequence")
    exact_text = f"; {', '.join(exact)}" if exact else ""
    return (
        f"{result.classification.upper():8} score={result.alignment.score:6.2f} "
        f"chord={result.alignment.chord_score:6.2f} "
        f"melody={result.alignment.melody_score:6.2f} "
        f"rhythm={result.alignment.rhythm_score:6.2f}\n"
        f"  left:  {left.record_id}  coverage={left_coverage:6.1%} "
        f"range={left_range[0]:.2f}-{left_range[1]:.2f}s\n"
        f"  right: {right.record_id}  coverage={right_coverage:6.1%} "
        f"range={right_range[0]:.2f}-{right_range[1]:.2f}s\n"
        f"  transpose {shift_text}: {result.alignment.transposition:+d} semitones; "
        f"matched_events={result.alignment.matched_events}{exact_text}"
    )


def print_report(
    left: LoadedSplit,
    right: LoadedSplit,
    results: Sequence[PairResult],
    *,
    query_is_left: bool,
    show_unmatched: int,
) -> None:
    flagged = sorted(
        (result for result in results if result.classification != "none"),
        key=lambda result: (
            result.classification == "likely",
            result.alignment.score,
            result.alignment.query_coverage,
            result.query.record_id,
            result.reference.record_id,
        ),
        reverse=True,
    )
    likely = [result for result in flagged if result.classification == "likely"]
    possible = [result for result in flagged if result.classification == "possible"]
    matched_queries = {result.query.record_id for result in flagged}
    query = left if query_is_left else right
    reference = right if query_is_left else left

    print("MIDI split similarity report")
    print(f"left:      {left.name} ({len(left.profiles)}/{left.requested_count} usable)")
    print(f"right:     {right.name} ({len(right.profiles)}/{right.requested_count} usable)")
    print(f"query:     {query.name} (smaller split)")
    print(f"reference: {reference.name}")
    print(f"aligned candidate pairs: {len(results)}")
    print(f"likely pairs:             {len(likely)}")
    print(f"possible pairs:           {len(possible)}")
    print(
        f"queries with no credible match: "
        f"{len(query.profiles) - len(matched_queries)}/{len(query.profiles)}"
    )
    if left.skipped or right.skipped:
        print(f"skipped files:            {len(left.skipped) + len(right.skipped)}")
        for item in (*left.skipped, *right.skipped):
            print(f"  {item}")

    print("\nLikely and possible matches")
    if not flagged:
        print("  none")
    else:
        for result in flagged:
            print(_format_pair(result, query_is_left))

    unmatched = sorted(
        (result for result in results if result.classification == "none"),
        key=lambda result: (
            result.alignment.score,
            result.alignment.query_coverage,
            result.query.record_id,
            result.reference.record_id,
        ),
        reverse=True,
    )[:show_unmatched]
    print(f"\nStrongest unmatched pairs ({len(unmatched)})")
    if not unmatched:
        print("  none")
    else:
        for result in unmatched:
            print(_format_pair(result, query_is_left))



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check two AirPiano split files for similar MIDI compositions."
    )
    parser.add_argument(
        "left_split",
        type=Path,
        help="path to the first split file",
    )
    parser.add_argument(
        "right_split",
        type=Path,
        help="path to the second split file",
    )
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=DEFAULT_MIDI_ROOT,
        help=f"MIDI root (default: {DEFAULT_MIDI_ROOT})",
    )
    parser.add_argument(
        "--screen-candidates",
        type=int,
        default=DEFAULT_SCREEN_CANDIDATES,
        help="detailed DTW candidates per query MIDI (default: 60)",
    )
    parser.add_argument(
        "--likely-threshold",
        type=float,
        default=DEFAULT_LIKELY_THRESHOLD,
        help="minimum 0-100 score for likely matches (default: 75)",
    )
    parser.add_argument(
        "--possible-threshold",
        type=float,
        default=DEFAULT_POSSIBLE_THRESHOLD,
        help="minimum 0-100 score for possible matches (default: 60)",
    )
    parser.add_argument(
        "--show-unmatched",
        type=int,
        default=DEFAULT_SHOW_UNMATCHED,
        help="number of strongest non-matches to print (default: 10)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="continue after missing or unreadable MIDIs and report reduced coverage",
    )
    args = parser.parse_args(argv)
    if args.screen_candidates < 1:
        parser.error("--screen-candidates must be at least 1")
    if args.show_unmatched < 0:
        parser.error("--show-unmatched cannot be negative")
    if not 0.0 <= args.possible_threshold <= 100.0:
        parser.error("--possible-threshold must be between 0 and 100")
    if not 0.0 <= args.likely_threshold <= 100.0:
        parser.error("--likely-threshold must be between 0 and 100")
    if args.possible_threshold > args.likely_threshold:
        parser.error("--possible-threshold cannot exceed --likely-threshold")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        midi_root = args.midi_root.expanduser()
        # Import here so --help and argument validation remain lightweight.
        from airpiano.data.midi import MidoNoteReader

        cache: dict[Path, MidiProfile] = {}
        progress = lambda message: print(message, file=sys.stderr, flush=True)
        left = load_split(
            args.left_split,
            midi_root,
            MidoNoteReader(),
            skip_missing=args.skip_missing,
            profile_cache=cache,
            progress=progress,
        )
        right = load_split(
            args.right_split,
            midi_root,
            MidoNoteReader(),
            skip_missing=args.skip_missing,
            profile_cache=cache,
            progress=progress,
        )
        query_is_left = len(left.profiles) <= len(right.profiles)
        query = left if query_is_left else right
        reference = right if query_is_left else left
        results = compare_splits(
            query,
            reference,
            screen_candidates=min(args.screen_candidates, len(reference.profiles)),
            likely_threshold=args.likely_threshold,
            possible_threshold=args.possible_threshold,
            progress=progress,
        )
        print_report(
            left,
            right,
            results,
            query_is_left=query_is_left,
            show_unmatched=args.show_unmatched,
        )
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
