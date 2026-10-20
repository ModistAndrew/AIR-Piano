"""Public data API for AirPiano."""

from .augmentation import WindowAugmenter
from .dataset import (
    AlignmentDataset,
    build_data_components,
    build_dataset,
    collate_alignment_batch,
    make_loader,
)
from .files import EagerSongReader, FileLoader, RelativeStemPairer
from .hands import (
    CANONICAL_HAND_SCHEMA,
    CanonicalNpzReader,
    HandLoader,
    OffsetScaleTransform,
)
from .midi import InitializerChordMerger, MidiLoader, MidoNoteReader
from .slicing import FutureChordSlicer
from .targets import ChordOnsetEvents, TargetGenerator, TriangularSoftener
from .tokenization import ChordTokenizer
from .types import (
    AlignmentWindow,
    Chord,
    FilePair,
    HandSequence,
    MidiSequence,
    Note,
    Song,
)

__all__ = [
    "AlignmentDataset",
    "AlignmentWindow",
    "CANONICAL_HAND_SCHEMA",
    "CanonicalNpzReader",
    "Chord",
    "ChordOnsetEvents",
    "ChordTokenizer",
    "EagerSongReader",
    "FileLoader",
    "FilePair",
    "FutureChordSlicer",
    "HandLoader",
    "HandSequence",
    "InitializerChordMerger",
    "MidiLoader",
    "MidiSequence",
    "MidoNoteReader",
    "Note",
    "OffsetScaleTransform",
    "Song",
    "RelativeStemPairer",
    "TargetGenerator",
    "TriangularSoftener",
    "WindowAugmenter",
    "build_data_components",
    "build_dataset",
    "collate_alignment_batch",
    "make_loader",
]
