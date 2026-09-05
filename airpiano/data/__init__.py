"""Public data API for AirPiano."""

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
    CenteredXYZ10Transform,
    HandLoader,
)
from .midi import InitializerChordMerger, MidiLoader, MidoNoteReader
from .slicing import FutureChordSlicer
from .targets import ChordOnsetEvents, TargetGenerator, TriangularSoftener
from .types import Chord, FilePair, HandSequence, MidiSequence, Note, Song

__all__ = [
    "AlignmentDataset",
    "CANONICAL_HAND_SCHEMA",
    "CanonicalNpzReader",
    "CenteredXYZ10Transform",
    "Chord",
    "ChordOnsetEvents",
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
    "Song",
    "RelativeStemPairer",
    "TargetGenerator",
    "TriangularSoftener",
    "build_data_components",
    "build_dataset",
    "collate_alignment_batch",
    "make_loader",
]
