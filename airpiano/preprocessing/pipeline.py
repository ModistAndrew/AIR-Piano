"""Configuration-facing orchestration for offline hand-pose preprocessing."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig

from airpiano.data.files import read_record_stems
from airpiano.utils import resolve_path


def preprocess(cfg: DictConfig) -> None:
    """Run the configured WiLoR preprocessing phase or both phases in order."""

    from airpiano.preprocessing.canonical import raw_cache_has_markers

    task = str(cfg.preprocess.task)
    if task not in {"video_to_ragged", "ragged_to_canonical", "all"}:
        raise ValueError(f"unknown preprocess task: {task}")

    split_files = [resolve_path(path) for path in cfg.preprocess.split_files]
    records = read_record_stems(split_files)
    video_root = resolve_path(cfg.data.video_root)
    raw_root = resolve_path(cfg.data.raw_wilor_root)
    canonical_root = resolve_path(cfg.data.hand_root)
    overwrite = bool(cfg.preprocess.overwrite)

    # Construct only the implementation needed by the selected task. This
    # keeps canonical-only training preparation independent of the WiLoR stack.
    video_to_ragged = (
        instantiate(cfg.preprocess.video_to_ragged)
        if task in {"video_to_ragged", "all"}
        else None
    )
    ragged_to_canonical = (
        instantiate(cfg.preprocess.ragged_to_canonical)
        if task in {"ragged_to_canonical", "all"}
        else None
    )

    missing: list[str] = []
    converted = 0
    for ordinal, record_id in enumerate(records, 1):
        video_path = video_root / f"{record_id}.mp4"
        raw_dir = raw_root / record_id
        canonical_path = canonical_root / f"{record_id}.npz"

        if video_to_ragged is not None:
            # A completed raw cache remains reusable after its source MP4 is removed.
            if not video_path.is_file() and not raw_cache_has_markers(raw_dir):
                message = f"{record_id}: video not found at {video_path}"
                missing.append(message)
                print(f"[preprocess missing] {message}", flush=True)
                continue
            video_to_ragged.run(
                video_path=video_path,
                output_dir=raw_dir,
                overwrite=overwrite,
            )

        if ragged_to_canonical is not None:
            # Trust an existing canonical result before consulting retained raw
            # input. Otherwise only absent completion markers are skippable.
            reuse_canonical = canonical_path.exists() and not overwrite
            if not reuse_canonical and not raw_cache_has_markers(raw_dir):
                message = f"{record_id}: incomplete ragged cache at {raw_dir}"
                missing.append(message)
                print(f"[preprocess missing] {message}", flush=True)
                continue
            ragged_to_canonical.run(
                raw_record_dir=raw_dir,
                output_path=canonical_path,
                overwrite=overwrite,
            )

        converted += 1
        if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(records):
            print(
                f"preprocess {ordinal}/{len(records)}: {record_id}",
                flush=True,
            )

    print(
        f"preprocess complete: requested={len(records)} "
        f"completed_or_reused={converted} missing={len(missing)}",
        flush=True,
    )
