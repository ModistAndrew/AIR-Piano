# AirPiano operational handoff

AirPiano is a compact implementation of hand-to-MIDI chord alignment with one
baseline and a wrist-relative-coordinate ablation. It converts videos to
resumable WiLoR hand detections, converts those detections to canonical 30 Hz
two-hand poses, eagerly pairs complete songs with MIDI, and trains one fixed
direct transformer with masked binary cross entropy.
This README is the operational source of truth for an agent with no prior
conversation history. Read [`AGENTS.md`](AGENTS.md) before changing or running
the project.

## Objective and supported behavior

The behavioral reference is exactly:

```text
/storage/jinxuan/workspace/V2N/configs/oofv_1s/pianomotionnew_wilor_2p5d_alignment_pitch_only_direct_transformer_chord_onset_continuous_fixed96_centered_xy_z10_30fps.yaml
```

That experiment is named `baseline` here. AirPiano preserves its essential
30 Hz, 210-frame, fixed-width, pitch-only, continuous chord-onset behavior
without its legacy configuration matrix. The legacy experiment used 96 chord
slots; the current baseline uses 200 so the expanded canonical-data
intersection does not truncate its one 119-overlap record.

The supported stages are:

- `preprocess`: video to ragged WiLoR cache, ragged cache to canonical pose, or
  both operations.
- `train`: training, per-epoch validation, and checkpoint selection.
- `test`: test-set evaluation and per-slice inference from a configurable
  checkpoint.
- `realtime`: batch preprocessing and inference for direct synchronized
  video/MIDI pairs.
- `visualize`: prediction and target heatmaps for every direct inference NPZ.
- `sonify`: prediction and ground-truth MIDI for every direct inference NPZ.

`realtime` is an end-to-end file wrapper, not a webcam or latency-bounded
streaming service.

Deliberate non-goals include model architecture switches, alternate losses,
note-token fields, hand-presence filtering, global-song
stitching, distributed training, checkpoint resume, video downloading, and
compatibility with V2N checkpoints. The model and loss expose no YAML options.

## Repository map

| Path | Responsibility |
| --- | --- |
| `run.py` | Hydra bootstrap and six-entry stage dispatch only. |
| `configs/baseline.yaml` | Baseline runtime configuration. |
| `configs/wrist_xyz10.yaml` | Wrist-relative local-XYZ ×10 ablation inheriting the baseline. |
| `scripts/convert_legacy_splits.py` | One-time conversion of the original PianoMotion split lists to generalized record stems. |
| `scripts/migrate_pianomotion_data.py` | Resumable, verified migration of PianoMotion postprocessed data to `/storage2`. |
| `assets/splits/pianomotion10m/` | Ignored, locally generated train, validation, and test record stems used by the baseline. |
| `third_party/WiLoR/` | Full vendored snapshot of the official WiLoR source tree. |
| `assets/wilor/` | Ignored, locally copied WiLoR, detector, and MANO binary assets. |
| `airpiano/preprocessing/pipeline.py` | Config-owned offline record traversal and preprocessing orchestration. |
| `airpiano/preprocessing/video.py` | Public `VideoToWiLoR` Hydra component. |
| `airpiano/preprocessing/wilor/backend.py` | Headless model loading, baseline crop preprocessing, and batched inference. |
| `airpiano/preprocessing/wilor/cache.py` | Video probing/decoding, atomic chunks/manifests, resumption, and completion. |
| `airpiano/preprocessing/canonical.py` | Best-left/right selection and fixed-rate canonicalization. |
| `airpiano/data/` | Relative-stem pairing, MIDI/hand loading, slicing, targets, eager datasets, and DataLoader construction. |
| `airpiano/model/` | Fixed transformer layers, assembly, and masked BCE. |
| `airpiano/train.py` | Dataset construction, training, validation, and checkpoint creation. |
| `airpiano/inference.py` | Checkpoint loading, shared dataset inference, and config-owned test artifact dumping. |
| `airpiano/realtime.py` | Root-scanned batch preprocessing and inference workflow. |
| `airpiano/render/` | Artifact loading, heatmaps, and sonification. |
| `airpiano/utils.py` | Direct JSON/NPZ output and artifact-name helpers. |
| `tests/` | Focused data, preprocessing, model, training, realtime, and rendering tests. |

Each stage handler accepts the unchanged root `DictConfig`. `run.py` selects a
handler and contains no stage-specific initialization. Extend an existing stage
in its owning component; only a genuinely new top-level stage needs a new
dispatch entry.

Useful explicit imports are:

```python
from airpiano.preprocessing.pipeline import preprocess
from airpiano.preprocessing.video import VideoToWiLoR
from airpiano.preprocessing.canonical import WiLoRCoordinateConverter, WiLoRToCanonical
from airpiano.data import build_data_components, build_dataset, make_loader
from airpiano.train import train
from airpiano.inference import test
from airpiano.realtime import realtime
from airpiano.model.transformer import AirPianoTransformer
from airpiano.model.loss import masked_bce_loss
from airpiano.render.artifact import load_artifact
from airpiano.render.visualize import visualize, visualize_artifact
from airpiano.render.sonify import sonify, sonify_artifact, sonify_ground_truth
```

## Environment setup

Run on Linux from the repository root with the existing Conda environment
`v2n`. Do not silently use system Python and do not install or alter packages
without user approval.

```bash
cd /storage/jinxuan/workspace/AirPiano
conda run -n v2n python -c \
  "import torch,numpy,mido,hydra,omegaconf,wandb,matplotlib,cv2; print(torch.__version__)"
conda run -n v2n python run.py --cfg job --resolve
```

System `ffmpeg` and `ffprobe`, OpenCV, PyTorch, Ultralytics, and the dependencies
declared by WiLoR are required for video preprocessing. Confirm the system tools
without modifying the environment:

```bash
ffmpeg -version
ffprobe -version
```

### Vendored WiLoR snapshot

`third_party/WiLoR` is an ordinary, fully tracked directory copied from the
official repository:

```text
https://github.com/rolpotamias/WiLoR.git
```

Its source revision is:

```text
fcb911312a38fa8badd30d9656a167485d61b8f9
```

There is no nested Git repository, source initialization, or additional source
download. A normal AirPiano checkout already contains the complete upstream
snapshot, including its demos, images, documentation, and utility scripts.
Those files preserve upstream context and attribution but are inert from
AirPiano's perspective: they are not supported `run.py` stages or alternate
AirPiano entry points. Retain upstream copyright/license notices and attribution;
do not present the vendored code as original AirPiano work.

AirPiano imports only the WiLoR modules needed by its in-tree backend. It
temporarily adds the absolute vendored path to `sys.path`, does not change the
working directory, and lets Python manage bytecode normally. Python
`__pycache__` directories and bytecode are excluded from the vendored snapshot
and ignored by the outer AirPiano repository.

### Install required external assets

These large or separately licensed files are intentionally untracked:

```text
assets/wilor/wilor_final.ckpt
assets/wilor/detector.pt
assets/wilor/MANO_RIGHT.pkl
```

Obtain these files from their official sources and accept their respective
licenses. AirPiano requires independent regular-file copies: do not use
symlinks or reflinks. On the development workstation, the approved sibling
checkout can seed the ignored local asset directory:

```bash
mkdir -p assets/wilor
cp --reflink=never /storage/jinxuan/workspace/WiLoR/pretrained_models/wilor_final.ckpt assets/wilor/
cp --reflink=never /storage/jinxuan/workspace/WiLoR/pretrained_models/detector.pt assets/wilor/
cp --reflink=never /storage/jinxuan/workspace/WiLoR/mano_data/MANO_RIGHT.pkl assets/wilor/
```

Verify the exact files before running video preprocessing:

```bash
sha256sum assets/wilor/wilor_final.ckpt \
  assets/wilor/detector.pt \
  assets/wilor/MANO_RIGHT.pkl
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `wilor_final.ckpt` | 2,564,989,533 | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| `detector.pt` | 53,582,271 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| `MANO_RIGHT.pkl` | 3,821,356 | `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767` |

WiLoR model files are distributed under the terms stated in
`third_party/WiLoR/license.txt`. The detector/Ultralytics and MANO assets have
their own terms; MANO requires an account and acceptance of its license. Verify
that the intended use satisfies all upstream licenses before downloading or
running the assets. AirPiano does not redistribute them.

The model configuration and MANO mean parameters are small files included in
the vendored snapshot and require no separate download:

```text
third_party/WiLoR/pretrained_models/model_config.yaml
third_party/WiLoR/mano_data/mano_mean_params.npz
```

### GPU, W&B, and storage

The baseline uses `device: cuda:1`, normally physical GPU 1. If
`CUDA_VISIBLE_DEVICES=1` exposes only that GPU, it becomes logical `cuda:0`, so
also override `device=cuda:0`.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
conda run -n v2n wandb login
df -h /storage/jinxuan/data/PianoMotion10M \
  /storage2 /storage/jinxuan/workspace/AirPiano
```

Training requires online W&B and stops clearly if authentication or network
setup fails. Raw caches are intentionally retained and videos, chunks, canonical
files, checkpoints, and eager in-memory datasets are large. Check disk and host
RAM before preprocessing or training.

## Generalized record paths

A record is identified by one root-relative POSIX stem. A split line may contain
nested directories, dots, or spaces:

```text
performer_07/session.v1/take.03
ensemble A/rehearsal/finale
```

Blank lines and lines whose stripped form begins with `#` are ignored. Duplicate
stems are removed in first-seen order. The complete stripped line is the
`record_id`; it is not tokenized into dataset-specific identity fields.

Extensions are appended to the complete stem rather than replacing its final
suffix. Given `performer_07/session.v1/take.03`, offline roots resolve to:

```text
<video_root>/performer_07/session.v1/take.03.mp4
<midi_root>/performer_07/session.v1/take.03.mid
<raw_wilor_root>/performer_07/session.v1/take.03/
<hand_root>/performer_07/session.v1/take.03.npz
```

`RelativeStemPairer` reports each missing MIDI/canonical path, prints a
requested/paired/missing summary, and returns the available intersection. Hand
presence never affects pairing.

### Converted baseline splits

The complete `assets/` directory is ignored by Git. A prepared workstation has
these local files selected by `baseline.yaml`, but a fresh checkout must
generate them before preprocessing or training:

```text
assets/splits/pianomotion10m/train.txt    882 records
assets/splits/pianomotion10m/valid.txt    101 records
assets/splits/pianomotion10m/test.txt     983 records
```

To reproduce them from the original PianoMotion lists at the script's default
source root:

```bash
conda run -n v2n python scripts/convert_legacy_splits.py
```

To use different locations:

```bash
conda run -n v2n python scripts/convert_legacy_splits.py \
  --source-root=/absolute/path/to/original_splits \
  --output-root=/absolute/path/to/generalized_splits
```

The converter preserves first-seen order, removes duplicates, emits POSIX
stems, and writes newline-terminated files. It is a migration utility only;
runtime split readers accept only generalized stems.

## PianoMotion postprocessed-data migration

The baseline keeps immutable inputs on the original volume and generated data
on the larger volume:

```text
/storage/jinxuan/data/PianoMotion10M/
├── midi/
└── Video/

/storage2/jinxuan/data/PianoMotion10M/
├── .airpiano-migration-v1.json
├── airpiano_wilor_2p5d_30fps/
├── processed_pianomotion_image_xy_wrist_z_30fps/
├── processed_pianomotion_mano_30fps/
├── processed_pianomotionnew_wilor_2p5d_30fps/
└── wilor/
```

The one-time migration uses these defaults:

```text
source:  /storage/jinxuan/data/PianoMotion10M
staging: /storage2/jinxuan/data/.PianoMotion10M.airpiano-migration
target:  /storage2/jinxuan/data/PianoMotion10M
splits:  <AirPiano>/assets/splits/pianomotion10m
```

The live report on 2026-09-05 found this split coverage:

| Split | Full | MIDI | Video | Raw manifest | Complete raw | Canonical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 882 | 882 | 835 | 834 | 833 | 581 |
| Valid | 101 | 101 | 93 | 93 | 93 | 35 |
| Test | 983 | 983 | 336 | 336 | 336 | 64 |
| Total | 1,966 | 1,966 | 1,264 | 1,263 | 1,262 | 680 |

The extra video relative to the original 1,263-video inventory is
`688183660/BV1CZ42127hq`; it has no raw manifest yet. Treat a fresh `report`,
not this dated snapshot, as authoritative if producers add more files.

`report` is read-only. It independently audits the full split union, MIDI,
video, raw-cache, completed-raw, and canonical intersections, reports
percentages and byte totals, identifies incomplete records, and checks target
capacity. Use the optional missing-record output when investigating a mismatch:

```bash
conda run -n v2n python scripts/migrate_pianomotion_data.py report
conda run -n v2n python scripts/migrate_pianomotion_data.py report \
  --show-missing
```

For a different checkout or storage layout, place path overrides before the
subcommand:

```bash
conda run -n v2n python scripts/migrate_pianomotion_data.py \
  --source-root /source/PianoMotion10M \
  --destination-root /destination/PianoMotion10M \
  --split-root /path/to/splits report
```

The retained trees currently contain 35,125,061,684 logical bytes (32.71 GiB;
about 32.74 GiB of filesystem usage). `copy` requires the `/storage2` parent to
be writable and to have enough free space. It resumes into the hidden staging
directory with `rsync`, checksum-verifies every retained file, and never
removes or modifies a source file:

```bash
findmnt -no OPTIONS /storage2
```

At the time of implementation this environment reported `/storage2` as
read-only. An administrator must make the mount writable before `copy`; the
migration utility never changes mounts or permissions itself.

```bash
conda run -n v2n python scripts/migrate_pianomotion_data.py copy
```

It copies `wilor` in full, including historical top-level status logs, plus the
three `processed_*` legacy trees shown above. It intentionally does not copy the
existing `airpiano_wilor_2p5d_30fps`; those canonical files are inexpensive to
regenerate.

Within staging, each recognized WiLoR manifest is atomically converted from
`pianomotion-wilor-v1` to `airpiano-wilor-ragged-v1`. Conversion removes only
the dataset-specific `subject_id`, `video_id`, and `source_url` fields. It
preserves the remaining timestamps, video metadata, model settings, inference
settings, chunk inventory, and status, and it does not invent MANO provenance.
The incomplete record `66685747/BV1wf4y157th` remains incomplete without
`_SUCCESS`; its three resumable chunks are retained. Runtime itself remains
strict and rejects every unconverted legacy manifest.

After verification, staging contains `.airpiano-migration-v1.json`, a compact
catalog of per-tree counts, byte sizes, and SHA-256 digests. `finalize` uses it
to detect changes and recover safely if publication or source cleanup is
interrupted. Keep this file with the published dataset root.

After inspecting the report and staged copy, explicitly publish and clean up:

Before the final `copy` pass, stop every process that reads or writes the five
source trees, including WiLoR producers, canonical preprocessing, training,
and status-log writers. Keep them stopped until `finalize` finishes. The
utility verifies filesystem contents but cannot lock legacy processes that do
not participate in its protocol; a writer racing source quarantine could
otherwise create data that was not part of the verified copy, while a reader
would lose its old path during cleanup.

```bash
conda run -n v2n python scripts/migrate_pianomotion_data.py copy
conda run -n v2n python scripts/migrate_pianomotion_data.py finalize \
  --confirm-delete-source
```

`finalize` first re-verifies staging against source, atomically publishes it as
the target, and then deletes only these source directories:

```text
airpiano_wilor_2p5d_30fps
processed_pianomotion_image_xy_wrist_z_30fps
processed_pianomotion_mano_30fps
processed_pianomotionnew_wilor_2p5d_30fps
wilor
```

It creates an empty target `airpiano_wilor_2p5d_30fps` for regeneration. The
confirmation flag is mandatory because source cleanup is destructive. Unknown
manifest schemas, checksum differences, insufficient capacity, an unwritable
destination, and unexpected pre-existing target content all stop the operation
before source deletion. Source directories are first atomically quarantined to
allowlisted hidden tombstones, so an interruption partway through recursive
cleanup can be resumed without mistaking a partial tree for the verified
source.

Both mutating commands are restart-safe. Rerun `copy` after an interrupted
transfer. If interruption occurs during or after publication, rerun the shown
`finalize` command with `--confirm-delete-source`; it verifies an
already-published destination before treating a missing allowlisted source
directory as cleaned up. Do not manually merge staging with a pre-existing
target.

After finalization, regenerate all currently available canonical records without
loading WiLoR:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=ragged_to_canonical
```

The expected result is 1,262 canonical records, with the one incomplete raw
cache and 703 unavailable records skipped. Once the WiLoR assets are available,
resume the incomplete cache and canonicalize it through the normal combined
pipeline:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n v2n python run.py \
  stage=preprocess \
  preprocess.task=all \
  device=cuda:0
```

## Preprocessing contracts

### Video to ragged WiLoR cache

`VideoToWiLoR.run(video_path, output_dir, overwrite=False) -> Path` uses the
in-tree backend and cache implementation. Model objects are loaded lazily on
the first uncached video and reused by later calls on the same component.
Renderer initialization is disabled for headless execution. Ordinary video is
decoded with OpenCV; AV1 or an OpenCV-open failure uses system FFmpeg.

The output layout for one stem is:

```text
<raw_wilor_root>/<stem>/
├── manifest.json
├── _SUCCESS
└── chunks/
    ├── 000000.npz
    ├── 000001.npz
    └── ...
```

Only this exact manifest schema is accepted:

```text
airpiano-wilor-ragged-v1
```

The manifest contains `schema_version`, `status`, `video_path`, `video`,
`model`, `inference`, `vertex_reconstruction`, `chunks`, creation/update times,
and—once work advances—`decoded_frame_count` and `detection_count`. A complete
manifest also has `completed_at`. `video` records probed codec, dimensions,
source rate, duration, and file metadata. `model` records absolute upstream
paths and asset hashes. `inference` records all extraction settings. Each chunk
inventory entry contains `file`, `frame_start`, `frame_end`, and `detections`.

Each compressed chunk is source-frame-major. Let `F` be frames in the chunk and
`D` be all hand detections across them:

| Array | Shape/type and meaning |
| --- | --- |
| `frame_indices` | `int64[F]`, zero-based source indices. |
| `frame_timestamps_seconds` | `float64[F]`, source timestamps. |
| `frame_detection_offsets` | `int64[F+1]`, ragged boundaries into every detection array. |
| `boxes_xyxy` | `float32[D,4]`, detector boxes. |
| `detector_confidence` | `float32[D]`, detector scores. |
| `is_right` | `uint8[D]`, `0` left and `1` right. |
| `pred_global_orient` | `float32[D,1,3,3]`, MANO global rotations. |
| `pred_hand_pose` | `float32[D,15,3,3]`, MANO hand rotations. |
| `pred_betas` | `float32[D,10]`, MANO shape coefficients. |
| `pred_cam`, `pred_cam_t`, `pred_cam_t_full` | `float32[D,3]` camera outputs. |
| `focal_length` | `float32[D,2]`. |
| `scaled_focal_length` | `float32[D]`. |
| `pred_keypoints_3d` | `float32[D,21,3]`, handedness-corrected local joints. |
| `pred_keypoints_2d` | `float32[D,21,2]`, crop-space joints. |
| `pred_keypoints_2d_full` | `float32[D,21,2]`, full-image pixel joints. |
| `box_center`, `image_size_wh` | `float32[D,2]`. |
| `box_size` | `float32[D]`. |

Zero-detection frames remain represented by equal adjacent offsets. Vertices
are omitted; the saved MANO values are sufficient for later reconstruction.

Chunks and manifests are written to temporary files, flushed, and atomically
published. The manifest starts with `status: processing`, is updated after each
published chunk, becomes `status: complete` after decode-length checks, and only
then receives `_SUCCESS`. If interrupted, rerunning with `overwrite=false`
reuses contiguous completed chunks, advances the decoder past their source
frames, and resumes at the first missing chunk. A valid completed cache is
reused before looking for the source MP4, so the video may be archived after
successful extraction.

`overwrite=true` removes the selected raw record directory before regeneration.
Treat it as destructive and verify the exact root/stem first.

### Raw schema compatibility policy

Runtime has no raw-cache compatibility layer. Any manifest whose schema is not
`airpiano-wilor-ragged-v1`, including `pianomotion-wilor-v1`, raises a
schema-mismatch `ValueError` in both producer reuse checks and canonicalization.
Runtime never reuses or converts a legacy manifest.

The one-time `scripts/migrate_pianomotion_data.py` utility is deliberately
separate from runtime. During a verified storage migration it converts the
known `pianomotion-wilor-v1` manifests at rest to
`airpiano-wilor-ragged-v1`, without relaxing the producer or canonicalizer.
Unknown schemas are rejected. See
[PianoMotion postprocessed-data migration](#pianomotion-postprocessed-data-migration)
before operating on the legacy corpus.

Existing canonical NPZs remain valid training inputs and are trusted when
`overwrite=false`. However, a missing canonical file cannot be rebuilt from an
incompatible raw cache. Either preserve the existing canonical result or
generate current raw data from its source video in a clean root:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=all \
  data.raw_wilor_root=/absolute/path/to/new_airpiano_raw
```

Using a fresh raw root is safest when old data must be retained. To replace one
incompatible raw destination in place, use a one-record split with
`preprocess.task=video_to_ragged preprocess.overwrite=true`, then canonicalize
it separately. The corpus migration utility is the only supported bulk
conversion of the known legacy signature.

### Ragged cache to canonical pose

`WiLoRToCanonical` reads a marked, exact-schema raw cache and independently
selects the highest-confidence left and right detection in every frame. Exact
confidence ties select the first detection. Slot 0 is left and slot 1 is right.

The canonical output remains:

```text
<hand_root>/<stem>.npz
```

| Array | Exact contract |
| --- | --- |
| `schema_version` | Unicode scalar `"airpiano-wilor-canonical-v1"`. |
| `pose` | `float32[T,2,21,3]`. |
| `presence` | `bool[T,2]`. |
| `fps` | `float64` scalar, baseline `30.0`. |

The configured `WiLoRCoordinateConverter` owns spatial coordinates. The
baseline uses full-image pixels divided by image width/height for XY and local
joint depth minus wrist depth for Z. The `wrist_xyz10` ablation subtracts the
selected hand's local 3D wrist joint from all XYZ joints. Coordinate conversion
happens before temporal resampling; scaling is deliberately deferred to the
configured `HandLoader` transform.

Absent slots are exact zero. Mixed input rates are sampled at target-frame
centers using linear pose interpolation; a hand is present only if present at
both endpoints, otherwise the full target slot is zero. Output length is:

```text
floor(source_frames * target_fps / source_fps)
```

Interpolation happens here only. `HandLoader` does not resample.

When `overwrite=false`, an existing canonical path is trusted and returned
before raw input is inspected. Otherwise offline orchestration skips and reports
only a missing raw directory, `manifest.json`, or `_SUCCESS`. Once both markers
exist, schema mismatch, malformed JSON, corrupt chunks, missing fields, and
algorithm failures propagate directly.

## Dataset, model, and outputs

### MIDI and eager dataset

`MidoNoteReader` parses the merged tempo-aware MIDI clock, treats zero-velocity
note-on as note-off, pairs repeated channel/pitch notes FIFO, ignores sustain,
and keeps positive-duration pairs. `InitializerChordMerger` groups later notes
within 0.05 seconds of the initializer note; grouping is inclusive and
initializer-anchored rather than transitive.

`OffsetScaleTransform` applies configured per-axis `(pose - offset) * scale` to
present slots while preserving exact zero absent slots. The baseline uses
offset `[0.5,0.5,0]` and scale `[1,1,10]`; `wrist_xyz10` uses zero offset and
scale `[10,10,10]`. `FileLoader` eagerly loads every paired song.
`AlignmentDataset` precomputes frame starts and chord selection indices, omits
windows whose chord selection is entirely padding, then slices/pads and
generates targets in `__getitem__`. Such post-score windows have no valid cell
on which masked BCE could be defined.

One baseline item is:

| Field | Type/shape | Meaning |
| --- | --- | --- |
| `hands` | `float32[210,2,21,3]` | Transformed left/right joints. |
| `hand_presence` | `bool[210,2]` | Per-hand detection flags. |
| `frame_mask` | `bool[210]` | Real frame versus right padding only. |
| `chords` | binary `float32[C,128]` | MIDI-pitch multi-hot chord rows; baseline `C=200`. |
| `chord_mask` | `bool[C]` | Real chord versus padding. |
| `chord_ids` | `int64[C]` | Global chord IDs, `-1` for padding. |
| `target` | `float32[210,C]` | Soft chord-onset alignment. |
| `record_id` | string | Complete root-relative stem. |
| `frame_start` | integer | Global left-edge frame index. |
| `fps` | float | Canonical/source clock used by the slicer. |

The slicer emits regular 105-frame starts plus the exact right-aligned final
window, padding a short song. It keeps every chord with any member overlapping
the half-open window, appends later chords until the configured fixed width is
available, and pads at score end. More overlapping chords than that width raises
instead of truncating positives. Duplicate pitches remain distinct notes
internally but collapse to one multi-hot entry.

Targets use global frame times and chord initializer onsets:

```text
target[t,c] = max(0, 1 - abs(global_frame_time[t] - chord_onset[c]) / 0.2)
```

Padded cells are zero. Missing hands do not remove frames, targets, or loss
positions.

### Fixed model and criterion

`AirPianoTransformer` takes:

```text
hands          [B,T,2,21,3]
hand_presence  [B,T,2]
chords         [B,C,128]
frame_mask     [B,T]
chord_mask     [B,C]
```

and returns logits `[B,T,C]`. It embeds 126 hand coordinates plus two presence
bits and separately embeds 128-pitch chords into width 512. Four chord
transformer layers and four frame self/cross-attention blocks feed 32-wide
pairwise projections. The score bias starts from positive prior `0.0055`.
Absent hand coordinates are zeroed, presence remains an input feature, padded
attention is protected from NaNs, and padded logits are zero.

`masked_bce_loss` is ordinary `BCEWithLogitsLoss` over:

```text
frame_mask[:, :, None] & chord_mask[:, None, :]
```

It rejects a batch with no valid cell. Model architecture and criterion are
fixed in code. Existing AirPiano checkpoints remain compatible after the module
split; V2N checkpoints do not.

### Fixed local output tree

Every generated training, inference, logging, and rendering artifact belongs to
one fixed project directory. The baseline layout is:

```text
outputs/baseline/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── inference/
│   ├── test/
│   │   ├── *.npz
│   │   └── test_loss.json
│   └── realtime/<stem>/
│       ├── *.npz
│       └── realtime_loss.json
├── logs/
│   └── wandb/
├── realtime/
│   ├── <stem>.mp4
│   ├── <stem>.mid
│   ├── raw_wilor/<stem>/
│   └── canonical/<stem>.npz
├── visualizations/<artifact>.png
└── sonifications/
    ├── predict/<artifact>.mid
    └── ground_truth/<artifact>.mid
```

There is no version or run-identity layer. Starting training a second time in
the same project directory is unsupported: checkpoints are replaced while
older inference files and logs may remain, producing a mixed tree. AirPiano
does not detect, clear, or resume such a run. Before an intentional rerun,
inspect and manually archive or delete the complete `outputs/baseline/`
directory.

Relative checkpoint, realtime, inference, W&B, visualization, and sonification
paths are resolved beneath `outputs/baseline/`. Absolute Hydra overrides stay
absolute. Dataset and preprocessing roots retain their independent path rules.

### Checkpoints and inference artifacts

Training writes `checkpoints/last.pt` after every validation epoch and updates
`checkpoints/best.pt` on a strictly lower validation loss. Each payload contains
`format`, `model`, `optimizer`, `scheduler`, `epoch`, `global_step`,
`best_valid_loss`, and the resolved `config`. Training always starts fresh; it
does not resume optimizer state.

The separate test stage defaults to `checkpoints/last.pt`, as does realtime;
`best.pt` or any compatible checkpoint remains available through an explicit
override. Test inference writes one independent compressed NPZ per slice plus
`test_loss.json`, and realtime writes the same slice format plus
`realtime_loss.json`. There is no inference manifest or cross-slice stitching.

Inspect the epoch and optimizer-update step stored in either checkpoint without
loading it onto a GPU:

```bash
conda activate v2n
python - <<'PY'
from pathlib import Path
import torch

root = Path("outputs/baseline/checkpoints")
for name in ("best.pt", "last.pt"):
    checkpoint = torch.load(root / name, map_location="cpu", weights_only=False)
    print(
        f"{name}: epoch={checkpoint['epoch']} "
        f"global_step={checkpoint['global_step']} "
        f"best_valid_loss={checkpoint['best_valid_loss']}"
    )
PY
```

`epoch` is one-based and `global_step` counts completed optimizer updates after
gradient accumulation, not individual DataLoader batches.

Each slice NPZ contains:

| Array | Contract |
| --- | --- |
| `schema_version` | Unicode scalar `"airpiano-inference-v1"`. |
| `probability`, `target` | `float32[T,C]`. |
| `frame_mask` | `bool[T]`. |
| `chord_mask` | `bool[C]`. |
| `chord_pitches` | `bool[C,128]`. |
| `chord_ids` | `int64[C]`. |
| `record_id` | Unicode scalar root-relative stem. |
| `frame_start` | `int64` scalar. |
| `fps` | `float64` scalar. |
| `loss` | `float64` scalar per-item BCE. |

Artifact filenames replace `/` and `\\` in the record ID with `__`, then append
`__f<frame_start as nine digits>.npz`. Canonical files and inference artifacts
are unchanged by the generalized path migration.

## Stage workflows and commands

All commands run from the AirPiano root. Hydra is configured not to change the
working directory.

### Inspect and convert splits

```bash
conda run -n v2n python run.py --cfg job --resolve
conda run -n v2n python scripts/convert_legacy_splits.py
```

### Preprocess

Training reads only canonical input; it never triggers preprocessing.

Convert current-schema ragged caches to canonical pose:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=ragged_to_canonical
```

Extract ragged pose from videos, or run both phases:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=video_to_ragged

conda run -n v2n python run.py stage=preprocess \
  preprocess.task=all
```

If the configured raw root contains an incompatible cache schema, choose a
clean destination rather than expecting compatibility:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=all \
  data.raw_wilor_root=/storage2/jinxuan/data/PianoMotion10M/airpiano_wilor_raw
```

For a small smoke list:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=all \
  'preprocess.split_files=[/absolute/path/to/smoke.txt]'
```

To preprocess, train, and then infer the test set sequentially in one
invocation, pass a Hydra list:

```bash
conda run -n v2n python run.py \
  'stage=[preprocess,train,test]' \
  preprocess.task=ragged_to_canonical
```

Stages run once each in the listed order and receive the same resolved root
configuration. Execution is fail-fast: if one stage raises, subsequent stages
are not run.

### Wrist-relative XYZ ×10 ablation

The ablation reuses the baseline raw WiLoR caches but writes unscaled
wrist-relative local XYZ to its own canonical root. Generate that cache before
training:

```bash
conda run -n v2n python run.py --config-name wrist_xyz10 \
  stage=preprocess preprocess.task=ragged_to_canonical

conda run -n v2n python run.py --config-name wrist_xyz10 stage=train
conda run -n v2n python run.py --config-name wrist_xyz10 stage=test
```

Its canonical files live under
`${data.processed_root}/airpiano_wilor_wrist_xyz_30fps`, while checkpoints,
logs, inference, and rendering use `outputs/wrist_xyz10/`. The ×10 factor is an
online hand transform, so later scale ablations can reuse the same canonical
cache.

### Train and validate

```bash
conda run -n v2n python run.py stage=train
```

The fixed loop uses AdamW, 20 epochs, batch size 2, accumulation 8, learning
rate `3e-4`, weight decay `0.01`, clipping `1.0`, 5% linear warmup followed by
cosine decay, BF16 CUDA autocast, and seed 42. W&B logs only `train/loss` and
`valid/loss`. The online run uses project `air_piano`, name `baseline`, and
stores local W&B files under `outputs/baseline/logs/wandb/`. Training writes
checkpoints but does not load the test split or dump inference artifacts.

Do not use a second invocation as resume or overwrite behavior. If the fixed
tree already belongs to a prior run, inspect it and move it to an external
archive or explicitly remove it before starting again.

### Test-set inference

Run test evaluation and dump every slice using the default `last.pt`:

```bash
conda run -n v2n python run.py stage=test
```

Choose another compatible checkpoint and, optionally, another artifact
directory without retraining:

```bash
conda run -n v2n python run.py stage=test \
  test.checkpoint=checkpoints/best.pt \
  test.output_dir=inference/test-best
```

Both paths are relative to `outputs/baseline/` unless given as absolute paths.
The stage writes `test_loss.json` beside its per-slice NPZ files and does not
open a W&B run.

### Realtime batch

Place synchronized pairs directly inside the configured root. Matching is by
the complete basename, so dotted names are preserved:

```text
outputs/baseline/realtime/
├── performance.v2.mp4
├── performance.v2.mid
├── raw_wilor/performance.v2/
└── canonical/performance.v2.npz
```

Run every direct pair with the default `last.pt` checkpoint:

```bash
conda run -n v2n python run.py stage=realtime
```

Outputs for this example are written below
`outputs/baseline/inference/realtime/performance.v2/`. Direct video or MIDI
files without a matching counterpart are reported and skipped. Nested inputs
are not scanned. WiLoR, canonical, and dataset components are constructed once
and reused across the sorted pairs.

Select a different root, output directory, or checkpoint as needed:

```bash
conda run -n v2n python run.py stage=realtime \
  realtime.root=/data/performance \
  realtime.output_dir=inference/realtime-best \
  realtime.checkpoint=checkpoints/best.pt
```

Each basename becomes `record_id`. Realtime calls the same configured
`VideoToWiLoR`, `WiLoRToCanonical`, `HandLoader`, slicer, target generator, and
checkpoint inference as the offline workflow. Every video/MIDI pair must
describe the same performance and timeline.

### Visualize and sonify batches

Both stages scan direct `*.npz` files under `inference/test` by default:

```bash
conda run -n v2n python run.py stage=visualize
conda run -n v2n python run.py stage=sonify sonify.threshold=0.2
```

For an input named `test.npz`, the derived outputs are:

```text
outputs/baseline/visualizations/test.png
outputs/baseline/sonifications/predict/test.mid
outputs/baseline/sonifications/ground_truth/test.mid
```

Only direct NPZ files are processed, in sorted order. Dotted basenames are
preserved and nested directories are ignored. Relative roots and output
directories are project-relative; absolute overrides are accepted:

```bash
conda run -n v2n python run.py stage=visualize \
  visualize.root=inference/test-best \
  visualize.output_dir=visualizations/test-best

conda run -n v2n python run.py stage=sonify \
  sonify.root=/absolute/path/to/artifacts \
  sonify.output_dir=/absolute/path/to/sonifications
```

Visualization draws probability and target heatmaps with shared `[0,1]`
scaling. Sonification selects each valid chord's earliest maximum frame and
emits all active pitches when that maximum is at least the threshold. Notes are
one second at velocity 64 by default. It also resolves
`<sonify.midi_root>/<record_id>.mid` and writes the source notes beginning in
the real, unpadded slice. Ground-truth pitch, velocity, and duration are
preserved, offsets are clipped at the slice end, and timing is rebased to zero.
Notes already active before the slice are omitted. Both outputs use one track
at 120 BPM; pedals, programs, channels, source tempo events, and other MIDI
metadata are not preserved. Timing is slice-relative and overlapping slices
are not merged.

For realtime artifacts, point the artifact and MIDI roots at their batch
directories:

```bash
conda run -n v2n python run.py stage=sonify \
  sonify.root=inference/realtime/performance.v2 \
  sonify.midi_root=outputs/baseline/realtime
```

Inspect artifacts directly without pickle:

```bash
conda run -n v2n python -c \
  "import numpy as np; p=np.load('/absolute/path/to/slice.npz',allow_pickle=False); print({k:(p[k].dtype,p[k].shape) for k in p.files})"
```

## Configuration guide

All runtime settings are in `configs/baseline.yaml` and can be overridden with
Hydra `key=value` syntax.

### Runtime and data roots

| Field | Baseline | Meaning |
| --- | --- | --- |
| `stage` | `train` | One of the six supported handlers, or an ordered Hydra list. |
| `device` | `cuda:1` | Training and WiLoR device. |
| `seed` | `42` | Python, NumPy, Torch, CUDA, and shuffle seed. |
| `output.root` | `outputs` | Parent of generated project artifacts. |
| `output.project_name` | `baseline` | Single local project directory name. |
| `output.dir` | `${output.root}/${output.project_name}` | Fixed base for relative stage-output paths. |
| `data.root` | `/storage/jinxuan/data/PianoMotion10M` | Parent of immutable source modalities. |
| `data.processed_root` | `/storage2/jinxuan/data/PianoMotion10M` | Parent of generated and migrated modalities. |
| `data.midi_root` | `${data.root}/midi` | `<stem>.mid` files. |
| `data.video_root` | `${data.root}/Video` | `<stem>.mp4` source videos. |
| `data.raw_wilor_root` | `${data.processed_root}/wilor` | `<stem>/` current-schema raw caches. |
| `data.hand_root` | `${data.processed_root}/airpiano_wilor_2p5d_30fps` | `<stem>.npz` canonical files. |
| `data.splits.*` | `assets/splits/pianomotion10m/*.txt` | Ignored local generalized root-relative stems. |

### WiLoR and canonicalization

| Field | Baseline | Meaning |
| --- | --- | --- |
| `preprocess.task` | `ragged_to_canonical` | `video_to_ragged`, `ragged_to_canonical`, or `all`. |
| `preprocess.split_files` | all three configured splits | First-seen ordered union to process. |
| `preprocess.overwrite` | `false` | Reuse compatible raw/canonical products; `true` regenerates selected work. |
| `video_to_ragged._target_` | `airpiano.preprocessing.video.VideoToWiLoR` | Public extraction component. |
| `wilor_repo` | `third_party/WiLoR` | Full vendored official source snapshot. |
| `checkpoint` | `assets/wilor/wilor_final.ckpt` | WiLoR checkpoint. |
| `model_config` | vendored `model_config.yaml` | Upstream architecture configuration. |
| `detector_checkpoint` | `assets/wilor/detector.pt` | Ultralytics hand detector. |
| `mano_model` | `assets/wilor/MANO_RIGHT.pkl` | Licensed MANO right-hand model. |
| `mano_mean_params` | vendored mean-parameter NPZ | Upstream mean pose data. |
| `detector_confidence` | `0.3` | Detector threshold. |
| `rescale_factor` | `2.0` | Hand crop expansion. |
| `frame_batch_size` | `8` | Frames sent to detection per step. |
| `hand_batch_size` | `16` | Crops sent to WiLoR per step. |
| `chunk_frames` | `512` | Source frames per resumable chunk. |
| `fast` | `false` | Enable upstream half/skip-block fast path. |
| `compile` | `false` | Compile the backbone when fast mode is used. |
| `ragged_to_canonical._target_` | `airpiano.preprocessing.canonical.WiLoRToCanonical` | Current raw schema to canonical NPZ. |
| `coordinates._target_` | `airpiano.preprocessing.canonical.WiLoRCoordinateConverter` | Convert each selected raw detection into one canonical hand. |
| `coordinates.xy_source` | `normalized_2d` | Use normalized full-image XY; `wrist_relative_3d` selects local XYZ relative to the wrist. |
| `coordinates.z_source` | `wrist_relative_3d` | Local WiLoR depth relative to the selected hand's wrist. |
| `target_fps` | `30.0` | Canonical output rate. |

Relative code/asset paths rely on Hydra's `job.chdir: false`; launch from the
repository root or override them with absolute paths.

### Dataset components

| Field | Baseline behavior |
| --- | --- |
| `file_loader` / `RelativeStemPairer` | Pair `<stem>.mid` and `<stem>.npz`, report the intersection, then read full songs. |
| `MidoNoteReader` | Tempo-aware physical note parsing. |
| `InitializerChordMerger.tolerance_seconds` | Initializer-relative grouping at `0.05`. |
| `CanonicalNpzReader` | Direct trusted canonical array loading. |
| `OffsetScaleTransform` | Apply configured per-axis offset and scale, then restore absent slots to zero. |
| `FutureChordSlicer` | 30 Hz, 210 frames, stride 105, fixed 200 future-extending candidates. |
| `ChordOnsetEvents` | One event at each chord initializer onset. |
| `TriangularSoftener.radius_seconds` | Continuous target radius `0.2`. |

Each `_target_` is a replaceable seam, but its parent-facing contract must stay
stable. The model and loss intentionally have no `_target_`.

### Training, realtime, and rendering

| Field | Baseline | Meaning |
| --- | --- | --- |
| `epochs` | `20` | Validation/checkpoint cycles. |
| `batch_size` | `2` | Train/evaluation/inference loader batch size. |
| `gradient_accumulation_steps` | `8` | Batches per optimizer update. |
| `learning_rate` | `3e-4` | AdamW initial rate. |
| `weight_decay` | `0.01` | AdamW decay. |
| `max_grad_norm` | `1.0` | Update-time global norm clipping. |
| `warmup_fraction` | `0.05` | Linear warmup fraction before cosine decay. |
| `bf16` | `true` | CUDA BF16 autocast. |
| `wandb.project` | `air_piano` | Required online project. |
| `wandb.name` | `baseline` | Run name. |
| `wandb.entity` | `null` | Optional team/entity. |
| `wandb.dir` | `logs` | Project-relative parent; W&B creates its `wandb/` directory here. |
| `test.checkpoint` | `checkpoints/last.pt` | Project-relative checkpoint loaded by `stage=test`; absolute paths also work. |
| `test.output_dir` | `inference/test` | Project-relative test artifact and loss directory. |
| `realtime.root` | `realtime` | Project-relative root scanned for direct matching `.mp4`/`.mid` files. |
| `realtime.checkpoint` | `checkpoints/last.pt` | Project-relative default AirPiano checkpoint; absolute paths also work. |
| `realtime.output_dir` | `inference/realtime` | Project-relative parent for per-stem predictions. |
| `realtime.overwrite` | `false` | Raw and canonical preprocessing reuse policy. |
| `visualize.root` | `inference/test` | Project-relative root scanned non-recursively for NPZ artifacts. |
| `visualize.output_dir` | `visualizations` | Project-relative directory for basename-matched PNG files. |
| `sonify.root` | `inference/test` | Project-relative root scanned non-recursively for NPZ artifacts. |
| `sonify.output_dir` | `sonifications` | Parent of basename-matched `predict/` and `ground_truth/` MIDI files. |
| `sonify.midi_root` | `${data.midi_root}` | Root used with the artifact `record_id` to locate source MIDI. |
| `sonify.threshold` | `0.2` | Inclusive chord peak threshold. |
| `sonify.duration_seconds` | `1.0` | Fixed output-note duration. |
| `sonify.velocity` | `64` | Fixed note-on velocity. |

The checkpoint directory remains fixed in the training handler. Test checkpoint
and artifact paths are owned by the test handler and exposed independently.
Relative paths in this table resolve beneath `output.dir`; an absolute override
bypasses that prefix.

## Extension contracts

The narrow replaceable interfaces are:

| Component | Required interface |
| --- | --- |
| `VideoToWiLoR` | `run(video_path, output_dir, overwrite=False) -> Path`; emit the configured canonicalizer's raw contract. |
| `WiLoRToCanonical` | `run(raw_record_dir, output_path, overwrite=False) -> Path`; emit canonical pose NPZ. |
| `RelativeStemPairer` | `pair(split_file, midi_root, hand_root) -> list[FilePair]`. |
| `EagerSongReader` | `read(pair, midi_loader, hand_loader) -> Song`; the loader exposes it as `.reader` for realtime. |
| `MidoNoteReader` | `read(path) -> tuple[Note,...]`. |
| `InitializerChordMerger` | `merge(notes) -> tuple[Chord,...]`. |
| `CanonicalNpzReader` | `read(path) -> HandSequence`. |
| `WiLoRCoordinateConverter` | `(raw_arrays, detection_index) -> float32[21,3]` canonical hand. |
| Hand transform | `(pose, presence) -> pose` with unchanged shape and absent slots zero. |
| Slicer | Expose fixed dimensions and implement starts, candidate selection, and slicing. |
| Event generator | `times(chords) -> float64[event_count]`. |
| Softener | `soften(frame_times, event_times) -> [T,event_count]`. |

Common changes stay localized:

- New file layout: replace the pairer target and update offline/realtime path
  ownership together if the record-path contract changes.
- New MIDI parser or grouping rule: replace the reader or chord merger.
- New pose format or normalization: replace the canonical reader or hand
  transform; preserve `[T,2,21,3]`, left/right order, presence, and zero absence.
- New WiLoR coordinate representation: change the converter's XY/Z sources or
  replace its `_target_`; use a distinct canonical root when the stored
  coordinate meaning changes.
- Earlier rather than later chord extension: replace the slicer while retaining
  all overlapping chords and the fixed-width overflow safeguard.
- Note alignment: use singleton chords plus a compatible slicer/event generator;
  the note sequence already exists and the model accepts pitch vectors.
- New target profile: replace the event generator and/or softener, retaining
  finite `[T,C]` BCE targets.
- New raw backend: replace `VideoToWiLoR` and canonicalizer targets together.
  Realtime automatically instantiates the same configured components.
- Model/loss changes: edit `airpiano/model/` deliberately and version checkpoint
  compatibility; they are not configuration options.

Any new training target or inference field must be considered in visualization,
sonification, artifact tests, and documentation in the same change.

## Critical invariants and common traps

1. `frame_mask` means real versus padded time only. Presence is not a frame
   mask and never filters records, frames, targets, or BCE positions. Dataset
   windows are omitted only when they contain no real chord candidate.
2. Hand slot order is `[Left, Right]`; chord vectors always span MIDI pitches
   `0..127`.
3. Canonical FPS and slicer FPS are both 30. Interpolation occurs only in
   ragged-to-canonical preprocessing.
4. Targets use global frame time, even though artifacts contain individual
   slices.
5. Fixed-width selection extends toward later chords and raises rather than
   truncating when overlap exceeds the configured width.
6. Split entries are complete root-relative stems. Do not split on whitespace,
   infer identity segments, or replace a dotted basename's suffix.
7. Only `airpiano-wilor-ragged-v1` raw caches are accepted. Marker presence does
   not grant compatibility.
8. Current-schema raw chunks are resumable and retained. Do not delete them
   after canonicalization without an explicit storage decision.
9. `overwrite=true` can delete and regenerate the selected raw record. Prefer a
   clean raw root when preserving incompatible data matters; use the migration
   utility only for the recognized legacy PianoMotion corpus.
10. Training requires canonical NPZs; it never starts either preprocessing
    phase.
11. Realtime requires direct synchronized `.mp4`/`.mid` pairs under one root.
12. Three complete split datasets live in host RAM simultaneously. Lowering
    batch size does not reduce eager dataset memory.
13. Training always starts online W&B project `air_piano` and has no
    disabled/offline mode.
14. Test and realtime inference default to `last.pt`; selecting `best.pt` is an
    explicit override.
15. The fixed project directory is not cleared or validated before training.
    Duplicate runs are unsupported and can mix overwritten and stale files.
16. Test/realtime predictions have no manifest. Visualization and sonification
    batch direct slices without stitching overlap.
17. Raw chunk/manifest writes are atomic; canonical, checkpoint, inference,
    JSON, and rendered outputs are direct writes and may be partial after an
    interruption.

## Validation and troubleshooting

Run the complete discovered suite and static import checks after any change:

```bash
cd /storage/jinxuan/workspace/AirPiano
conda run -n v2n python -m unittest discover -v
conda run -n v2n python -m compileall -q airpiano scripts tests run.py
conda run -n v2n python run.py --cfg job --resolve
test -f third_party/WiLoR/license.txt
test ! -e third_party/WiLoR/.git
```

Do not hard-code a test count; use the exit status because the suite evolves.
Do not run bulk preprocessing or full training merely to validate a code change.

Useful inspections:

```bash
# Count generalized split records.
wc -l assets/splits/pianomotion10m/*.txt

# Verify the ignored local WiLoR binaries.
sha256sum assets/wilor/wilor_final.ckpt assets/wilor/detector.pt \
  assets/wilor/MANO_RIGHT.pkl

# Count completed current-root records and canonical files.
find /absolute/path/to/raw_wilor -name _SUCCESS -type f | wc -l
find /absolute/path/to/canonical -name '*.npz' -type f | wc -l

# Inspect raw schema and status.
conda run -n v2n python -c \
  "import json; from pathlib import Path; p=Path('/absolute/path/to/raw/manifest.json'); m=json.loads(p.read_text()); print(m['schema_version'],m['status'],m.get('decoded_frame_count'))"

# Inspect a canonical or inference archive.
conda run -n v2n python -c \
  "import numpy as np; p=np.load('/absolute/path/to/file.npz',allow_pickle=False); print({k:(p[k].dtype,p[k].shape) for k in p.files})"
```

| Symptom | Cause and action |
| --- | --- |
| Vendored WiLoR files are absent | Restore `third_party/WiLoR` from the AirPiano checkout. Do not create a nested clone or external Git link. |
| `wilor` imports fail while the vendored files exist | Check the `v2n` environment against upstream requirements; source initialization is not required. |
| Missing split file | Generate the three ignored `assets/splits/pianomotion10m` files with `scripts/convert_legacy_splits.py`. |
| Missing checkpoint, detector, or MANO file | Copy the three ignored `assets/wilor` binaries as independent regular files after accepting their licenses, then verify their hashes. |
| `ffprobe`/`ffmpeg` not found or video decode fails | Verify both system executables. OpenCV is primary except AV1; FFmpeg is the fallback. |
| Raw schema mismatch mentioning `pianomotion-wilor-v1` | Runtime intentionally rejects that cache. For the legacy PianoMotion corpus, run the documented migration; otherwise preserve canonical output or regenerate raw from video in a clean root. |
| `[preprocess missing] ... incomplete ragged cache` | The raw directory, manifest, or success marker is absent. Resume video extraction or verify roots/stems. |
| JSON, NumPy, key, or indexing error after both markers exist | A purportedly complete cache is corrupt or malformed. Native failure is intentional; inspect/regenerate it. |
| Pairing reports `paired=0` | Split stems, modality roots, or canonical files do not line up. Remember extensions are appended to the full stem. |
| Canonical alignment is time-scaled | Keep canonical target FPS and slicer FPS at 30; loaders do not audit equality. |
| `overlapping chords` exceeds the configured width | Inspect the MIDI/grouping or increase the fixed width. Positive overlaps are intentionally never truncated. |
| No valid target cell | The input bypassed normal empty-window filtering or contains no usable chord candidates; inspect MIDI and masks. |
| CUDA device error | Inspect logical device numbering and coordinate `device` with `CUDA_VISIBLE_DEVICES`. |
| WiLoR CUDA OOM | Lower frame and hand batch sizes; retain the fixed crop/model semantics. |
| Training CUDA OOM | Lower training batch size while preserving architecture and accumulation intent. |
| Host RAM exhaustion | Diagnose with smaller split files or redesign eager loading; batch size is not the main cause. |
| W&B authentication/network failure | Run `wandb login` and confirm online connectivity. |
| Mixed or stale files beneath `outputs/baseline` | A duplicate run used the unsupported fixed destination. Inspect it, then manually archive or delete the complete project directory before rerunning. |
| Disk exhaustion or interrupted preprocessing | Free space without deleting wanted caches, then rerun with `overwrite=false` to resume completed chunks. |
| Checkpoint state mismatch | Use an AirPiano checkpoint produced by this model, not a V2N or modified-model checkpoint. |
| Renderer key/shape error | Inspect the NPZ and use an artifact produced by this checkout; rendering trusts the artifact contract. |

## For the next Codex agent

1. Read `AGENTS.md`, this README, and `configs/baseline.yaml` before acting.
2. Inspect worktree status and preserve unrelated user or agent changes.
3. Confirm the vendored WiLoR source and license are present, contain no nested
   `.git`, and still represent the documented official revision.
4. Treat the external V2N project only as a read-only parity oracle.
5. Use `v2n`, GPU 1 by default, and inspect logical CUDA numbering.
6. Verify licensed assets, video/MIDI roots, generalized split stems, and free
   disk/RAM before starting preprocessing or training.
7. Runtime must reject unsupported raw schemas. Use the explicit PianoMotion
   migration utility only for its recognized legacy corpus; otherwise choose a
   fresh raw root rather than adding silent compatibility.
8. Preserve record-stem, raw chunk, canonical tensor, model input, checkpoint,
   and inference artifact contracts unless explicitly authorized to version
   them.
9. Keep low-level Hydra components narrow and let each stage handler own its
   root configuration and orchestration.
10. Run focused tests during development, then full unit discovery,
    `compileall`, resolved Hydra config, and vendored-source checks before
    claiming success. Report exactly what was and was not run.
