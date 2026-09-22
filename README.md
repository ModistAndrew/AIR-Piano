# AirPiano operational handoff

AirPiano is a compact implementation of hand-to-MIDI chord alignment with one
baseline plus wrist-relative-coordinate, duration-conditioned, and causal
chord-window ablations. It converts videos to
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
30 Hz, 210-frame, bounded-width, pitch-only, continuous chord-onset behavior
without its legacy configuration matrix. The legacy experiment used 96 chord
slots; the current baseline uses 128 so the expanded canonical-data
intersection does not truncate its one 119-overlap record.

The supported stages are:

- `preprocess`: video to ragged WiLoR cache, ragged cache to canonical pose, or
  both operations.
- `train`: training, per-epoch validation, and checkpoint selection.
- `test`: test-set evaluation and per-slice inference from a configurable
  checkpoint.
- `realtime`: batch preprocessing and inference for direct synchronized
  video/MIDI pairs.
- `full-song`: realtime-style preprocessing followed by frame-causal,
  dynamically windowed inference into one artifact per song.
- `visualize`: prediction, target, and residual heatmaps for every direct
  inference NPZ.
- `sonify`: prediction and ground-truth MIDI per NPZ plus final onset summaries.

`realtime` and `full-song` are end-to-end file wrappers, not webcam or
latency-bounded streaming services. Full-song inference simulates online frame
visibility after preprocessing the complete recording.

Deliberate non-goals include pluggable model architectures, note-level tokens,
distributed training, checkpoint resume, video downloading, and
compatibility with V2N checkpoints. The model exposes causal frame-attention,
chord-duration, chord-offset, and zero-hand-input switches and is constructed
through a Hydra `_target_`.
The alignment loss is selected by `alignment_loss`; BCE remains the default.

## Repository map

| Path | Responsibility |
| --- | --- |
| `run.py` | Hydra bootstrap and seven-entry stage dispatch only. |
| `configs/baseline.yaml` | Baseline runtime configuration. |
| `configs/lowest_v.yaml` | Lowest-wrist-V selection ablation inheriting the baseline. |
| `configs/smoothed.yaml` | Lowest-wrist selection, presence filtering, short-gap filling, and pose smoothing preset. |
| `configs/pianovam.yaml` | PianoVAM-only 30 Hz configuration inheriting the baseline. |
| `configs/full.yaml` | Combined PianoMotion/PianoVAM training and evaluation configuration. |
| `configs/causal_chord24_extended.yaml` | Causal 24-chord ablation using the extended combined training split. |
| `configs/causal_chord24_extended_chord_duration_mean_normalized.yaml` | Duration-conditioned causal 24-chord extended-data ablation with deterministic song-mean duration normalization. |
| `configs/wrist_xyz10_causal_chord24_extended.yaml` | Wrist-relative causal 24-chord ablation using the extended combined training split. |
| `configs/wrist_xyz10_causal_chord24_extended_quantized.yaml` | Extended wrist-relative causal ablation with beat-grid chord-duration tokens. |
| `configs/wrist_xyz10_causal_chord24_extended_last_chord_softmax.yaml` | Extended wrist-relative causal ablation with persistent last-chord targets and categorical softmax. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta.yaml` | Extended wrist-relative causal ablation with camera-space wrist deltas in joint zero. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20.yaml` | Wrist-delta ablation with score-bounded outer windows and a 20% fully-missing-hand frame limit. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_chord_duration_mean_normalized.yaml` | Zero-hand-filter20 wrist-delta ablation with song-mean-normalized chord durations. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized.yaml` | Zero-hand-filter20 wrist-delta ablation with 12-subdivision beat-grid chord durations. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_last_chord_softmax.yaml` | Quantized zero-hand-filter20 wrist-delta ablation with persistent last-chord targets and categorical softmax. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_mean_time.yaml` | Quantized zero-hand-filter20 wrist-delta ablation using mean grouped-event times. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_offset.yaml` | Quantized zero-hand-filter20 wrist-delta ablation that adds note-offset actions. |
| `configs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_offset_last_chord_softmax.yaml` | Quantized onset/offset ablation with persistent last-action targets and categorical softmax. |
| `configs/wrist_xyz10.yaml` | Wrist-relative local-XYZ ×10 ablation inheriting the baseline. |
| `configs/chord_duration.yaml` | Duration-conditioned chord-token ablation inheriting the baseline. |
| `configs/causal_chord24.yaml` | Causal frame-attention and 24-chord slicing ablation inheriting the baseline. |
| `configs/wrist_xyz10_chord24_random_hand_shift2s.yaml` | Non-causal wrist-relative 24-chord ablation with independently jittered training hand-window edges. |
| `configs/wrist_xyz10_causal_chord24_zero_hand.yaml` | Wrist-relative causal 24-chord ablation with all hand features zeroed. |
| `configs/wrist_xyz10_causal_chord24_target_delay1s.yaml` | Wrist-relative causal 24-chord ablation with targets delayed by one second. |
| `configs/wrist_xyz10_causal_chord24_stride1.yaml` | Wrist-relative causal 24-chord ablation with a one-chord window stride. |
| `configs/wrist_xyz10_causal_chord24_hand_context1.yaml` | Wrist-relative causal 24-chord ablation with neighboring-onset hand context. |
| `configs/wrist_xyz10_causal_chord24_chord_duration_relative.yaml` | Wrist-relative, causal 24-chord model with song-scaled duration tokens. |
| `scripts/convert_legacy_splits.py` | One-time conversion of the original PianoMotion split lists to generalized record stems. |
| `scripts/convert_pianovam_splits.py` | Generate AirPiano splits from PianoVAM's canonical v1.1 metadata. |
| `assets/splits/pianomotion10m/` | Ignored, locally generated train, validation, and test record stems used by the baseline. |
| `assets/splits/pianovam/` | Ignored, locally generated PianoVAM train, validation, and test record stems. |
| `assets/splits/full/` | Combined train split plus PianoMotion validation and test stems. |
| `assets/splits/extended/` | Expanded combined train split with the same validation and test stems. |
| `third_party/WiLoR/` | Full vendored snapshot of the official WiLoR source tree. |
| `assets/wilor/` | Ignored, locally copied WiLoR, detector, and MANO binary assets. |
| `airpiano/preprocessing/pipeline.py` | Config-owned offline record traversal and preprocessing orchestration. |
| `airpiano/preprocessing/video.py` | Public `VideoToWiLoR` Hydra component. |
| `airpiano/preprocessing/wilor/backend.py` | Headless model loading, baseline crop preprocessing, and batched inference. |
| `airpiano/preprocessing/wilor/cache.py` | Video probing/decoding, atomic chunks/manifests, resumption, and completion. |
| `airpiano/preprocessing/wilor/visualize.py` | Standalone native-video overlay for ragged detections and absolute depth. |
| `airpiano/preprocessing/canonical.py` | Best-left/right selection and fixed-rate canonicalization. |
| `airpiano/data/` | Relative-stem pairing, MIDI/hand loading, slicing, targets, eager datasets, and DataLoader construction. |
| `airpiano/model/` | Fixed transformer layers, assembly, and masked BCE. |
| `airpiano/train.py` | Dataset construction, training, validation, and checkpoint creation. |
| `airpiano/inference.py` | Checkpoint loading, shared dataset inference, and config-owned test artifact dumping. |
| `airpiano/realtime.py` | Root-scanned batch preprocessing and inference workflow. |
| `airpiano/full_song.py` | Frame-causal full-song controller and artifact workflow. |
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
from airpiano.preprocessing.canonical import PianoMotion10MPostprocessor, WiLoRCoordinateConverter, WiLoRToCanonical
from airpiano.preprocessing.wilor.visualize import render_ragged_video
from airpiano.data import build_data_components, build_dataset, make_loader
from airpiano.train import train
from airpiano.inference import test
from airpiano.realtime import realtime
from airpiano.full_song import full_song, infer_full_song
from airpiano.model.transformer import AirPianoTransformer
from airpiano.model.loss import masked_bce_loss
from airpiano.render.artifact import load_artifact
from airpiano.render.visualize import visualize, visualize_artifact
from airpiano.render.sonify import (
    calculate_onset_deltas,
    plot_onset_delta_summary,
    sonify,
    sonify_artifact,
    sonify_ground_truth,
    write_onset_delta_summary,
)
```

## Environment setup

Run on Linux from the repository root with the existing Conda environment
`v2n`. Do not silently use system Python and do not install or alter packages
without user approval.

```bash
cd /storage/jinxuan/workspace/AirPiano
conda run -n v2n python -c \
  "import torch,numpy,scipy,mido,hydra,omegaconf,wandb,matplotlib,cv2; print(torch.__version__)"
conda run -n v2n python run.py --cfg job --resolve
```

System `ffmpeg` and `ffprobe`, OpenCV, PyTorch, SciPy, Ultralytics, and the
dependencies declared by WiLoR are required for video preprocessing and
canonical post-processing. Confirm the system tools without modifying the
environment:

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
df -h /storage/jinxuan/data/AirPianoFull \
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
PianoMotion10M/performer_07/session.v1/take.03
PianoVAM_v1/2024-09-02_12-45-12
```

Blank lines and lines whose stripped form begins with `#` are ignored. Duplicate
stems are removed in first-seen order. Every record begins with the owning
dataset directory, `PianoMotion10M/` or `PianoVAM_v1/`. The complete stripped
line is the `record_id`; it is not tokenized into separate identity fields.

Extensions are appended to the complete stem rather than replacing its final
suffix. Given `PianoMotion10M/performer_07/session.v1/take.03`, offline roots
resolve to:

```text
<video_root>/PianoMotion10M/performer_07/session.v1/take.03.mp4
<midi_root>/PianoMotion10M/performer_07/session.v1/take.03.mid
<raw_wilor_root>/PianoMotion10M/performer_07/session.v1/take.03/
<hand_root>/PianoMotion10M/performer_07/session.v1/take.03.npz
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
assets/splits/pianomotion10m/test.txt      13 records
assets/splits/pianomotion10m/test_full.txt 983 records
```

To reproduce `train.txt`, `valid.txt`, and the complete source `test_full.txt`
from the original PianoMotion lists at the script's default source root:

```bash
conda run -n v2n python scripts/convert_legacy_splits.py
```

To use different locations:

```bash
conda run -n v2n python scripts/convert_legacy_splits.py \
  --source-root=/absolute/path/to/original_splits \
  --output-root=/absolute/path/to/generalized_splits
```

The converter preserves first-seen order, removes duplicates, prefixes every
stem with `PianoMotion10M/`, and writes newline-terminated files. It deliberately
does not overwrite the 13-record held-out `test.txt`. Runtime split readers
accept only generalized, dataset-prefixed stems.

### PianoVAM splits and 30 Hz preparation

`configs/pianovam.yaml` trains only on PianoVAM v1.1. Its source and generated
roots are intentionally separate:

```text
source:     /storage/jinxuan/data/AirPianoFull
processed:  /storage2/jinxuan/data/AirPianoFull
splits:     <AirPiano>/assets/splits/pianovam
```

PianoVAM's `metadata.json` is the authoritative synchronization-aware split
source. Generate the ignored local lists with:

```bash
conda run -n v2n python scripts/convert_pianovam_splits.py
wc -l assets/splits/pianovam/*.txt
```

The converter sorts record IDs, prefixes them with `PianoVAM_v1/`, requires the
selected records' MP4 and MIDI files beneath the shared modality roots, merges
`ext-train` into training, and excludes the special blurry and four-hand
subsets. The expected counts are 81 train, 9 valid, and 9 test (99 records
total). Use `--data-root` and `--output-root` only for an alternate installation.

This preset does not reuse `processed_wilor_25fps`. For records without a raw
cache, it first-due samples the native video to 30 Hz before hand detection and
WiLoR inference. Valid completed raw caches are always reused, so earlier 60 Hz
records remain untouched while new records use 30 Hz. Canonicalization reads
each record's effective raw rate and produces uniform 30 Hz NPZs from either.
Before the long run, verify at least 20 GiB free on `/storage2`, GPU 1
availability, and the three licensed WiLoR assets:

```bash
conda run -n v2n python run.py --config-name pianovam --cfg job --resolve
df -h /storage2
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

head -n 1 assets/splits/pianovam/train.txt \
  > /tmp/airpiano_pianovam_smoke.txt
conda run -n v2n python run.py --config-name pianovam \
  stage=preprocess preprocess.task=all \
  'preprocess.split_files=[/tmp/airpiano_pianovam_smoke.txt]'
```

The smoke run writes to the final resumable roots, so the full run reuses it.
Inspect its canonical NPZ before continuing; it must contain `pose`,
`presence`, `fps`, and `schema_version`, with `float32[T,2,21,3]` pose,
`bool[T,2]` presence, and `fps=30.0`.

Run the expensive and cheap phases separately in a durable terminal session:

```bash
conda run -n v2n python run.py --config-name pianovam \
  stage=preprocess preprocess.task=video_to_ragged

conda run -n v2n python run.py --config-name pianovam \
  stage=preprocess preprocess.task=ragged_to_canonical
```

Keep `preprocess.overwrite=false` when restarting. Before changing a requested
raw rate, allow the currently processing record to complete; a partial cache
whose recorded rate does not match the request is deliberately refused to
prevent mixed-rate chunks. Do not delete a partial record or its source video
as part of this rollout. Completed caches are never regenerated automatically,
even if their rate differs from the current setting. Completion requires 99
`_SUCCESS` markers under the configured ragged root and 99 canonical NPZs:

```bash
find /storage2/jinxuan/data/AirPianoFull/wilor/PianoVAM_v1 \
  -name _SUCCESS -type f | wc -l
find /storage2/jinxuan/data/AirPianoFull/airpiano_wilor_2p5d_30fps/PianoVAM_v1 \
  -maxdepth 1 -name '*.npz' -type f | wc -l
```

After both counts reach 99, train and evaluate with:

```bash
conda run -n v2n python run.py --config-name pianovam stage=train
conda run -n v2n python run.py --config-name pianovam stage=test
```

Artifacts live under `outputs/pianovam`. PianoVAM is CC BY-NC-SA 4.0; the
derived cache also remains subject to WiLoR and MANO terms. Do not commit or
redistribute the source data, model assets, or derived pose arrays.

### Combined PianoMotion and PianoVAM training

`configs/full.yaml` reads both datasets directly from the shared modality
roots. Its train list is PianoMotion train followed by PianoVAM train (963
records); validation and test remain PianoMotion-only with 101 and 13 records.
No symlink view or preparation command is required.

This preset is only for training and evaluation. Continue preprocessing with
`baseline` for PianoMotion or `pianovam` for PianoVAM so each corpus retains its
own pre-WiLoR sampling policy. Train and evaluate with:

```bash
conda run -n v2n python run.py --config-name full stage=train
conda run -n v2n python run.py --config-name full stage=test
```

With every canonical file present, the expected pairing summaries are
963/101/13 for train/valid/test. Until then, the configured loader preserves
its normal skip-and-report behavior.

### Extended combined training

The extended split keeps the combined validation and held-out test lists
unchanged while expanding PianoMotion training to 1,852 records. It appends
PianoMotion's `test_full.txt` records other than the 13 held-out test records
to the existing PianoMotion training list. PianoVAM is excluded.

All extended modalities are addressed directly by their dataset-prefixed
record IDs under the shared roots. Train the two extended ablations with:

```bash
conda run -n v2n python run.py --config-name causal_chord24_extended stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended stage=train
```

## Consolidated dataset layout

AirPiano uses modality-first roots with a dataset namespace inside each
modality. Immutable source data lives under `/storage/jinxuan/data/AirPianoFull`;
derived caches live under `/storage2/jinxuan/data/AirPianoFull`.

```text
/storage/jinxuan/data/AirPianoFull/
├── Video/{PianoMotion10M,PianoVAM_v1}/
├── MIDI/{PianoMotion10M,PianoVAM_v1}/
├── Audio/{PianoMotion10M,PianoVAM_v1}/
├── Annotation/PianoMotion10M/
├── Handskeleton/PianoVAM_v1/
├── TSV/PianoVAM_v1/
└── Metadata/{PianoMotion10M,PianoVAM_v1}/

/storage2/jinxuan/data/AirPianoFull/
├── wilor/{PianoMotion10M,PianoVAM_v1}/
├── airpiano_wilor_*_30fps/{PianoMotion10M,PianoVAM_v1}/
├── processed_*/<owning-dataset>/
├── beats/PianoMotion10M/
├── .cache/PianoVAM_v1/
└── Metadata/PianoMotion10M/
```

Split stems carry the same dataset prefix, so preprocessing and loading resolve
directly against these roots without compatibility symlinks.

## Preprocessing contracts

### Video to ragged WiLoR cache

`VideoToWiLoR.run(video_path, output_dir, overwrite=False) -> Path` uses the
in-tree backend and cache implementation. Model objects are loaded lazily on
the first uncached video and reused by later calls on the same component.
Renderer initialization is disabled for headless execution. Ordinary video is
decoded with OpenCV; AV1 or an OpenCV-open failure uses system FFmpeg.
`target_fps=None` processes the native rate. A positive target at or below the
source rate uses deterministic first-due temporal sampling before the backend;
for example, 60 to 30 Hz selects source frames `0, 2, 4, ...`. Upsampling and
non-positive targets are rejected.

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
manifest also has `completed_at`. `video.fps` is the effective cached rate used
by canonicalization. Downsampled caches additionally record `video.source_fps`
and `video.sampling_policy: first_due`; manifests without these optional fields
mean native-rate processing. The remaining video fields describe the probed
source. `model` records absolute upstream paths and asset hashes. `inference`
records all extraction settings. Each chunk inventory entry contains `file`,
`frame_start`, `frame_end`, and `detections`.

Each compressed chunk is cached-frame-major. Let `F` be frames in the chunk and
`D` be all hand detections across them:

| Array | Shape/type and meaning |
| --- | --- |
| `frame_indices` | `int64[F]`, contiguous zero-based indices on the effective cached timeline. |
| `frame_timestamps_seconds` | `float64[F]`, timestamps of the selected source frames. |
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
reuses contiguous completed chunks, advances the sampling-aware decoder past
them, and resumes at the first missing chunk. Resumption refuses an effective
rate, source rate, or sampling-policy mismatch so one record cannot contain
mixed-rate chunks. A valid completed cache is reused before looking for the
source MP4 and without comparing its rate to the current request, so retained
native-rate caches remain reusable and the video may be archived after
successful extraction.

`overwrite=true` removes the selected raw record directory before regeneration.
Treat it as destructive and verify the exact root/stem first.

### Inspect ragged motion on the source video

The standalone ragged visualizer is separate from `stage=visualize`, which
draws inference heatmaps. It requires no WiLoR model and does not produce or
modify pose caches:

```bash
conda run -n v2n python -m airpiano.preprocessing.wilor.visualize \
  --video /storage/jinxuan/data/AirPianoFull/Video/PianoMotion10M/442401135/BV1ED4y1D75s.mp4 \
  --raw-cache /storage2/jinxuan/data/AirPianoFull/wilor/PianoMotion10M/442401135/BV1ED4y1D75s \
  --output outputs/baseline/visualizations/wilor/BV1ED4y1D75s.mp4
```

By default it renders every frame in the manifest's published chunk snapshot.
For a first-due sampled cache, source reading applies the manifest's same policy
so overlays use exactly the frames processed by WiLoR.
Use a half-open time excerpt when a full performance is unnecessary:

```bash
conda run -n v2n python -m airpiano.preprocessing.wilor.visualize \
  --video /absolute/path/to/source.mp4 \
  --raw-cache /absolute/path/to/raw_cache \
  --output /absolute/path/to/overlay.mp4 \
  --start-seconds 30 --duration-seconds 7 --overwrite
```

Every detector box is shown, while skeletons use only the same independently
selected highest-confidence Left and Right detections as canonicalization.
Joint screen positions come from `pred_keypoints_2d_full`. Their color is
absolute full-camera depth:

```text
pred_keypoints_3d[..., 2] + pred_cam_t_full[..., 2]
```

Bones use their endpoint mean depth. One exact selected-joint min/max range and
numeric near/far legend are shared across the rendered interval, so color does
not flicker between frames. Source audio for the interval is retained when it
exists; silent videos remain silent. A cache with `status: processing` can be
inspected through its already-published chunks, and chunks published after the
manifest snapshot are intentionally excluded.

### Raw schema compatibility policy

Runtime has no raw-cache compatibility layer. Any manifest whose schema is not
`airpiano-wilor-ragged-v1`, including `pianomotion-wilor-v1`, raises a
schema-mismatch `ValueError` in both producer reuse checks and canonicalization.
Runtime never reuses or converts a legacy manifest.

The consolidated PianoMotion cache contains the already migrated
`airpiano-wilor-ragged-v1` manifests. Unknown and legacy schemas remain
rejected; storage layout does not relax the producer or canonicalizer.

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
happens before temporal resampling, and optional temporal post-processing runs
on the canonical-rate sequence. Scaling is deliberately deferred to the
configured `HandLoader` transform.

The baseline sets `selector: null`, `postprocessor: null`, and
`minimum_presence: null`, preserving highest-confidence selection until
resampling. The `smoothed` preset instead selects the largest full-image wrist
V independently for each handedness, then marks that slot absent when the
selected detection confidence is below `0.75`. It runs
`PianoMotion10MPostprocessor` after resampling to 30 Hz. Independently for each
hand, the postprocessor rejects pose-level Hampel outliers with a 20-frame
window and 3-MAD threshold, fills bounded gaps up to 29 frames, removes present
runs shorter than 15 frames, and applies an order-3 Savitzky-Golay filter with
an 11-frame window to each remaining run. Leading, trailing, and gaps of 30 or
more frames remain absent and exactly zero.

The multivariate Hampel criterion is an explicit reproduction assumption: for
each present hand frame, it computes a local median and scaled MAD independently
for all 63 pose coordinates, then rejects the frame when the median standardized
residual across those coordinates exceeds `3.0`. Missing frames are excluded
from every local statistic, and all frame decisions use the unchanged input
presence mask before any outliers are marked absent.

After resampling, `smoothed` requires mean presence over both hand slots to be
at least `0.90`. A rejected record raises a dedicated presence exception before
writing its canonical NPZ; offline preprocessing reports the rejection and
continues with the next record. The postprocessor does not track candidates or
reject identity switches, so an incorrect selected hand can still influence
nearby smoothed frames.

Absent slots are exact zero. Mixed input rates are sampled at target-frame
centers using linear pose interpolation; a hand is present only if present at
both endpoints, otherwise the full target slot is zero. Output length is:

```text
floor(source_frames * target_fps / source_fps)
```

Canonical resampling happens here only. `HandLoader` does not resample.

When `overwrite=false`, an existing canonical path is trusted and returned
before raw input is inspected. Otherwise offline orchestration skips and reports
only a missing raw directory, `manifest.json`, or `_SUCCESS`. Once both markers
exist, schema mismatch, malformed JSON, corrupt chunks, missing fields, and
algorithm failures propagate directly.

## Dataset, model, and outputs

### MIDI and eager dataset

`MidoNoteReader` parses the merged tempo-aware MIDI clock, treats zero-velocity
note-on as note-off, pairs repeated channel/pitch notes FIFO, ignores sustain,
and keeps positive-duration pairs. `MidiLoader` expands those physical notes
into an immutable `EventSequence`: onset actions are always present and offset
actions are opt-in. Every `NoteEvent` retains its physical `Note` and exposes
its action type, pitch, seconds, and optional tick.

`InitializerChordMerger` groups later events within 0.05 seconds of the
initializer event; grouping is inclusive and initializer-anchored rather than
transitive. Every resulting `Chord` stores its events, separate onset/offset
note tuples, its selected representative time, its duration, and a precomputed
`float32[200]` duration one-hot. `InitializerChordMerger.time_mode` selects the
first event time by default or the arithmetic mean of all grouped event times.
Raw duration is the gap between successive selected chord times. The final
onset-only chord lasts to its latest member release; the final chord in an
offset-enabled sequence has duration zero. Configurations may transform
durations before quantization without changing these raw values.

With `MidoNoteReader.quantization=true`, the reader also consumes the paired
beat text file's first column. Beat zero maps to tick zero and each following
beat advances by `subdivision` ticks (12 by default); note endpoints snap to
the nearest tick, with half-ties moving to the later tick. Times before or
after the detected beat span extrapolate the first or last beat interval.
Original MIDI seconds remain on every note for hand slicing and targets, while
the added integer ticks control chord grouping and duration tokens. Missing
beat files are reported and omitted during pairing; malformed present files
raise normally.

`OffsetScaleTransform` applies configured per-axis `(pose - offset) * scale` to
present slots while preserving exact zero absent slots. The baseline uses
offset `[0.5,0.5,0]` and scale `[1,1,10]`; `wrist_xyz10` uses zero offset and
scale `[10,10,10]`. `FileLoader` eagerly loads every paired song.
`AlignmentDataset` precomputes frame bounds and chord selection indices, omits
windows without real frames or chord candidates, then slices real tensors and
generates targets in `__getitem__`. An optional training-only hand-window shift
independently jitters the two precomputed frame bounds while retaining the same
chord selection. Padding and masks are added only when a batch is collated.

One baseline item is:

| Field | Type/shape | Meaning |
| --- | --- | --- |
| `hands` | `float32[T,2,21,3]` | Real transformed left/right frames; baseline `T<=210`. |
| `hand_presence` | `bool[T,2]` | Per-hand detection flags. |
| `chords` | binary `float32[C,456]` | 128 onset-pitch bits, 128 offset-pitch bits, and a 200-way duration one-hot; baseline `C<=128`. |
| `chord_ids` | `int64[C]` | Global IDs for the selected chords. |
| `target` | `float32[T,C]` | Soft chord-action alignment. |
| `record_id` | string | Complete root-relative stem. |
| `frame_start` | integer | Global left-edge frame index. |
| `fps` | float | Canonical/source clock used by the slicer. |

The slicer emits regular 105-frame starts plus the exact right-aligned final
window. It keeps every chord whose onset or offset notes physically overlap the
half-open window, appends later chords until the configured fixed width is
available or the score ends, and returns only real frames and chords. More
matching chords than that width raises instead of truncating positives.
Duplicate pitch/action pairs remain distinct internally but collapse to one
multi-hot entry.

During MIDI loading, every chord's duration is quantized using
MIDI2ScoreTransformer's performance-MIDI mapping. By default the input is its
absolute duration. When `duration_scale_range` is configured, one scale is
sampled uniformly per song and applied to every duration. Alternatively,
`duration_normalization: song_mean` deterministically replaces each raw chord
duration `d_i` with its value in mean-chord-duration units:

```text
normalized_duration_i = d_i * chord_count / sum(raw_chord_durations)
                      = d_i / mean(raw_chord_durations)
```

The final chord's sustain duration participates in both the sum and chord
count. Consequently, the normalized durations have mean `1.0` before
quantization and remain unchanged by uniformly scaling the timing of the whole
song. Raw `Chord.duration` values remain in seconds. The selected duration is
then mapped to a bin using:

```text
bin = clamp(floor(200 * log(4 * duration_seconds + 1) / log(65)), 0, 199)
```

The resulting one-hot is stored as `Chord.quantized_dur`; slicing appends it to
the 256 onset/offset pitch bits. Both transformations preserve within-song duration
ratios, while song-mean normalization removes global tempo scale
deterministically. Values outside the mapping's representable range retain the
existing clipping behavior. Quantization happens before window selection, so
the normalization covers the full song and the last chord in a slice does not
incorrectly receive the final-chord rule.

Quantized events at the same tick form one chord even when onsets and offsets
are mixed. Durations use the integer gap to the next chord tick; the final
onset-only chord uses its latest member-release tick and the final
offset-enabled chord uses zero. The value is clamped to `[0,199]` and directly
selects the duration bin. The quantized presets use 12 subdivisions per beat,
chord-duration conditioning, and the beat root
`/storage2/jinxuan/data/AirPianoFull/beats`.

With `mode=chord`, `segment_frames` and `stride_frames` instead count chord
tokens and `chord_count` is ignored. Token windows retain the exact
right-aligned tail. The first score window starts at song frame 0; other
windows start at the frame at or before their first chord time. The final
score window ends at song end; other windows include the first sampled frame
at or after their last chord time plus the configured end padding. End padding
defaults to zero. `hand_context_chords` can move these hand-frame boundaries to
earlier and later neighboring chord times without adding those neighbors to
the model's chord tokens or targets.

`crop_to_chord_bounds=true` trims only the score's outer boundaries. A window
that begins with the song's first chord starts at the sampled frame at or before
that chord time instead of frame zero. A window that reaches the song's final
chord ends at `chord.time + chord.duration` instead of continuing to the
hand-sequence end. Interior window bounds retain the normal context and
end-padding behavior.
`max_zero_hand_fraction` optionally removes
windows whose fraction of frames with neither hand present is strictly greater
than the configured value; frames with either hand present do not count as
zero-hand frames.

Targets use global frame times and each chord's selected action time plus the
configured event delay, which defaults to zero:

```text
target[t,c] = max(0, 1 - abs(global_frame_time[t]
                              - (chord_time[c] + delay)) / 0.2)
```

The collator pads both axes to the batch maxima, creates `frame_mask` and
`chord_mask`, uses `-1` for padded chord IDs, and leaves padded target cells
zero. Missing hands do not remove frames, targets, or loss positions.

### Fixed model and alignment objectives

`AirPianoTransformer` takes:

```text
hands          [B,T,2,21,3]
hand_presence  [B,T,2]
chords         [B,C,456]
frame_mask     [B,T]
chord_mask     [B,C]
```

and returns logits `[B,T,C]`. It embeds 126 hand coordinates plus two presence
bits and always embeds the first 128 onset-pitch values. When
`model.include_chord_offsets=true`, a separate embedding processes the next
128 offset-pitch values and is added to the onset embedding. When
`model.include_chord_duration=true`, a separate embedding processes the last
200 values and is added to the pitch embedding. Four chord transformer layers
and four frame self/cross-attention blocks feed 32-wide pairwise projections.
The score bias starts from positive prior `0.0055`.
Absent hand coordinates are zeroed, presence remains an input feature, padded
attention is protected from NaNs, and padded logits are zero. When
`model.zero_hand_input=true`, all 126 coordinates and both presence bits are
replaced by zero before hand embedding; frame masks and temporal positional
encoding remain active. When
`model.causal_frame_attention=true`, frame self-attention at time `t` sees only
frames through `t`; chord encoding and frame-to-chord attention still see all
selected chord tokens. The checkpoint's saved model settings, including
zero-hand-input mode, control inference. Older checkpoints default to
non-causal, pitch-only operation with hand input enabled; they cannot initialize
a duration-conditioned model.

`masked_bce_loss` is ordinary `BCEWithLogitsLoss` over:

```text
frame_mask[:, :, None] & chord_mask[:, None, :]
```

It rejects a batch with no valid cell. When
`alignment_loss=cross_entropy`, the loss instead selects the one-hot target's
class index for every real frame with an active target, masks padded chord
logits, and averages categorical cross-entropy over those frames. Zero-target
frames remain model inputs without contributing to the loss. Inference uses
sigmoid for BCE and softmax over real chord slots for cross-entropy. Existing
AirPiano checkpoints remain compatible after the module split; V2N checkpoints
do not.

### Fixed local output trees

Training and test artifacts remain in the configured project directory:

```text
outputs/baseline/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── inference/
│   └── test/
│       ├── *.npz
│       ├── *.png
│       ├── *.mid
│       ├── *.gt.mid
│       ├── onset_delta_summary.json
│       ├── onset_delta_summary.png
│       └── test_loss.json
├── logs/
│   └── wandb/
```

Realtime-style stages keep inputs and preprocessing caches together under their
explicitly configured input root. Cache directory names match the final
component of `data.raw_wilor_root` and `data.hand_root`, so preprocessing presets
use the same layout as their offline datasets:

```text
<runtime-root>/
├── video/<stem>.mp4
├── midi/<stem>.mid
├── wilor/<stem>/
└── airpiano_wilor_2p5d_30fps/<stem>.npz

<output.dir>/
├── realtime/<stem>/
│   ├── *.npz
│   └── realtime_loss.json
└── full_song/
    ├── *.npz
    ├── *.png
    ├── *.mid
    ├── *.gt.mid
    ├── onset_delta_summary.json
    ├── onset_delta_summary.png
    └── full_song_loss.json
```

There is no version or run-identity layer. Starting training a second time in
the same project directory is unsupported: checkpoints are replaced while
older inference files and logs may remain, producing a mixed tree. AirPiano
does not detect, clear, or resume such a run. Before an intentional rerun,
inspect and manually archive or delete the complete `outputs/baseline/`
directory.

Every configurable path follows one rule: absolute paths are used unchanged,
and relative paths are resolved from the AirPiano repository root, independent
of the directory from which `run.py` is launched. Defaults that belong to a
project explicitly include `${output.dir}`, such as
`${output.dir}/checkpoints/last.pt`.

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
`realtime_loss.json`. Full-song inference writes one artifact per complete
recording plus `full_song_loss.json`. There is no inference manifest.

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
| `loss` | `float64` scalar per-item configured alignment loss. |

`T` and `C` are the padded maxima of the item's inference batch; the masks
identify the real slice extents. With a one-item batch they are the real frame
and chord counts.

Artifact filenames replace `/` and `\\` in the record ID with `__`, then append
`__f<frame_start as nine digits>.npz`. Canonical files and inference artifacts
are unchanged by the generalized path migration.

Full-song artifacts use the same schema with `frame_start=0`, all-true masks,
and `[all frames, all chords]` probability and target arrays. At each frame,
chords outside the active score window have probability zero.

## Stage workflows and commands

All commands run from the AirPiano root. Hydra is configured not to change the
working directory.

### Inspect and convert splits

```bash
conda run -n v2n python run.py --cfg job --resolve
conda run -n v2n python scripts/convert_legacy_splits.py
conda run -n v2n python scripts/convert_pianovam_splits.py
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
  data.raw_wilor_root=/storage2/jinxuan/data/AirPianoFull/airpiano_wilor_raw
```

For a small smoke list:

```bash
conda run -n v2n python run.py stage=preprocess \
  preprocess.task=all \
  'preprocess.split_files=[/absolute/path/to/smoke.txt]'
```

To reuse the same ragged detections while generating the opt-in smoothed cache:

```bash
conda run -n v2n python run.py --config-name smoothed \
  stage=preprocess preprocess.task=ragged_to_canonical
```

This preset writes accepted canonical files to
`${data.processed_root}/airpiano_wilor_2p5d_smoothed_30fps` and project artifacts
to `outputs/smoothed/`. Canonical NPZs do not embed post-processing settings,
and an existing output is trusted when `preprocess.overwrite=false`. After
changing smoothing settings, select another `data.hand_root` or deliberately
regenerate the selected canonical files with `preprocess.overwrite=true`; the
ragged cache itself remains reusable.

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

### Camera-space wrist-delta ablation

`wrist_xyz10_causal_chord24_extended_wrist_delta` keeps joints 1 through 20
in the same wrist-relative local XYZ representation as `wrist_xyz10`. Joint
zero instead stores the camera-space wrist displacement between adjacent
canonical 30 Hz frames:

```text
delta[t] = (pred_keypoints_3d[t, wrist] + pred_cam_t_full[t])
         - (pred_keypoints_3d[t-1, wrist] + pred_cam_t_full[t-1])
```

The first wrist delta is zero. A wrist delta is also zero unless that hand is
present in both adjacent frames, preventing a reappearance after a gap from
creating an artificial jump. Deltas are computed after resampling and receive
the same online ×10 scale as every other XYZ coordinate. The resulting tensor
shape and model architecture are unchanged, and no absolute wrist position is
passed to the model.

Check free space, then generate the separate PianoMotion cache. The explicit
split override includes the `test_full.txt` records used by the extended
training split:

```bash
df -h /storage2

conda run -n v2n python run.py \
  --config-name wrist_xyz10_wrist_delta \
  stage=preprocess preprocess.task=ragged_to_canonical \
  'preprocess.split_files=[assets/splits/pianomotion10m/train.txt,assets/splits/pianomotion10m/valid.txt,assets/splits/pianomotion10m/test_full.txt]'
```

Generate the PianoVAM cache from its three configured splits:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_wrist_delta_pianovam \
  stage=preprocess preprocess.task=ragged_to_canonical
```

Train and evaluate after the canonical roots contain the required records:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta stage=test
```

The corpus caches live under
`${data.processed_root}/airpiano_wilor_wrist_xyz_delta_30fps`; experiment
artifacts use
`outputs/wrist_xyz10_causal_chord24_extended_wrist_delta/`. Existing
wrist-relative caches are not read or overwritten.

The filtered ablation reuses the same cache and model settings, trims the first
and final windows to the score boundaries, and removes windows with more than
20% frames where neither hand is present:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20 \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20 \
  stage=test
```

Its artifacts use
`outputs/wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20/`.

The two quantized wrist-delta ablations inherit score-bounded outer windows
and the 20% fully-missing-hand frame limit while reusing the same hand cache.
The first adds beat-grid duration tokens; the second also expands MIDI into
onset and offset actions, selects mean event time for each chord, and enables
the offset-pitch embedding:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_offset \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_offset_last_chord_softmax \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_offset_last_chord_softmax \
  stage=test
```

The final variant retains the quantized onset/offset action sequence and uses
the persistent last-action target with categorical cross-entropy.

Three additional filter20 ablations isolate duration normalization, target
generation, and chord timing. The first replaces beat-grid duration tokens
with deterministic song-mean-normalized durations. The second keeps beat-grid
durations but trains against the persistent last-chord target with categorical
cross-entropy. The third keeps the quantized input and onset target while using
the mean grouped-event time instead of the first event time, without adding
note-offset actions:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_chord_duration_mean_normalized \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_chord_duration_mean_normalized \
  stage=test
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_last_chord_softmax \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_last_chord_softmax \
  stage=test
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_mean_time \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_wrist_delta_zero_hand_filter20_quantized_mean_time \
  stage=test
```

### Causal 24-chord ablation

This combined ablation uses causal frame self-attention and slices the score
into 24-chord windows with a 12-chord stride. It reuses the baseline canonical
hand cache and writes all outputs under `outputs/causal_chord24/`:

```bash
conda run -n v2n python run.py --config-name causal_chord24 stage=train
conda run -n v2n python run.py --config-name causal_chord24 stage=test
```

### Last-played-chord softmax ablation

`wrist_xyz10_causal_chord24_extended_last_chord_softmax` inherits the extended
wrist-relative causal 24-chord preset. It replaces the triangular onset target
with one hard target per target-bearing frame: a chord becomes active at its
selected action time, remains active until the next chord time, and the final
chord remains active through the slice end. Direct target generation returns
zero before the first supplied onset. Windowed datasets crop each slice to the
first sampled frame at or after its first selected chord time, ensuring every
windowed target row contains exactly one class. Full-song inference retains its
original complete frame history and does not apply the dataset crop.

The model architecture is unchanged. Training uses categorical cross-entropy
over the 24 chord slots, test and realtime artifacts contain softmax
probabilities, and full-song inference retains the original complete frame
history and moving-window controller rather than applying the dataset crop:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_last_chord_softmax \
  stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_extended_last_chord_softmax \
  stage=test
```

Outputs are written under
`outputs/wrist_xyz10_causal_chord24_extended_last_chord_softmax/`. Existing
visualization and sonification behavior is unchanged.

### Random hand-window shift ablation

`wrist_xyz10_chord24_random_hand_shift2s` inherits the non-causal
`wrist_xyz10_chord24` preset. Each time a training item is read, its left and
right hand-window edges receive independent integer offsets sampled uniformly
from `[-60, 60]` frames (plus or minus two seconds at 30 Hz). Shifted bounds
are clamped to the real song, and crossed or empty bounds are resampled. The 24
selected chord tokens remain unchanged, while hands, targets, and `frame_start`
use the shifted interval. Validation, test, and realtime windows remain
unshifted for comparable evaluation:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_chord24_random_hand_shift2s stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_chord24_random_hand_shift2s stage=test
```

The preset reuses the wrist-relative canonical cache and writes under
`outputs/wrist_xyz10_chord24_random_hand_shift2s/`.

### Wrist-relative causal diagnostic ablations

These presets inherit `wrist_xyz10_causal_chord24`, including its
wrist-relative cache, causal frame attention, and 24-chord windows with stride
12. The first zeros all hand coordinates and presence bits so predictions
depend on chord tokens, frame position, and masks:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_zero_hand stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_zero_hand stage=test
```

The second delays every target onset by `1.0` second. It retains each chord
window's original start for causal history and extends non-final window ends by
the same duration, clamped to the real song. Test, realtime, full-song,
visualization, and prediction MIDI outputs retain the delayed timeline:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_target_delay1s stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_target_delay1s stage=test
```

Outputs are written below directories matching the two config names under
`outputs/`. Neither preset requires a new canonical hand cache.

### Stride-1 chord-window ablation

This preset also inherits `wrist_xyz10_causal_chord24`, but changes the
training and test chord-window stride from 12 tokens to 1 while keeping the
window width at 24. For long songs this produces roughly 12 times as many
highly overlapping windows and optimizer updates because the 20-epoch training
schedule is unchanged. The full-song trailing trigger remains at 12 chords so
its controller behavior is not part of the ablation:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_stride1 stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_stride1 stage=test
```

The preset reuses the existing wrist-relative canonical hand cache and writes
under `outputs/wrist_xyz10_causal_chord24_stride1/`.

### Neighbor-onset hand-window ablation

This preset keeps the base 24 chord tokens and stride 12, but expands only the
corresponding hand window by one neighboring chord onset on each side. A middle
window starts at the onset preceding its first selected chord and includes the
first sampled frame at or after the onset following its last selected chord.
At score boundaries it instead uses song frame zero or the real song end. The
neighboring chords remain context only and receive neither chord tokens nor
target columns:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_hand_context1 stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_hand_context1 stage=test
```

The preset reuses the wrist-relative canonical cache and writes under
`outputs/wrist_xyz10_causal_chord24_hand_context1/`.

### Chord-duration ablation

This ablation keeps the baseline frame slicing but enables the separate
duration embedding for the always-present duration portion of each chord
token. It writes outputs under `outputs/chord_duration/`:

```bash
conda run -n v2n python run.py --config-name chord_duration stage=train
conda run -n v2n python run.py --config-name chord_duration stage=test
```

A duration-enabled model requires a duration-enabled checkpoint. Pitch-only
checkpoints retain their original 128-input pitch projection and ignore the
packed duration suffix.

The wrist-relative causal variant below samples one uniform duration scale in
`[0.5, 2.0]` for each loaded song. The same scale is used for all of that
song's chord-duration tokens in training, validation, test, and realtime:

```bash
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_chord_duration_relative stage=train
conda run -n v2n python run.py \
  --config-name wrist_xyz10_causal_chord24_chord_duration_relative stage=test
```

The extended-data variant below instead uses deterministic song-mean
normalization. It inherits the standard hand coordinates, causal frame
attention, and 24-chord windows with stride 12 from `causal_chord24_extended`,
and writes to
`outputs/causal_chord24_extended_chord_duration_mean_normalized/`:

```bash
conda run -n v2n python run.py \
  --config-name causal_chord24_extended_chord_duration_mean_normalized stage=train
conda run -n v2n python run.py \
  --config-name causal_chord24_extended_chord_duration_mean_normalized stage=test
```

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

### SKY-Piano motion conversion

Convert the downloaded imputed SKY-Piano marker CSVs directly to the
unscaled canonical representation used by the `wrist_xyz10` presets:

```bash
conda run -n v2n python scripts/convert_skypiano_motion.py
```

By default, this reads `assets/sky_piano/motion_hand/*_imputed.csv` and writes
matching NPZs below
`assets/sky_piano/airpiano_wilor_wrist_xyz_30fps/`. It maps the 21 markers per
hand into WiLoR/OpenPose joint order, subtracts each wrist, converts millimeters
to meters, and rotates the SKY coordinates 180 degrees around Y to match the
PianoMotion top-camera convention. Existing outputs are reused; pass
`--overwrite` to replace them. The configured `HandLoader` still applies the
online factor of 10.

### Realtime batch

Place synchronized pairs in the configured root's `video/` and `midi/`
directories. Matching is by the complete basename, so dotted names are
preserved:

```text
<runtime-root>/
├── video/performance.v2.mp4
├── midi/performance.v2.mid
├── wilor/performance.v2/
└── airpiano_wilor_2p5d_30fps/performance.v2.npz

<output.dir>/
└── realtime/performance.v2/
```

Run every direct pair with the default `last.pt` checkpoint:

```bash
conda run -n v2n python run.py stage=realtime \
  realtime.root=/data/performance
```

With the baseline config, outputs for this example are written below
`outputs/baseline/realtime/performance.v2/`. Video or MIDI files without a
matching counterpart are reported and skipped. Only direct files inside the
two modality directories are scanned. Canonicalization is attempted first, so
an existing canonical file or ragged cache avoids constructing WiLoR. If the
ragged input is missing, realtime runs video-to-ragged and retries
canonicalization. Components are reused across the sorted pairs. Override
`realtime.output_dir` to select another inference destination.

Select a different root or checkpoint as needed:

```bash
conda run -n v2n python run.py stage=realtime \
  realtime.root=/data/performance \
  realtime.checkpoint=checkpoints/best.pt
```

Each basename becomes `record_id`. Realtime calls the same configured
`VideoToWiLoR`, `WiLoRToCanonical`, `HandLoader`, slicer, target generator, and
checkpoint inference as the offline workflow. Every video/MIDI pair must
describe the same performance and timeline.

### Causal full-song inference

The `full-song` stage accepts the same synchronized MP4/MID layout and
preprocessing cache structure as `realtime`. It supports only chord-mode
slicing and checkpoints trained with causal frame attention:

```bash
conda run -n v2n python run.py --config-name causal_chord24 \
  stage=full-song \
  full_song.root=/data/performance
```

For every arriving video frame, the controller forwards only the current frame
history against the active 24-chord window and permanently commits the newest
probability row. Once any of the trailing 12 chords reaches the inclusive `0.2`
running-peak threshold, the score window advances by one on the next frame. Its
history then begins at the earliest predicted peak of the new first chord. After
the final score window is reached, it remains active through the end of the
video. Override `full_song.alignment_threshold` or
`full_song.advance_after_chords` to change the trailing trigger width.

Outputs are written directly under `full_song.output_dir`, which defaults to
`${output.dir}/full_song`, one NPZ per input basename, with a weighted aggregate
loss in `full_song_loss.json`.

### Visualize and sonify batches

Both stages recursively scan `*.npz` files under `inference/test` by default:

```bash
conda run -n v2n python run.py stage=visualize
conda run -n v2n python run.py stage=sonify sonify.threshold=0.2
```

For an input named `test.npz`, all derived files are written beside it:

```text
outputs/baseline/inference/test/test.png
outputs/baseline/inference/test/test.mid
outputs/baseline/inference/test/test.gt.mid
outputs/baseline/inference/test/onset_delta_summary.json
outputs/baseline/inference/test/onset_delta_summary.png
```

NPZ files are processed recursively in sorted order. Dotted basenames and
nested directory structure are preserved, and each rendered sidecar is written
beside its source artifact. Relative roots are resolved from the AirPiano
repository root:

```bash
conda run -n v2n python run.py stage=visualize \
  visualize.root=outputs/baseline/inference/test-best

conda run -n v2n python run.py stage=sonify \
  sonify.root=/absolute/path/to/artifacts
```

Visualization draws probability and target heatmaps with shared `[0,1]`
scaling, followed by a signed `prediction - target` residual heatmap with a
zero-centered `[-1,1]` scale. Sonification selects each valid chord's earliest
maximum frame and emits all active pitches when that maximum is at least the
threshold. Notes are
one second at velocity 64 by default. It also resolves
`<sonify.midi_root>/<record_id>.mid` and writes the source notes beginning in
the real, unpadded slice. Ground-truth pitch, velocity, and duration are
preserved, offsets are clipped at the slice end, and timing is rebased to zero.
Notes already active before the slice are omitted. Both outputs use one track
at 120 BPM; pedals, programs, channels, source tempo events, and other MIDI
metadata are not preserved. Timing is slice-relative and overlapping slices
are not merged.

Sonification also matches artifact source IDs to source-MIDI chords and keeps
each valid chord's timing calculation in memory. Onsets use the absolute source
clock. The signed delta is `predicted_onset_seconds - real_onset_seconds`, so a
positive value is late. Chords below the inclusive threshold count as misses
and do not contribute a delta. The configured chord tolerance must match the one
used to build the artifact.

After processing every artifact, sonification writes only the final onset
metrics: `onset_delta_summary.json` and `onset_delta_summary.png` at
`sonify.root`. They group all chunk observations by relative `chord_id`, so a
chord-24 configuration has at most IDs 0 through 23. The JSON reports
observation, detected, and missed counts plus the mean signed and absolute
deltas over detected observations; an ID that is never detected has null means.
The PNG contains aligned bar charts of mean absolute delta seconds and missing
prediction count by relative chord ID.

For realtime artifacts, point the artifact root at the parent of the per-record
directories and the MIDI root at the runtime input directory:

```bash
conda run -n v2n python run.py stage=sonify \
  sonify.root=outputs/baseline/realtime \
  sonify.midi_root=/data/performance/midi
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
| `stage` | `train` | One of the seven supported handlers, or an ordered Hydra list. |
| `device` | `cuda:1` | Training and WiLoR device. |
| `seed` | `42` | Python, NumPy, Torch, CUDA, and shuffle seed. |
| `output.root` | `outputs` | Parent of generated project artifacts. |
| `output.project_name` | `baseline` | Single local project directory name. |
| `output.dir` | `${output.root}/${output.project_name}` | Explicit project artifact directory; relative values resolve from the AirPiano root. |
| `data.root` | `/storage/jinxuan/data/AirPianoFull` | Parent of immutable source modalities. |
| `data.processed_root` | `/storage2/jinxuan/data/AirPianoFull` | Parent of generated modalities. |
| `data.midi_root` | `${data.root}/MIDI` | `<dataset>/<stem>.mid` files. |
| `data.video_root` | `${data.root}/Video` | `<dataset>/<stem>.mp4` source videos. |
| `data.raw_wilor_root` | `${data.processed_root}/wilor` | `<dataset>/<stem>/` current-schema raw caches. |
| `data.hand_root` | `${data.processed_root}/airpiano_wilor_2p5d_30fps` | `<dataset>/<stem>.npz` canonical files. |
| `data.splits.*` | `assets/splits/pianomotion10m/*.txt` | Ignored local dataset-prefixed stems. |

The `pianovam` preset uses the same shared roots with
`assets/splits/pianovam/*.txt`. It requests 30 Hz first-due sampling before
WiLoR for missing raw records and retains the baseline 30 Hz canonical rate,
210-frame, 105-stride, 128-chord, and pose-transform settings.

The training/evaluation-only `full` preset uses the same shared modality roots
with `assets/splits/full/*.txt`; dataset-prefixed stems select the corpus.

### WiLoR and canonicalization

| Field | Baseline | Meaning |
| --- | --- | --- |
| `preprocess.task` | `ragged_to_canonical` | `video_to_ragged`, `ragged_to_canonical`, or `all`. |
| `preprocess.split_files` | all three configured splits | First-seen ordered union to process. |
| `preprocess.overwrite` | `false` | Reuse compatible raw/canonical products; `true` regenerates selected work. |
| `video_to_ragged._target_` | `airpiano.preprocessing.video.VideoToWiLoR` | Public extraction component. |
| `video_to_ragged.target_fps` | `null` | Optional pre-WiLoR first-due sampling rate; PianoVAM overrides this to `30.0`. |
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
| `chunk_frames` | `512` | Effective cached frames per resumable chunk. |
| `fast` | `false` | Enable upstream half/skip-block fast path. |
| `compile` | `false` | Compile the backbone when fast mode is used. |
| `ragged_to_canonical._target_` | `airpiano.preprocessing.canonical.WiLoRToCanonical` | Current raw schema to canonical NPZ. |
| `ragged_to_canonical.selector` | `null` | Optional ragged-detection selector; `smoothed` uses lowest wrist V followed by a `0.75` confidence cutoff. |
| `ragged_to_canonical.postprocessor` | `null` | Optional canonical-rate `(pose, presence, fps) -> (pose, presence)` step after resampling. |
| `ragged_to_canonical.minimum_presence` | `null` | Optional final mean-presence gate evaluated after resampling and before writing. |
| `coordinates._target_` | `airpiano.preprocessing.canonical.WiLoRCoordinateConverter` | Convert each selected raw detection into one canonical hand. |
| `coordinates.xy_source` | `normalized_2d` | Use normalized full-image XY; `wrist_relative_3d` selects local XYZ relative to the wrist. |
| `coordinates.z_source` | `wrist_relative_3d` | Local WiLoR depth relative to the selected hand's wrist. |
| `coordinates.wrist_source` | `converted` | Keep the converted joint zero; `camera_3d` temporarily inserts the camera-space wrist for a canonical-rate postprocessor. |
| `target_fps` | `30.0` | Canonical output rate. |

The `lowest_v` preset changes only candidate selection: independently for each
handedness and frame, it chooses the detection whose wrist has the largest
full-image V coordinate. Its selector confidence is `0.0`, so it adds no guard
beyond the detector threshold already applied when building the ragged cache.
It retains the baseline postprocessor and presence-gate values (`null`) and
writes canonical records to a distinct root.

The `smoothed` preset configures `LowestWristHandSelector(0.75)`,
`PianoMotion10MPostprocessor(hampel_window_frames=20,
hampel_n_sigma=3.0, max_missing_run_frames=29,
min_presence_run_frames=15, savgol_window_frames=11,
savgol_polyorder=3)`, and a `0.90` final presence gate. It writes accepted
records to a distinct canonical root.

These code and asset paths use the same repository-root resolver as every
other AirPiano path. Hydra also keeps `job.chdir: false`, so it never relocates
the process into an output directory.

### Dataset components

| Field | Baseline behavior |
| --- | --- |
| `file_loader` / `RelativeStemPairer` | Pair `<stem>.mid` and `<stem>.npz`, optionally require `<stem>.txt` beats, report the intersection, then read full songs. |
| `MidoNoteReader` | Tempo-aware physical note parsing, with optional beat-grid tick quantization. |
| `MidiLoader.include_offsets` | `false`; optionally expand physical notes into both onset and offset actions. |
| `InitializerChordMerger.tolerance_seconds` | Initializer-relative grouping at `0.05`, with absolute chord durations assigned after grouping. |
| `InitializerChordMerger.time_mode` | `first`; select the first grouped event time, or `mean` to average all grouped event times. |
| `InitializerChordMerger.duration_scale_range` / `duration_normalization` | Baseline `null` / `none`; optionally apply one random song scale or deterministic `song_mean` normalization before duration quantization. |
| `CanonicalNpzReader` | Direct trusted canonical array loading. |
| `OffsetScaleTransform` | Apply configured per-axis offset and scale, then restore absent slots to zero. |
| `FutureChordSlicer` | Frame mode: 30 Hz, 210 frames, stride 105, fixed 128 future-extending candidates. Chord mode can add end padding, neighboring-action context, or opt-in chord-bound cropping. Every chord token contains onset pitch, offset pitch, and duration features. |
| `FutureChordSlicer.crop_to_chord_bounds` | `false`; trim the first and final chord-mode windows to the score boundaries while retaining normal context and padding for interior bounds. |
| `FutureChordSlicer.max_zero_hand_fraction` | `null`; optionally reject windows whose fraction of frames with neither hand present exceeds the configured value. |
| `ChordOnsetEvents` | One target at each selected chord action time plus `delay_seconds`. |
| `TriangularSoftener.radius_seconds` | Continuous target radius `0.2`. |

Each `_target_` is a replaceable seam, but its parent-facing contract must stay
stable. This includes the model's five-input, `[B,T,C]` output contract. The
loss intentionally remains fixed in code.

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
| `dataset.random_hand_shift_seconds` | `0.0` | Maximum signed shift independently sampled for each training hand-window edge; validation and inference remain unshifted. |
| `model._target_` | `airpiano.model.transformer.AirPianoTransformer` | Configured alignment model instantiated for training and restored from checkpoints. |
| `model.include_chord_duration` | `false` | Consume the packed 200-bin duration suffix through a separate embedding. |
| `model.include_chord_offsets` | `false` | Consume the packed 128 offset-pitch values through a separate embedding. |
| `model.zero_hand_input` | `false` | Replace all coordinates and hand-presence features with zero before embedding. |
| `wandb.project` | `air_piano` | Required online project. |
| `wandb.name` | `${output.project_name}` | Run name shared with the local output directory. |
| `wandb.entity` | `null` | Optional team/entity. |
| `wandb.dir` | `${output.dir}/logs` | W&B parent directory. |
| `test.checkpoint` | `${output.dir}/checkpoints/last.pt` | Checkpoint loaded by `stage=test`. |
| `test.output_dir` | `${output.dir}/inference/test` | Test artifact and loss directory. |
| `realtime.root` | `null` | AirPiano-relative or absolute root containing matching `video/*.mp4` and `midi/*.mid` files. |
| `realtime.checkpoint` | `${output.dir}/checkpoints/last.pt` | Default AirPiano checkpoint. |
| `realtime.output_dir` | `${output.dir}/realtime` | Per-record inference artifact and loss parent directory. |
| `realtime.overwrite` | `false` | Raw and canonical preprocessing reuse policy. |
| `full_song.root` | `null` | AirPiano-relative or absolute root containing matching `video/*.mp4` and `midi/*.mid` files. |
| `full_song.checkpoint` | `${output.dir}/checkpoints/last.pt` | Causal AirPiano checkpoint loaded once for the stage. |
| `full_song.output_dir` | `${output.dir}/full_song` | Full-song inference artifact and aggregate loss directory. |
| `full_song.overwrite` | `false` | Raw and canonical preprocessing reuse policy. |
| `full_song.alignment_threshold` | `0.2` | Inclusive running-peak confidence required by any trailing trigger chord. |
| `full_song.advance_after_chords` | slicer stride | Number of trailing chord slots checked to trigger advancement. |
| `visualize.root` | `${output.dir}/inference/test` | Filesystem tree recursively scanned for NPZs and receiving adjacent PNG files. |
| `sonify.root` | `${output.dir}/inference/test` | Filesystem tree recursively scanned for NPZs and receiving adjacent MIDI sidecars plus final onset-delta JSON and PNG summaries. |
| `sonify.midi_root` | `${data.midi_root}` | Root used with the artifact `record_id` to locate source MIDI. |
| `sonify.threshold` | `0.2` | Inclusive chord peak threshold. |
| `sonify.chord_tolerance_seconds` | Dataset chord tolerance | Retained for call compatibility; onset reports use each NPZ target's peak frame. |
| `sonify.duration_seconds` | `1.0` | Fixed output-note duration. |
| `sonify.velocity` | `64` | Fixed note-on velocity. |

All values in this table that represent paths accept either an absolute path or
a path relative to the AirPiano repository root. Project placement is explicit
through `${output.dir}` rather than being inferred by individual stages.

## Extension contracts

The narrow replaceable interfaces are:

| Component | Required interface |
| --- | --- |
| `VideoToWiLoR` | `run(video_path, output_dir, overwrite=False) -> Path`; emit the configured canonicalizer's raw contract. |
| `WiLoRToCanonical` | `run(raw_record_dir, output_path, overwrite=False) -> Path`; emit canonical pose NPZ or raise the configured presence rejection before writing. |
| Canonical hand selector | `(raw_arrays) -> int64[frames,2]` absolute detection indices, using `-1` for absence. |
| Canonical postprocessor | `(pose, presence, fps) -> (pose, presence)` after canonical resampling, with unchanged `[T,2,21,3]`/`[T,2]` shapes, `float32`/`bool` dtypes, and exact-zero absent slots. |
| `RelativeStemPairer` | `pair(split_file, midi_root, hand_root) -> list[FilePair]`. |
| `EagerSongReader` | `read(pair, midi_loader, hand_loader) -> Song`; the loader exposes it as `.reader` for realtime. |
| `MidoNoteReader` | `read(path) -> tuple[Note,...]`. |
| `InitializerChordMerger` | `merge(event_sequence) -> tuple[Chord,...]`; every chord includes grouped events, selected action time, absolute-seconds `duration`, and a `float32[200]` `quantized_dur`. |
| `CanonicalNpzReader` | `read(path) -> HandSequence`. |
| `WiLoRCoordinateConverter` | `(raw_arrays, detection_index) -> float32[21,3]` canonical hand. |
| Hand transform | `(pose, presence) -> pose` with unchanged shape and absent slots zero. |
| Slicer | Produce real frame/chord slices and implement starts and candidate selection. |
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
- New ragged-candidate policy: replace `ragged_to_canonical.selector`; preserve
  its absolute-index contract.
- New temporal cleanup: replace `ragged_to_canonical.postprocessor`; preserve
  its array contract and use a distinct canonical root for changed settings.
- Earlier rather than later chord extension: replace the slicer while retaining
  all overlapping chords and the fixed-width overflow safeguard.
- Note alignment: use singleton chords plus a compatible slicer/event generator;
  the note sequence already exists and the model accepts pitch vectors.
- New target profile: replace the event generator and/or softener, retaining
  finite `[T,C]` BCE targets.
- New raw backend: replace `VideoToWiLoR` and canonicalizer targets together.
  Realtime automatically instantiates the same configured components.
- Model changes: provide a compatible Hydra `_target_` and version checkpoint
  compatibility. Loss changes remain deliberate edits in `airpiano/model/`.

Any new training target or inference field must be considered in visualization,
sonification, artifact tests, and documentation in the same change.

## Critical invariants and common traps

1. `frame_mask` and `chord_mask` are created only by batch collation. Presence
   is not a frame mask and never filters records, frames, targets, or BCE
   positions. Dataset windows are omitted only when they contain no real frame
   or chord candidate.
2. Hand slot order is `[Left, Right]`; the first 128 chord-vector entries always
   span MIDI pitches `0..127`.
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
11. Realtime-style roots contain synchronized `video/<stem>.mp4` and
    `midi/<stem>.mid` pairs; caches and stage outputs remain under that root.
12. Three complete split datasets live in host RAM simultaneously. Lowering
    batch size does not reduce eager dataset memory.
13. Training always starts online W&B project `air_piano` and has no
    disabled/offline mode.
14. Test and realtime inference default to `last.pt`; selecting `best.pt` is an
    explicit override.
15. The fixed project directory is not cleared or validated before training.
    Duplicate runs are unsupported and can mix overwritten and stale files.
16. Predictions have no manifest. Visualization and sonification batch direct
    NPZs and write sidecars in the same directory.
17. Raw chunk/manifest writes and ragged visualization videos are atomic;
    canonical, checkpoint, inference, JSON, heatmap, and MIDI outputs are
    direct writes and may be partial after an interruption.

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
| Missing split file | Generate PianoMotion lists with `scripts/convert_legacy_splits.py` or PianoVAM lists with `scripts/convert_pianovam_splits.py`, matching the selected config. |
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
