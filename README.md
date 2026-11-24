# AIR-Piano

[![arXiv](https://img.shields.io/badge/arXiv-2611.30049-b31b1b?logo=arxiv)](https://arxiv.org/abs/2611.30049) [![Hugging Face checkpoint](https://img.shields.io/badge/Hugging%20Face-Checkpoint-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/ModistAndrew/AIR-Piano)

A system that turns tabletop piano gestures into sound using score-guided visual alignment.

## Branches

- `demo-review`: demo interface.
- `streaming_test`: streaming test; code hasn't been cleaned up.

Other branches contain some legacy experiments.

## Dependencies

The project uses Python 3.11. FFmpeg/ffprobe are required. See [requirements.txt](requirements.txt) for dependencies (a fresh installation has not been tested).

### Files to download

Download the pretrained AIR-Piano checkpoint for inference and the following assets for hand extraction to the paths expected by the default configuration:

| Asset | Destination |
| --- | --- |
| [AIR-Piano checkpoint](https://huggingface.co/ModistAndrew/AIR-Piano/resolve/main/last.pt) | `outputs/ours/checkpoints/last.pt` |
| [WiLoR weights](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt) | `assets/wilor/wilor_final.ckpt` |
| [Hand detector weights](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) | `assets/wilor/detector.pt` |
| [MANO right-hand model](https://mano.is.tue.mpg.de/) | `assets/mano/MANO_RIGHT.pkl` |

MANO requires registration and agreement to its license. The WiLoR code, model configuration, and MANO mean parameters are included under [third_party/WiLoR/](third_party/WiLoR/).

## Data folder structure

Run commands from the repository root. Each record uses the same relative path across modalities using the roots specified in the configuration.

For example, the record ID `collection/song` maps to:

```text
<data.video_root>/collection/song.mp4
<data.midi_root>/collection/song.mid
<data.beat_root>/collection/song.txt
<data.raw_wilor_root>/collection/song/
<data.hand_root>/collection/song.npz
```

Audio, such as `/path/to/Audio/collection/song.m4a`, is only required by the preprocessing scripts.

Update the five `data.*_root` paths in [configs/ours.yaml](configs/ours.yaml) to point to your input data or desired output directories.

Split files contain one relative record ID per line, without a file extension.

## Training

### Preparing the dataset

The training and validation splits (`default_train.txt` and `default_valid.txt`) use **PianoMotion10M** and **PianoYT**. You may download the source video and audio according to the listed IDs, following the data folder structure. The default test split (`vsf_all.txt`) contains our VSF datasets; see [here](#download-vsf-piano-and-vsf-tabletop).

Training requires aligned MIDI, beat files, and canonical hand caches. Generate MIDI with TransKun, beats with Beat This!, and raw hand caches with WiLoR, then convert the hand caches to canonical format. Match the paths below to `data.*_root` in your configuration:

```bash
python scripts/preprocess_transkun.py /path/to/data/Audio /path/to/data/MIDI --device cuda:0
python scripts/preprocess_beats.py /path/to/data/Audio /path/to/data/beats --device cuda:0
python scripts/preprocess_wilor.py /path/to/data/Video /path/to/data/wilor --device cuda:0
python run.py stage=preprocess preprocess.task=ragged_to_canonical device=cuda:0
```

### Run training

Training uses online Weights & Biases logging:

```bash
wandb login
python run.py --config-name ours stage=train device=cuda:0
```

### Ablations

All ablation configs inherit [configs/ours.yaml](configs/ours.yaml), changing only the indicated settings and the run name.

```bash
# Absolute joint XY: center camera-space XY for non-wrist joints; retain wrist-relative Z and wrist displacement.
# Fit the wrist XY mean on training data only, build canonical absolute-XY caches for all three splits, then train.
python run.py --config-name ablation_absolute_joint_xy 'stage=[fit_hand_mean,preprocess,train]' device=cuda:0

# Zero the wrist displacement slot; retain wrist-relative joint coordinates.
python run.py --config-name ablation_no_wrist_displacement stage=train device=cuda:0

# Disable random motion-window boundary perturbations.
python run.py --config-name ablation_no_hand_window_jitter stage=train device=cuda:0

# Disable the IOI feature input; retain the augmentation settings.
python run.py --config-name ablation_no_ioi_feature stage=train device=cuda:0

# Disable IOI scaling; retain the IOI feature input.
python run.py --config-name ablation_no_ioi_scaling stage=train device=cuda:0

# Use a 0.4 s triangular target radius with no target shift; the controller onset offset remains 0.2 s.
python run.py --config-name ablation_radius400ms_no_target_shift stage=train device=cuda:0
```

## Inference

### Required files

- A trained AIR-Piano checkpoint.
- A score MIDI and matching recorded `.mp4` video.

Configure the same data roots as above. Missing hand caches are prepared from the video; missing beat files can be generated from MIDI metadata during inference.

The score MIDI does not need to be temporally aligned with the video for inference. In our VSF dataset, however, we establish this alignment through musical subdivision annotations to enable evaluation.

### Run inference

Create a split file such as `splits/custom.txt`, then run:

```bash
python run.py 'stage=[full_song_real_time,visualize,sonify]' \
  full_song_real_time.split_file=splits/custom.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0
```

## Evaluation

### Download VSF-Piano and VSF-Tabletop

Download [VSF-Piano and VSF-Tabletop from Hugging Face](https://huggingface.co/datasets/ModistAndrew/VSF). The release includes videos, aligned score MIDI, and subdivision annotations. Hand caches are generated automatically during full-song inference.

### Run evaluation

Run full-song inference followed by evaluation for each dataset. Replace the checkpoint path if needed:

```bash
python run.py 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/vsf_piano.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0

python run.py 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/vsf_tabletop.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0
```

## Reproducibility

Use these scripts to reproduce duration subsets, split checks, and latency measurements. Replace input paths as needed and use a new output directory for each benchmark run.

### Generate PianoMotion10M duration variants

Use `scripts/preprocess_pianomotion_durations.py` to create deterministic 30 s, 60 s, and 120 s video/audio clips from the PianoMotion10M validation recordings. Run MIDI, beat, and hand preprocessing on the resulting clips:

```bash
python scripts/preprocess_pianomotion_durations.py /path/to/data splits/default_valid.txt
```

### Check MIDI split similarity

Use `scripts/check_midi_similarity.py` to check two split files for exact or composition-level MIDI overlap:

```bash
python scripts/check_midi_similarity.py splits/default_train.txt splits/default_valid.txt --midi-root /path/to/data/MIDI
```

### Benchmark alignment latency

Use `scripts/benchmark_realtime_latency.py` to measure frame-level alignment and controller latency using the configured checkpoint, MIDI, timing grids, and canonical hand caches; hand extraction is excluded. Set the data roots in `configs/ours.yaml` to match the selected record:

```bash
python scripts/benchmark_realtime_latency.py VSF-Piano/P1/bach_prelude --config configs/ours.yaml --device cuda:0 --output-dir outputs/latency/ours
```

### Benchmark WiLoR latency

Use `scripts/benchmark_wilor_latency.py` to measure single-frame WiLoR extraction latency on preloaded video frames, excluding video decoding:

```bash
python scripts/benchmark_wilor_latency.py /path/to/data/Video/VSF-Piano/P1/bach_prelude.mp4 --device cuda:0 --mode single_frame --output-dir outputs/latency/wilor
```
