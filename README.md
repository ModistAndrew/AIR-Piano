# AIR-Piano

A system that turns tabletop piano gestures into sound using score-guided visual alignment.

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

Update the five `data.*_root` paths in [configs/ours.yaml](configs/ours.yaml), or your selected ablation config, to point to your input data or desired output directories.

Split files contain one relative record ID per line, without a file extension.

### Check MIDI split similarity

Use `scripts/check_midi_similarity.py` to check two split files for exact or composition-level MIDI overlap:

```bash
python scripts/check_midi_similarity.py splits/default_train.txt splits/default_valid.txt
```

## Dependencies

The project uses Python 3.11. FFmpeg/ffprobe are required. See [requirements.txt](requirements.txt) for dependencies (a fresh installation has not been tested).

### Files to download

For hand extraction from videos, download these assets to the paths expected by the default configuration:

| Asset | Destination |
| --- | --- |
| [WiLoR weights](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt) | `assets/wilor/wilor_final.ckpt` |
| [Hand detector weights](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) | `assets/wilor/detector.pt` |
| [MANO right-hand model](https://mano.is.tue.mpg.de/) | `assets/mano/MANO_RIGHT.pkl` |

MANO requires registration and agreement to its license. The WiLoR code, model configuration, and MANO mean parameters are included under [third_party/WiLoR/](third_party/WiLoR/).

## Training

### Preparing the dataset

The default splits use **PianoMotion10M** and **PianoYT**. You may download the source video and audio according to the listed IDs, following the data folder structure.

Training requires aligned MIDI, beat files, and canonical hand caches. Generate MIDI with TransKun, beats with Beat This!, and raw hand caches with WiLoR, then convert the hand caches to canonical format. Match the paths below to `data.*_root` in your configuration:

```bash
python scripts/preprocess_transkun.py /path/to/data/Audio /path/to/data/MIDI --device cuda:0
python scripts/preprocess_beats.py /path/to/data/Audio /path/to/data/beats --device cuda:0
python scripts/preprocess_wilor.py /path/to/data/Video /path/to/data/wilor --device cuda:0
python run.py stage=preprocess preprocess.task=ragged_to_canonical device=cuda:0
```

### Generate PianoMotion10M duration variants

Generate **PianoMotion10M30s**, **PianoMotion10M60s**, and **PianoMotion10M120s** video and audio from default_valid data using the following script: 

```bash
python scripts/preprocess_pianomotion_durations.py /path/to/data splits/default_valid.txt
```

### Run training

The default configuration is [configs/ours.yaml](configs/ours.yaml). It trains for 20 epochs with seed 42, using onset targets advanced by 100 ms (3 frames at 30 fps) and a triangular target radius of 0.2 s. Pitch augmentation is disabled. Motion-window start and end boundaries are independently perturbed by up to 0.5 s, and score IOI features share one scaling factor per segment sampled uniformly from `{0.5, 1, 2}`.

Training uses online Weights & Biases logging:

```bash
wandb login
python run.py --config-name ours stage=train device=cuda:0
```

### Ablations

All seven configurations in [configs/](configs/) are standalone: `ours` and the six ablations below. They do not inherit from one another; update data paths in each configuration you use. Apart from the run name and representation-specific cache settings, each ablation changes only the indicated settings.

```bash
# Use centered absolute XY for non-wrist joints; retain relative Z and wrist displacement.
python run.py --config-name ablation_absolute_joint_xy stage=train device=cuda:0

# Set the wrist slot to zero.
python run.py --config-name ablation_no_wrist_displacement stage=train device=cuda:0

# Disable random motion-window boundary perturbations.
python run.py --config-name ablation_no_hand_window_jitter stage=train device=cuda:0

# Disable the IOI feature input; retain the augmentation settings.
python run.py --config-name ablation_no_ioi_feature stage=train device=cuda:0

# Disable IOI scaling; retain the IOI feature input.
python run.py --config-name ablation_no_ioi_scaling stage=train device=cuda:0

# Use a 0.4 s triangular target radius with no target shift.
python run.py --config-name ablation_radius400ms_no_target_shift stage=train device=cuda:0
```

## Inference

### Required files

- A trained AIR-Piano checkpoint.
- A score MIDI and matching recorded `.mp4` video.

Configure the same data roots as above. Missing hand caches are prepared from the video; missing beat files can be generated from MIDI metadata during inference.

The score MIDI does not need to be temporally aligned with the video for inference. In our AIR dataset, however, we establish this alignment through musical subdivision annotations to enable evaluation.

### Run inference

Create a split file such as `splits/custom.txt` and run the command below. Inference defaults to `outputs/<config-name>/checkpoints/last.pt`; the explicit path below matches `ours`. Existing experiment checkpoints remain at their original paths and can be selected with `full_song_real_time.checkpoint`.

```bash
python run.py --config-name ours 'stage=[full_song_real_time,visualize,sonify]' \
  full_song_real_time.split_file=splits/custom.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0
```

The controller uses an onset offset of `radius / 2 - delay`, which is 0.2 s for `ours`, and a simulated processing latency of 50 ms. The score-skip penalty is configurable via `full_song_real_time.score_skip_penalty` and defaults to `1.0`.

## Evaluation

### Downloading AIR-PIANO and AIR-TABLETOP (WIP)

Dataset download links are **WIP**.

### Run evaluation

Run full-song inference followed by evaluation for each dataset. Replace the checkpoint path if needed:

```bash
python run.py --config-name ours 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/air_piano.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0

python run.py --config-name ours 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/air_tabletop.txt \
  full_song_real_time.checkpoint=outputs/ours/checkpoints/last.pt \
  device=cuda:0
```

The Piano and Tabletop splits contain 20 and 40 recordings, respectively. `splits/air_all.txt` combines all 60, and `splits/eval_all.txt` adds the 90 PM60 recordings for a total of 150. Evaluation uses physical reference onsets (`physical_target`), without an extra reference shift by default (`evaluation.gt_shift_frames=0`), and reports C-AR at tolerances of 2, 4, 8, and 16 frames. Reports include both pooled micro averages and recording-level macro averages; paper results use the latter to weight each performance equally.
