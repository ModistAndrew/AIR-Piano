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

Update the five `data.*_root` paths in [configs/default.yaml](configs/default.yaml) to point to your input data or desired output directories.

Split files contain one relative record ID per line, without a file extension.

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

Training uses online Weights & Biases logging:

```bash
wandb login
python run.py stage=train device=cuda:0
```

## Inference

### Required files

- A trained AIR-Piano checkpoint.
- A score MIDI and matching recorded `.mp4` video.

Configure the same data roots as above. Missing hand caches are prepared from the video; missing beat files can be generated from MIDI metadata during inference.

The score MIDI does not need to be temporally aligned with the video for inference. In our AIR dataset, however, we establish this alignment through musical subdivision annotations to enable evaluation.

### Run inference

Create a split file such as `splits/custom.txt`, then run:

```bash
python run.py 'stage=[full_song_real_time,visualize,sonify]' \
  full_song_real_time.split_file=splits/custom.txt \
  full_song_real_time.checkpoint=outputs/airpiano/checkpoints/best.pt \
  device=cuda:0
```

## Evaluation

### Downloading AIR-PIANO and AIR-TABLETOP (WIP)

Dataset download links are **WIP**.

### Run evaluation

Run full-song inference followed by evaluation for each dataset. Replace the checkpoint path if needed:

```bash
python run.py 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/air_piano.txt \
  full_song_real_time.checkpoint=outputs/airpiano/checkpoints/best.pt \
  device=cuda:0

python run.py 'stage=[full_song_real_time,evaluation,visualize,sonify]' \
  full_song_real_time.split_file=splits/air_tabletop.txt \
  full_song_real_time.checkpoint=outputs/airpiano/checkpoints/best.pt \
  device=cuda:0
```

