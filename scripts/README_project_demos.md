# Building the project-page videos

`build_project_demos.py` produces six **complete** tabletop-performance demos.
The only soundtrack input is the saved prediction `.mid`; `*.gt.mid` and the
recording's original audio are never used as soundtrack inputs.

## Requirements

- Python with `mido`, `numpy` and `scipy` (initial build: 1.3.3, 1.26.4 and 1.17.1 respectively)
- FFmpeg and FFprobe
- FluidSynth 2.6.1, including its runtime libraries
- A local piano SoundFont (the initial build uses the installed MuseScore
  `MS Basic.sf3`; the SoundFont is not copied into the website)

The initial portable FluidSynth archive came from the official release:

<https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1>

Archive: `fluidsynth-v2.6.1-win10-x64-cpp11.zip`

SHA-256: `fab7a2e4b85675b66970f97a39bbc239729c5e0f237198b5922a6a73cbc8677c`

## Command

From the repository root:

```text
python scripts/build_project_demos.py --workspace PATH_TO_AIRPIANO_DATA --fluidsynth PATH_TO_FLUIDSYNTH --soundfont PATH_TO_SOUNDFONT --ffmpeg PATH_TO_FFMPEG --ffprobe PATH_TO_FFPROBE
```

`PATH_TO_AIRPIANO_DATA` contains `REAL/`. The builder expects the paired aligned
videos in `REAL/<person>_tabletop_aligned/` and the predictions and NPZ artifacts
in `REAL/airpiano_air_tabletop/AIR/<person>_tabletop_aligned/`.

Outputs:

- `docs/assets/demos/`: H.264/AAC MP4s, JPEG posters and unchanged prediction MIDI files
- `docs/demos.json`: relative media URLs and display metadata
- Data workspace `.cache/website-media/`: temporary synthesis WAVs, process logs and reuse metadata
- Data workspace `output/website-media/build_report.json`: source hashes, exact event validation, encoding details and audio timing checks

The builder preserves MIDI silence, note events, velocities and timing. It uses
FluidSynth's sample clock, a common synthesis gain of `1.0`, and no chorus or
reverb. Videos begin at zero; no extra inference latency is added. The original
video timeline is retained, with a held final frame only if needed to reach
`max(video duration, last note-off + 1 second)`.

Before encoding, every saved MIDI note event is checked against the source NPZ
release times and pitches. After encoding, the decoded AAC is correlated with
the synthesis WAV near the first, middle and last predicted onsets. A mismatch,
unexpected start time, clipping risk or oversized output causes the build to
fail. Source files are never overwritten.

The default uses two workers. Pass `--jobs 1` for lower CPU usage or `--force`
to rebuild existing media. This script does not publish the site.
