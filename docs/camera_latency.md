# Headless camera latency calibration

`scripts/capture_camera_latency.py` captures directly from the server's USB webcam.
Your PC displays a synchronized clock as the camera's target. The server needs
no display, GPU, model checkpoint, or audio output. This tool uses the existing
`v2n` OpenCV and NumPy dependencies; it does not modify streaming configuration.

## Run the clock test

1. Stop any other application using the webcam. On the server, from the repository:

   ```bash
   conda activate v2n
   python scripts/capture_camera_latency.py --camera 0 --duration 60
   ```

   If the current shell lacks the webcam's `video` group, use the existing group
   membership without changing device permissions:

   ```bash
   sg video -c '/home/jinxuan/.conda/envs/v2n/bin/python scripts/capture_camera_latency.py --camera 0 --duration 60'
   ```

2. In a terminal **on your PC**, forward the clock server's port. Replace
   `YOUR_SERVER` with the same host/IP you normally use for SSH:

   ```bash
   ssh -N -L 8766:127.0.0.1:8766 jinxuan@YOUR_SERVER
   ```

3. Open <http://localhost:8766> on that PC. The page estimates its clock offset
   using 20 timestamp exchanges and the smallest round trip, then updates the
   displayed timer locally. It refreshes the estimate while waiting, then holds
   the offset constant during warmup and capture so corrections cannot introduce
   artificial jumps. Network synchronization uncertainty is shown separately
   from display delay; clock drift during capture is an additional uncertainty.
   No system clocks are changed. Use one clock page per capture session.

4. Aim the server's webcam at the large timer on your PC screen. Make the digits
   readable and keep the page visible. Click **Start capture**. The server drains
   five seconds of warmup frames, then measures for sixty seconds. The default
   camera settings match streaming: 1920×1080, MJPEG, requested 30 fps, two driver
   buffers. Options include `--width 1280 --height 720` and `--warmup 10`.

5. Once capture finishes, click **Open the latest saved sample**, or inspect the
   output directory printed in the terminal through your usual remote file
   browser. The page stays available until Ctrl+C; the camera is already released.

## Read the results

Each session creates a new directory under `outputs/camera_latency/`:

| File | Contents |
| --- | --- |
| `samples/frame_*.jpg` | One full-resolution image per second, with arrival time in an added top margin |
| `frames.csv` | Every measured frame's read-completion timestamp, driver timestamp, read duration, and optional sample path |
| `summary.json` | Actual captured fps and average/median/p95/max driver-to-read timing |
| `clock_sync.jsonl` | Browser synchronization offset, round trip, and estimated uncertainty |
| `config.json` | Requested settings and the common server monotonic time origin |

For a saved image, subtract the clock **photographed within the image** from the
yellow `READ ARRIVAL` annotation. Both use the same server-relative timebase:

```text
READ ARRIVAL:       42.440 s
Clock in image:     42.350 s
Optical estimate:    0.090 s = 90 ms
```

Compare several images across the run, including early and late samples. This
optical estimate includes reference-screen rendering/refresh delay plus camera
exposure, processing, USB transfer, buffering, and decode up to Python's `read()`
completion. It does not include AirPiano inference, MIDI playback, or a remote
preview return trip. A 60 Hz screen updates every 16.7 ms; exposure blur, display
delay, and clock uncertainty prevent claiming 1 ms accuracy from the digits.
Unreadable or mixed digits cannot provide a reliable sample.

The script does **not** perform OCR: `optical_latency_ms` remains null in the
summary. Automatic `driver_to_read_ms` statistics are a different measurement.
They are reported only for a recognized UVC monotonic timestamp; unsupported or
invalid timestamps remain blank. These driver timestamps are not verified sensor
exposure timestamps. `read_call_ms` includes waiting for the next frame and must
not be interpreted as the age of that frame.

JPEG encoding runs on a separate thread with a bounded queue. If writing falls
behind, only diagnostic images are skipped; frame timing collection continues.
Ctrl+C during measurement preserves partial logs, images, and a summary.

## Capture without a browser

For a quick headless capture/driver-timing check that exits automatically:

```bash
python scripts/capture_camera_latency.py --capture-now --duration 10
```

This saves the same arrival-stamped images and timing logs. Without a synchronized
clock visible to the camera, it cannot measure the full optical delay. Use a new
output directory for every run; an existing `--output-dir` is never overwritten.
