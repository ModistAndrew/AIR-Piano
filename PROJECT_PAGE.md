# AIRPiano demo website

The static website in `docs/` is the anonymous review demo. It includes two live performances, each with the original recording and a camera view with MIDI replay, followed by eight evaluation examples.

## Playback and timing

Original live videos retain their recorded audio. Camera replays and evaluation examples use sound synthesized from saved prediction MIDI. The two views of each live demonstration have independent start times. Evaluation examples retain leading silence and failure periods; they use the original, unshifted reference timings for the reported scores.

## Hosting

GitHub Pages serves `main` and `/docs`. No frontend build is required. Keep `docs/.nojekyll` and relative asset paths intact. The complete site is approximately 149 MB, with every video under 24 MB. Videos are loaded on demand.

For local preview, run `node scripts/serve_project_docs.mjs` from the repository root and open the address it prints.

Only the website directory is intended as the public demo package. Source recordings, processing logs, and local audit files are not included. Check the anonymous mirror's links and video playback before using its URL in the paper.