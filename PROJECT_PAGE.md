# AIRPiano project page

The static project page lives in `docs/`. The model and training code are unchanged.

## Local preview

With Node.js installed, run from the repository root:

```sh
node scripts/serve_project_docs.mjs
```

Open <http://127.0.0.1:8796/AIR-Piano/>. The preview uses the same URL prefix as
GitHub Pages and supports byte-range requests for video seeking. No build step
or frontend package installation is required.

## Demo sources

Six complete recordings are paired with their same-stem **prediction** MIDIs:

- Jinxuan: `beethoven_8_3`, `summer`
- Huiran: `bach_prelude`, `schubert_serenade`
- Venkat: `minuet`, `chopin_waltz`

Videos come from `REAL/<performer>_tabletop_aligned/`. Prediction MIDIs come from
`REAL/airpiano_air_tabletop/AIR/<performer>_tabletop_aligned/`.
Ground-truth `*.gt.mid` files are excluded. The original recording soundtracks
are replaced with piano audio synthesized from the prediction MIDIs.

The MIDI and aligned video share time zero. Leading silence, event timing,
note duration and velocity are preserved; saved scheduling latency is not
added a second time. Where needed, the last video frame is held until one
second after the final MIDI note-off. The downloadable MIDIs are unchanged
copies of the prediction files.

`docs/demos.json` lists the public media URLs and recording labels. Media can
be regenerated with `scripts/build_project_demos.py`; run it with `--help`
for its source/tool options. Local source files, soundfonts and renderer
dependencies are not part of the website.

## Publish after preview approval

1. Review the local page and the six videos.
2. Commit and push `docs/`, the project-page scripts and this guide to `main`.
3. In the GitHub repository, open **Settings → Pages**.
4. Choose **Deploy from a branch**, branch **main**, folder **/docs**, and save.
5. Wait for the Pages deployment, then verify
   <https://modistandrew.github.io/AIR-Piano/> and its media URLs.

Do not publish until the local preview has been approved. `docs/.nojekyll`
allows the files to be served without a Jekyll build.

## Poster QR codes

After the project page is online and verified, create two black-on-white QR
codes for the project URL above and <https://github.com/ModistAndrew/AIR-Piano>.
Keep the full four-module quiet zone, export SVG and high-resolution PNG,
decode-check both URLs, and replace the two existing poster placeholders.
The current preview does not claim that Pages is already online.
