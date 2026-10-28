"use strict";

const gallery = document.getElementById("demo-gallery");
const status = document.getElementById("gallery-status");
const retry = document.getElementById("retry-load");
const displayOrder = [
  "jinxuan-beethoven-8-3",
  "jinxuan-summer",
  "huiran-bach-prelude",
  "huiran-schubert-serenade",
  "venkat-minuet",
  "venkat-chopin-waltz",
];

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds)));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function createDemoCard(demo) {
  const article = document.createElement("article");
  article.className = "demo-card";
  article.setAttribute("aria-labelledby", `${demo.id}-title`);

  const frame = document.createElement("div");
  frame.className = "video-frame";

  const video = document.createElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "none";
  video.poster = demo.poster;
  video.src = demo.video;
  video.setAttribute("aria-label", `${demo.title}, performed by ${demo.performer}; predicted MIDI audio`);
  video.setAttribute("aria-describedby", "audio-note");
  video.append("Your browser cannot play this video. Use the video link below to open it.");
  frame.append(video);

  const error = document.createElement("div");
  error.className = "video-error";
  error.hidden = true;
  const errorText = document.createElement("p");
  errorText.textContent = "This video could not be loaded.";
  const videoLink = document.createElement("a");
  videoLink.href = demo.video;
  videoLink.textContent = "Open the MP4 video";
  error.append(errorText, videoLink);
  frame.append(error);
  video.addEventListener("error", () => { error.hidden = false; });
  video.addEventListener("loadeddata", () => { error.hidden = true; });

  const caption = document.createElement("div");
  caption.className = "demo-caption";
  const info = document.createElement("div");
  const title = document.createElement("h3");
  title.id = `${demo.id}-title`;
  title.textContent = demo.title;
  const meta = document.createElement("p");
  meta.className = "demo-meta";
  const performer = document.createElement("span");
  performer.textContent = demo.performer;
  const separator = document.createElement("span");
  separator.className = "meta-separator";
  separator.textContent = "·";
  separator.setAttribute("aria-hidden", "true");
  const duration = document.createElement("span");
  duration.textContent = formatDuration(demo.duration_seconds);
  meta.append(performer, separator, duration);
  info.append(title, meta);

  const midi = document.createElement("a");
  midi.className = "midi-download";
  midi.href = demo.midi;
  midi.download = `${demo.id}.predicted.mid`;
  midi.textContent = "Predicted MIDI ↓";
  midi.setAttribute("aria-label", `Download predicted MIDI for ${demo.title}, ${demo.performer}`);
  caption.append(info, midi);
  article.append(frame, caption);
  return article;
}

function validDemo(demo) {
  const localAsset = (value) => typeof value === "string"
    && value.startsWith("assets/demos/") && !value.includes("..");
  return demo && displayOrder.includes(demo.id)
    && typeof demo.title === "string" && typeof demo.performer === "string"
    && Number.isFinite(Number(demo.duration_seconds)) && Number(demo.duration_seconds) > 0
    && [demo.video, demo.poster, demo.midi].every(localAsset)
    && !demo.midi.endsWith(".gt.mid");
}

async function loadDemos() {
  retry.hidden = true;
  status.hidden = false;
  status.textContent = "Loading performances…";
  try {
    const response = await fetch("demos.json");
    if (!response.ok) throw new Error("Demo manifest unavailable");
    const data = await response.json();
    if (!Array.isArray(data.demos) || data.demos.length !== displayOrder.length
        || !data.demos.every(validDemo)
        || new Set(data.demos.map((demo) => demo.id)).size !== displayOrder.length) {
      throw new Error("Demo manifest is incomplete");
    }
    const byId = new Map(data.demos.map((demo) => [demo.id, demo]));
    gallery.replaceChildren(...displayOrder.map((id) => createDemoCard(byId.get(id))));
    status.hidden = true;
  } catch (error) {
    status.textContent = "The performances could not be loaded. Please try again.";
    retry.hidden = false;
  }
}

// Native controls retain standard keyboard/touch behavior. A new performance
// pauses the previous one without resetting its position.
gallery.addEventListener("play", (event) => {
  if (!(event.target instanceof HTMLVideoElement)) return;
  gallery.querySelectorAll("video").forEach((video) => {
    if (video !== event.target) video.pause();
  });
}, true);

retry.addEventListener("click", loadDemos);
loadDemos();
