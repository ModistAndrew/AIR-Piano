"use strict";

// Videos remain native, keyboard-accessible controls. Never autoplay on load.
document.addEventListener("play", (event) => {
  if (!(event.target instanceof HTMLVideoElement)) return;
  document.querySelectorAll("video").forEach((video) => {
    if (video !== event.target && !video.paused) video.pause();
  });
}, true);

document.querySelectorAll("[data-live-card]").forEach((card) => {
  const video = card.querySelector("video");
  const buttons = [...card.querySelectorAll("[data-view]")];
  buttons.forEach((button) => button.addEventListener("click", () => {
    if (button.getAttribute("aria-pressed") === "true") return;
    video.pause();
    buttons.forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
    // Independent recording starts: changing view deliberately starts that file at zero.
    video.src = button.dataset.src;
    video.poster = button.dataset.poster;
    video.load();
    card.querySelector("[data-duration]").textContent = button.dataset.duration;
    const camera = button.dataset.view === "camera";
    card.querySelector("[data-audio-label]").textContent = camera ? "Synthesized live MIDI replay" : "Original recorded sound";
    card.querySelector("[data-view-description]").textContent = camera
      ? "The system-camera recording with audio synthesized from the MIDI saved during this live session."
      : "The live performance with its original audio and timing.";
    video.setAttribute("aria-label", `${card.querySelector("h3").textContent}: ${camera ? "camera view with MIDI replay" : "original live recording"}`);
  }));
});

const filters = [...document.querySelectorAll("[data-filter]")];
const examples = [...document.querySelectorAll("[data-example]")];
filters.forEach((button) => button.addEventListener("click", () => {
  filters.forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
  let visible = 0;
  examples.forEach((card) => {
    const show = button.dataset.filter === "all" || card.dataset.group === button.dataset.filter;
    if (!show) card.querySelector("video").pause();
    card.hidden = !show;
    if (show) visible += 1;
  });
  document.querySelector("#gallery-status").textContent = `Showing ${visible} recordings`;
}));

// Avoid leaving background audio playing after switching away from the page.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) document.querySelectorAll("video").forEach((video) => video.pause());
});
