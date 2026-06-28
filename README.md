# Gaze-Paced Reader — Feasibility Spike

A reading app that streams words **right-to-left**, where the **scroll speed is
paced by where you look**. Look toward the right end of the stream and it speeds
up; let your gaze fall back to the left and it slows down. The goal is to
auto-match scroll speed to your reading speed using only the front-facing camera.

This repo is a **feasibility spike** — the minimum needed to prove the core loop
works on a real Android phone before investing in polish.

## How it works

- **Camera:** `getUserMedia({ facingMode: "user" })` — front camera, fully
  client-side. Requires a secure origin (HTTPS or `localhost`).
- **Gaze:** [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  (`@mediapipe/tasks-vision`, loaded from CDN) runs in the browser at ~30 fps and
  gives 478 face landmarks including iris centers + a head-pose matrix.
- **Signal:** We don't need absolute "where on screen" gaze (too coarse on a phone).
  We compute a **relative horizontal signal** — iris offset within each eye, fused
  with head yaw, smoothed — then normalize it with a quick two-point calibration.
- **Control:** A closed loop integrates the gaze error into words-per-minute, so the
  speed self-adjusts until your gaze re-centers on the focus zone. Per-user bias
  cancels out; the loop is forgiving of noisy gaze.

See [`app.js`](app.js) for the whole pipeline (it's small and commented).

## Run it on your phone

`getUserMedia` needs HTTPS when served to a phone over the LAN. Two options:

### Option A — self-signed HTTPS (built-in `serve.js`, no deps)

```bash
# one-time: generate a self-signed cert into ./dev-cert/
mkdir -p dev-cert
openssl req -x509 -newkey rsa:2048 -nodes -keyout dev-cert/key.pem \
  -out dev-cert/cert.pem -days 365 -subj "/CN=gaze-reader.local"

node serve.js          # serves https://<this-machine>:8443
```

Find your machine's LAN IP (`ip addr` / `ifconfig`) and open
`https://<LAN-ip>:8443` in Chrome on the phone. Accept the
self-signed-certificate warning ("Advanced → proceed").

### Option B — a tunnel

Serve the folder any way you like and expose it with an HTTPS tunnel, e.g.
`npx http-server -p 8000` then `npx localtunnel --port 8000` (or ngrok/cloudflared).
Open the tunnel's `https://…` URL on the phone.

## Using the app

1. **Start camera** — grant permission. The model loads from CDN (first load needs network).
2. **Calibrate** — look at the left dot, then the right dot when prompted (~1.5 s total).
3. **Run** — a word stream scrolls right-to-left. The orange marker shows where your
   gaze maps along the strip.
4. **Debug** — toggles a live overlay (iris, yaw, smoothed/normalized gaze, error, WPM, fps)
   and a small mirrored camera preview. Use this to judge signal quality.

### What "success" looks like (the things we're testing)

- Deliberately staring at the **right edge** visibly **speeds up** the stream;
  staring **left** slows it down.
- Reading normally for a minute, the speed **self-tunes** to a comfortable pace
  and stays **stable** (no oscillation or runaway).
- Note in [`app.js`](app.js) `CFG` if you need to tune: `kp` (responsiveness),
  `emaAlpha` (smoothing), `targetGaze` (focus-zone center), WPM bounds.

Please record how it behaves **with/without glasses** and in **different lighting** —
that tells us whether to invest in the fallback (detecting the natural left→right
reading-sweep cadence) or move on to building the full app.

## Status / scope

Spike only. Out of scope for now: text import/library, typography polish, settings
persistence, offline caching of the model, accessibility, and any native
(Capacitor) wrapper. Those come after the gaze→speed loop is proven on-device.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Page skeleton: video, word stream, controls, overlays |
| `app.js` | Camera, MediaPipe, gaze signal, calibration, controller, render loop |
| `styles.css` | Layout + scrolling animation + overlays |
| `serve.js` | Zero-dep HTTPS static dev server |
| `manifest.webmanifest`, `sw.js`, `icon.svg` | PWA install bits |
