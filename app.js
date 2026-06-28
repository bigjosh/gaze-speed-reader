// Gaze-Paced Reader — feasibility spike
// Proves the core loop: front camera -> MediaPipe Face Landmarker -> horizontal
// gaze signal -> closed-loop controller -> right-to-left word-stream speed.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- MediaPipe 478-landmark indices (with iris refinement) ----
// Right eye = image-left side; Left eye = image-right side.
const IDX = {
  rightEyeOuter: 33,
  rightEyeInner: 133,
  leftEyeInner: 362,
  leftEyeOuter: 263,
  rightIris: 468, // iris center
  leftIris: 473,
};

// ---- Tunables ----
const CFG = {
  emaAlpha: 0.25,        // gaze smoothing (higher = snappier, noisier)
  targetGaze: -0.42,     // gaze value when looking at the focus zone (~29% screen)
  kp: 350,               // WPM change per unit gaze-error per second (integral-on-speed)
  wpmMin: 60,
  wpmMax: 1100,
  wpmStart: 250,
  avgCharsPerWord: 5.5,  // for px/sec estimate fallback
  detectHz: 24,          // gaze-inference rate; kept off the scroll path entirely
  // --- Scroll engine (compositor-driven, see startStripAnimation) ---
  animSpanSec: 30,       // length of the single linear animation before we re-baseline
  controlIntervalMs: 120,// how often the gaze controller updates speed + tops up words
  recycleIntervalMs: 1500,// how often we re-baseline (prune off-screen words); rare seam
  prefillSec: 3,         // seconds of words kept ready to the right (main-thread-stall margin)
};

const SAMPLE_TEXT = (
  "The quick brown fox jumps over the lazy dog while the morning sun climbs " +
  "slowly above the quiet hills and a gentle wind carries the smell of rain " +
  "across the open field where children once played among the tall grass and " +
  "forgotten stones that mark the edges of an older world now mostly silent " +
  "except for the steady rhythm of words moving past your eyes one after another"
).split(/\s+/);

// ---- State ----
const state = {
  landmarker: null,
  running: false,
  lastVideoTime: -1,
  gazeRaw: 0,        // smoothed raw signal (iris + yaw), arbitrary scale
  gazeNorm: 0,       // normalized to [-1, 1] via calibration
  calib: { min: -0.3, max: 0.3 }, // raw values at left / right calibration
  wpm: CFG.wpmStart,
  irisRatio: 0,
  headYaw: 0,
  fps: 0,             // gaze-inference rate (Hz)
  rfps: 0,            // render rate (Hz)
  lastFrame: 0,
  lastDetect: 0,
  lastRenderFrame: 0,
  faceSeen: false,
};

// ---- DOM ----
const el = {
  cam: document.getElementById("cam"),
  stream: document.getElementById("stream"),
  marker: document.getElementById("gaze-marker"),
  debug: document.getElementById("debug"),
  status: document.getElementById("status"),
  start: document.getElementById("btn-start"),
  calibrate: document.getElementById("btn-calibrate"),
  run: document.getElementById("btn-run"),
  debugBtn: document.getElementById("btn-debug"),
  calibOverlay: document.getElementById("calib-overlay"),
  calibTarget: document.getElementById("calib-target"),
  calibText: document.getElementById("calib-text"),
};

function setStatus(msg) { el.status.textContent = msg; }

// ---------------------------------------------------------------------------
// Camera + MediaPipe init
// ---------------------------------------------------------------------------
async function startCamera() {
  el.start.disabled = true;
  setStatus("Requesting camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    el.cam.srcObject = stream;
    await el.cam.play();
  } catch (err) {
    setStatus("Camera failed: " + err.message + " (needs HTTPS + permission)");
    el.start.disabled = false;
    return;
  }

  setStatus("Loading face model…");
  try {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    state.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    });
  } catch (err) {
    setStatus("Model load failed: " + err.message);
    el.start.disabled = false;
    return;
  }

  setStatus("Tracking. Calibrate for best results, then Run.");
  el.calibrate.disabled = false;
  el.run.disabled = false;
  requestAnimationFrame(detectLoop);
}

// ---------------------------------------------------------------------------
// Per-frame gaze detection
// ---------------------------------------------------------------------------
function detectLoop(now) {
  // Main-thread frame cadence (every rAF). The visible scroll runs on the
  // compositor, so this number can dip during inference without the scroll
  // stuttering — it's here as a health signal, not the scroll's frame rate.
  if (state.lastRenderFrame) {
    const rdt = (now - state.lastRenderFrame) / 1000;
    if (rdt > 0) state.rfps = state.rfps * 0.9 + (1 / rdt) * 0.1;
  }
  state.lastRenderFrame = now;

  if (state.landmarker && el.cam.readyState >= 2) {
    // Throttle the heavy inference call so it doesn't run on every paint frame.
    // Rendering keeps its own full-rate rAF loop; the gaze controller only needs
    // a slow control signal, so running inference at ~detectHz frees the main
    // thread to paint smoothly between detections.
    const due = now - state.lastDetect >= 1000 / CFG.detectHz;
    if (due && el.cam.currentTime !== state.lastVideoTime) {
      state.lastVideoTime = el.cam.currentTime;
      const res = state.landmarker.detectForVideo(el.cam, now);
      processResult(res);
      // inference fps (measured only on frames we actually ran inference)
      if (state.lastDetect) {
        const dt = (now - state.lastDetect) / 1000;
        state.fps = state.fps * 0.9 + (1 / dt) * 0.1;
      }
      state.lastDetect = now;
    }
  }
  updateDebug();
  requestAnimationFrame(detectLoop);
}

function processResult(res) {
  const faces = res.faceLandmarks;
  state.faceSeen = faces && faces.length > 0;
  if (!state.faceSeen) return;
  const lm = faces[0];

  // Horizontal iris position within each eye, averaged. ~0 center, +/- to sides.
  const rRatio = eyeRatio(lm, IDX.rightEyeOuter, IDX.rightEyeInner, IDX.rightIris);
  const lRatio = eyeRatio(lm, IDX.leftEyeInner, IDX.leftEyeOuter, IDX.leftIris);
  state.irisRatio = (rRatio + lRatio) / 2;

  // Head yaw from facial transformation matrix (radians-ish), as secondary cue.
  state.headYaw = extractYaw(res.facialTransformationMatrixes);

  // Combined raw signal. Calibration normalizes scale/sign, so weights are loose.
  const raw = state.irisRatio + 0.6 * state.headYaw;
  state.gazeRaw = state.gazeRaw + CFG.emaAlpha * (raw - state.gazeRaw);

  // Normalize into [-1, 1] using calibration endpoints.
  const span = state.calib.max - state.calib.min || 1;
  state.gazeNorm = clamp((2 * (state.gazeRaw - state.calib.min)) / span - 1, -1.2, 1.2);
}

// Fraction of iris position between two eye corners, centered around 0.
function eyeRatio(lm, aIdx, bIdx, irisIdx) {
  const a = lm[aIdx], b = lm[bIdx], iris = lm[irisIdx];
  const t = (iris.x - a.x) / (b.x - a.x); // 0 at corner a, 1 at corner b
  return t - 0.5; // center
}

function extractYaw(matrices) {
  if (!matrices || !matrices.length) return 0;
  const m = matrices[0].data; // column-major 4x4
  // yaw ~ rotation about vertical axis
  return Math.atan2(m[8], m[10]);
}

// ---------------------------------------------------------------------------
// Calibration: capture raw gaze while user looks at left edge, then right edge.
// ---------------------------------------------------------------------------
async function calibrate() {
  el.run.disabled = true;
  el.calibrate.disabled = true;
  el.calibOverlay.classList.remove("hidden");

  const left = await capturePoint("8%", "Look at the dot (left)…");
  const right = await capturePoint("92%", "Look at the dot (right)…");

  state.calib.min = Math.min(left, right);
  state.calib.max = Math.max(left, right);
  // Remember sign orientation so "look right" => higher gazeNorm
  if (right < left) {
    // signal decreases to the right; flip by swapping handled above + invert later
  }
  el.calibOverlay.classList.add("hidden");
  el.run.disabled = false;
  el.calibrate.disabled = false;
  setStatus(`Calibrated. range=[${left.toFixed(3)}, ${right.toFixed(3)}]`);
}

function capturePoint(leftPct, label) {
  return new Promise((resolve) => {
    el.calibTarget.style.left = leftPct;
    el.calibText.textContent = label;
    // Settle, then average raw signal over a short window.
    setTimeout(() => {
      const samples = [];
      const t0 = performance.now();
      const tick = () => {
        samples.push(state.gazeRaw);
        if (performance.now() - t0 < 700) {
          requestAnimationFrame(tick);
        } else {
          resolve(samples.reduce((s, v) => s + v, 0) / samples.length);
        }
      };
      requestAnimationFrame(tick);
    }, 600);
  });
}

// ---------------------------------------------------------------------------
// Word stream + closed-loop controller
// ---------------------------------------------------------------------------
let words = [];        // [{ el, width }]
let offset = 0;        // strip translateX at the current animation's start (px)
let nextWord = 0;      // index into SAMPLE_TEXT (loops)
let avgWordPx = 0;
const animV0 = 1000;   // base animation velocity (px/sec); real speed = V0 * playbackRate
let streamAnim = null; // compositor-driven Web Animation translating the strip
let lastControl = 0;   // last time the controller/refill ran
let lastRecycle = 0;   // last time we re-baselined (pruned off-screen words)

function addWord() {
  const w = document.createElement("span");
  w.className = "word";
  w.textContent = SAMPLE_TEXT[nextWord % SAMPLE_TEXT.length];
  nextWord++;
  el.stream.appendChild(w);
  const width = w.getBoundingClientRect().width;
  words.push({ el: w, width });
  return width;
}

function contentWidth() {
  let s = 0;
  for (const w of words) s += w.width;
  return s;
}

// Current translateX of the strip, derived from the running animation's own clock
// (which advances at V0 * playbackRate). Stays correct regardless of how
// playbackRate has been changed over the animation's life.
function currentOffset() {
  if (!streamAnim) return offset;
  const ct = streamAnim.currentTime || 0; // ms in the animation's timeline
  return offset - animV0 * (ct / 1000);
}

// (Re)start the single long linear animation from `offset`, running on the GPU
// compositor. We keep ONE animation alive and set speed via playbackRate, so the
// visible motion never touches the main thread — face inference can hitch the main
// thread without the scroll stuttering. Restarted only when we re-baseline.
function startStripAnimation(playbackRate) {
  const D = CFG.animSpanSec;
  const from = offset;
  const to = offset - animV0 * D;
  el.stream.style.transform = `translate3d(${from}px,0,0)`;
  if (streamAnim) streamAnim.cancel();
  streamAnim = el.stream.animate(
    [{ transform: `translate3d(${from}px,0,0)` },
     { transform: `translate3d(${to}px,0,0)` }],
    { duration: D * 1000, easing: "linear", fill: "forwards" }
  );
  streamAnim.playbackRate = playbackRate;
}

function startRun() {
  state.running = true;
  document.body.classList.add("running");
  el.run.textContent = "Pause";
  if (!words.length) offset = window.innerWidth; // first run: seed off the right edge
  const t = performance.now();
  lastControl = t - CFG.controlIntervalMs;       // run the controller immediately
  lastRecycle = t;
  startStripAnimation(0);                         // start parked; controller sets speed
  requestAnimationFrame(controlLoop);
}

// Decoupled from rendering. Updates speed (via playbackRate), tops up words on the
// right, and on a slow cadence prunes/re-baselines. Jank here (e.g. an inference
// spike landing on the same frame) never stutters the scroll — that's the whole point.
function controlLoop(now) {
  if (!state.running) return;
  requestAnimationFrame(controlLoop);
  if (now - lastControl < CFG.controlIntervalMs) return;
  const dt = (now - lastControl) / 1000;
  lastControl = now;

  // --- Controller: integrate gaze error into speed (drives gaze back to target).
  const error = state.gazeNorm - CFG.targetGaze; // + = looking ahead/right
  if (state.faceSeen) {
    state.wpm = clamp(state.wpm + CFG.kp * error * dt, CFG.wpmMin, CFG.wpmMax);
  }

  // --- Convert WPM to px/sec using measured average word width.
  if (avgWordPx === 0 && words.length) avgWordPx = contentWidth() / words.length;
  const px = avgWordPx || (CFG.avgCharsPerWord * parseFloat(getComputedStyle(el.stream).fontSize) * 0.5);
  const pxPerSec = (state.wpm / 60) * px;

  // --- Set speed seamlessly via playbackRate (no restart, no seam).
  if (streamAnim) streamAnim.playbackRate = pxPerSec / animV0;

  // --- Top up words on the right. Appending extends content rightward without
  // shifting the visible strip, so it needs no re-baseline. Pre-fill enough to
  // ride out a main-thread stall.
  const cur = currentOffset();
  const need = window.innerWidth + pxPerSec * CFG.prefillSec + 600;
  while (cur + contentWidth() < need) {
    const wpx = addWord();
    if (words.length) avgWordPx = avgWordPx * 0.95 + wpx * 0.05;
  }

  // --- Re-baseline on a slow cadence: prune fully-off-screen words and restart the
  // animation from the (compensated) current position. Positionally seamless; this
  // is the only restart, kept rare so any commit cost is unnoticeable. It also
  // resets the animation clock so animSpanSec is never exhausted.
  if (now - lastRecycle >= CFG.recycleIntervalMs) {
    lastRecycle = now;
    let newOffset = currentOffset();
    while (words.length && newOffset + words[0].width < -120) {
      const removed = words.shift();
      el.stream.removeChild(removed.el);
      newOffset += removed.width; // compensate so remaining words stay put
    }
    offset = newOffset;
    startStripAnimation(pxPerSec / animV0);
  }

  // --- Gaze marker position (normalized -1..1 -> 0..100% of width).
  el.marker.style.left = `${((state.gazeNorm + 1) / 2) * 100}%`;
}

function toggleRun() {
  if (!state.running) {
    startRun();
  } else {
    state.running = false;
    offset = currentOffset();                       // freeze where we are
    if (streamAnim) { streamAnim.cancel(); streamAnim = null; }
    el.stream.style.transform = `translate3d(${offset}px,0,0)`;
    document.body.classList.remove("running");
    el.run.textContent = "Run";
  }
}

// ---------------------------------------------------------------------------
// Debug overlay
// ---------------------------------------------------------------------------
function updateDebug() {
  if (el.debug.classList.contains("hidden")) return;
  el.debug.textContent =
    `face:   ${state.faceSeen ? "yes" : "NO"}   ${state.fps.toFixed(0)} gaze / ${state.rfps.toFixed(0)} main fps\n` +
    `(scroll runs on the GPU compositor, not main fps)\n` +
    `iris:   ${state.irisRatio.toFixed(3)}\n` +
    `yaw:    ${state.headYaw.toFixed(3)}\n` +
    `raw:    ${state.gazeRaw.toFixed(3)}  [${state.calib.min.toFixed(2)}..${state.calib.max.toFixed(2)}]\n` +
    `gazeN:  ${state.gazeNorm.toFixed(3)}\n` +
    `error:  ${(state.gazeNorm - CFG.targetGaze).toFixed(3)}\n` +
    `WPM:    ${state.wpm.toFixed(0)}`;
}

// ---------------------------------------------------------------------------
// Helpers + wiring
// ---------------------------------------------------------------------------
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

el.start.addEventListener("click", startCamera);
el.calibrate.addEventListener("click", calibrate);
el.run.addEventListener("click", toggleRun);
el.debugBtn.addEventListener("click", () => {
  el.debug.classList.toggle("hidden");
  document.body.classList.toggle("debug");
});

// Register service worker (PWA install); ignore failures in dev.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
