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
  fps: 0,
  lastFrame: 0,
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
  if (state.landmarker && el.cam.readyState >= 2) {
    if (el.cam.currentTime !== state.lastVideoTime) {
      state.lastVideoTime = el.cam.currentTime;
      const res = state.landmarker.detectForVideo(el.cam, now);
      processResult(res);
    }
    // fps
    if (state.lastFrame) {
      const dt = (now - state.lastFrame) / 1000;
      state.fps = state.fps * 0.9 + (1 / dt) * 0.1;
    }
    state.lastFrame = now;
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
let words = [];     // [{ el, width }]
let offset = 0;     // current translateX of #stream (<= 0)
let nextWord = 0;   // index into SAMPLE_TEXT (loops)
let avgWordPx = 0;
let runLast = 0;

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

function startRun() {
  state.running = true;
  document.body.classList.add("running");
  el.run.textContent = "Pause";
  // Seed the strip starting just off the right edge.
  offset = window.innerWidth;
  el.stream.style.transform = `translateX(${offset}px)`;
  runLast = performance.now();
  requestAnimationFrame(renderLoop);
}

function renderLoop(now) {
  if (!state.running) return;
  const dt = Math.min(0.05, (now - runLast) / 1000);
  runLast = now;

  // --- Controller: integrate gaze error into speed (drives gaze back to target).
  const error = state.gazeNorm - CFG.targetGaze; // + = looking ahead/right
  if (state.faceSeen) {
    state.wpm = clamp(state.wpm + CFG.kp * error * dt, CFG.wpmMin, CFG.wpmMax);
  }

  // --- Convert WPM to px/sec using measured average word width.
  if (avgWordPx === 0 && words.length) avgWordPx = contentWidth() / words.length;
  const px = avgWordPx || (CFG.avgCharsPerWord * parseFloat(getComputedStyle(el.stream).fontSize) * 0.5);
  const pxPerSec = (state.wpm / 60) * px;

  // --- Advance left.
  offset -= pxPerSec * dt;

  // --- Ensure content fills to the right.
  while (offset + contentWidth() < window.innerWidth + 600) {
    const wpx = addWord();
    if (words.length) avgWordPx = avgWordPx * 0.95 + wpx * 0.05;
  }

  // --- Prune words fully off the left; compensate offset so visuals don't jump.
  while (words.length && offset + words[0].width < -100) {
    const removed = words.shift();
    el.stream.removeChild(removed.el);
    offset += removed.width;
  }

  el.stream.style.transform = `translateX(${offset}px)`;

  // --- Gaze marker position (normalized -1..1 -> 0..100% of width).
  el.marker.style.left = `${((state.gazeNorm + 1) / 2) * 100}%`;

  requestAnimationFrame(renderLoop);
}

function toggleRun() {
  if (!state.running) {
    startRun();
  } else {
    state.running = false;
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
    `face:   ${state.faceSeen ? "yes" : "NO"}   fps: ${state.fps.toFixed(0)}\n` +
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
