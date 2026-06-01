// EM proofreading — M1 frontend (browser-native glide, driven by the FastAPI backend).
//
// The camera is animated entirely CLIENT-SIDE (a requestAnimationFrame loop mutating
// `navigationState.pose.position`) over the served sparse tube — no python in the loop.
// The backend (proofreading.em.api) provides: open/resume a cell, the branch checklist, and
// per-branch camera paths + the shared tube precomputed (served same-origin at /tube/...).
//
// Flow: POST /api/cells -> header; GET .../branches -> picker; GET .../branches/{id}/camera ->
// {points_nm, em_source, tgt_source}. The em/tgt layers are the SAME shared volume for every
// branch, so the viewer is built once; switching a branch swaps the camera path + re-buffers.

import "neuroglancer/unstable/ui/default_viewer.css";
// Side-effect registration of layer types + datasources (precomputed) + kvstores (http).
// Without this the viewer shell loads but layers have no renderer/data backend.
import "neuroglancer/unstable/main_module.js";
import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";

// Red overlay for the target-mask layer — mirrors proofreading/em/tube.py `_TINT`.
const TINT = `void main() {
  float v = toNormalized(getDataValue());
  emitRGBA(vec4(1.0, 0.2, 0.2, v > 0.5 ? 0.6 : 0.0));
}`;

interface Camera {
  path_id: number;
  root_id: string;
  resolution_nm: [number, number, number];
  points_nm: [number, number, number][];
  orientations: number[][] | null;
  step_nm: number;
  em_source: string;
  tgt_source: string;
  build: { cached: boolean; seconds: number };
}
interface Branch {
  path_id: number;
  state: string;
  n_nodes: number;
  length_nm: number;
  compartment: string;
  built: boolean;
}

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "http://localhost:8000").replace(/\/$/, "");
const ROOT_ID = params.get("root") || "864691135572530981";
const DATASTACK = params.get("datastack") || "minnie65_public";

const $ = (id: string) => document.getElementById(id)!;
const status = (msg: string, cls = "") => {
  $("status").textContent = msg;
  $("status").className = cls;
};

// --- module-scoped animation state (swapped per branch) ---
let viewer: any = null;
let ptsVox: [number, number, number][] = [];
let cum: number[] = [0];
let totalArc = 0;
let stepNm = 500;
let s = 0;
let dir = 1; // ping-pong direction
let phase: "buffer" | "play" = "buffer";
let running = true;
let speed = 2000;
let bufferToken = 0; // cancels an in-flight buffering sweep when the branch changes

// fps / degradation tracking
const t0 = performance.now();
let last = t0;
let frames = 0;
let fpsSmooth = 0;
let fpsMin = Infinity;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const setPosition = (vox: [number, number, number]) => {
  try {
    const pos = viewer.navigationState.pose.position;
    const cur = pos.value;
    if (cur && cur.length === 3) pos.value = Float32Array.of(vox[0], vox[1], vox[2]);
  } catch {
    /* coordinate space not ready yet; try next frame */
  }
};

// position (voxels) at arc length s (nm) along the current branch polyline
const posAt = (q: number): [number, number, number] => {
  q = Math.max(0, Math.min(totalArc, q));
  let lo = 0,
    hi = cum.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] <= q) lo = mid;
    else hi = mid;
  }
  const seg = cum[hi] - cum[lo] || 1;
  const f = (q - cum[lo]) / seg;
  const a = ptsVox[lo];
  const b = ptsVox[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
};

const fmt = (ms: number) => {
  const t = Math.floor(ms / 1000);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

const frame = (now: number) => {
  const dt = (now - last) / 1000;
  last = now;
  frames++;

  if (phase === "play" && running && dt > 0 && ptsVox.length >= 2) {
    s += dir * speed * dt;
    if (s >= totalArc) {
      s = totalArc;
      dir = -1;
    } else if (s <= 0) {
      s = 0;
      dir = 1;
    }
    setPosition(posAt(s));
  }

  if (dt > 0) {
    const inst = 1 / dt;
    fpsSmooth = fpsSmooth ? fpsSmooth * 0.9 + inst * 0.1 : inst;
    if (phase === "play" && now - t0 > 2000 && fpsSmooth < fpsMin) fpsMin = fpsSmooth;
  }

  if (frames % 6 === 0) {
    $("uptime").textContent = fmt(now - t0);
    $("frames").textContent = String(frames);
    $("fps").textContent = fpsSmooth.toFixed(0);
    const minEl = $("fpsmin");
    minEl.textContent = fpsMin === Infinity ? "–" : fpsMin.toFixed(0);
    minEl.className = "v" + (fpsMin < 40 ? " warn" : fpsMin >= 55 ? " ok" : "");
    $("progress").textContent =
      phase === "buffer" ? "buffering" : `${((s / totalArc) * 100).toFixed(0)}%`;
  }
  requestAnimationFrame(frame);
};

function setupViewer(cam: Camera) {
  const res = cam.resolution_nm;
  const start = [
    cam.points_nm[0][0] / res[0],
    cam.points_nm[0][1] / res[1],
    cam.points_nm[0][2] / res[2],
  ];
  const state = {
    dimensions: { x: [res[0] * 1e-9, "m"], y: [res[1] * 1e-9, "m"], z: [res[2] * 1e-9, "m"] },
    position: start,
    // zoom tight on the neurite so the cross-section stays inside the ~1 µm tube radius
    crossSectionScale: 0.12,
    projectionScale: 6000,
    layers: [
      { type: "image", name: "em", source: cam.em_source },
      { type: "image", name: "tgt", source: cam.tgt_source, shader: TINT, opacity: 0.85 },
    ],
    layout: "xy",
    showDefaultAnnotations: false,
  };
  viewer = setupDefaultViewer();
  viewer.state.restoreState(state);
  (window as any).viewer = viewer;

  // Cache limits — bound the working set to roughly ONE branch in BOTH caches so switching
  // branches EVICTS the previous one (LRU) rather than accumulating until the renderer process
  // is OOM-killed ("render process gone" = a blank tab needing reload). A branch tube is at most
  // a few hundred MB, so ~1.5 GB system / ~1 GB GPU holds one comfortably and evicts the prior.
  // Prefetch OFF: our buffering sweep already loads the branch; prefetch only inflates the burst.
  try {
    const cq = viewer.dataContext.chunkQueueManager;
    cq.capacities.gpuMemory.sizeLimit.value = 1e9; // visible/visited tiles; evicts on switch
    cq.capacities.gpuMemory.itemLimit.value = 1e6;
    cq.capacities.systemMemory.sizeLimit.value = 1.5e9; // one branch decoded + margin
    cq.capacities.systemMemory.itemLimit.value = 1e6;
    cq.capacities.download.itemLimit.value = 16;
    cq.enablePrefetch.value = false;
  } catch (e) {
    console.warn("[em] could not set cache limits", e);
  }

  // surface a WebGL context loss (GPU memory pressure) instead of a silent blank
  try {
    const canvas = viewer.display?.canvas as HTMLCanvasElement | undefined;
    canvas?.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      status("✗ WebGL context lost (GPU memory) — reload the page", "warn");
      console.error("[em] webglcontextlost");
    });
  } catch {
    /* ignore */
  }
  requestAnimationFrame(frame);
}

function setBranch(cam: Camera) {
  const res = cam.resolution_nm;
  ptsVox = cam.points_nm.map(
    (p) => [p[0] / res[0], p[1] / res[1], p[2] / res[2]] as [number, number, number],
  );
  cum = [0];
  for (let i = 1; i < cam.points_nm.length; i++) {
    const a = cam.points_nm[i - 1];
    const b = cam.points_nm[i];
    cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  totalArc = cum[cum.length - 1];
  stepNm = cam.step_nm || 500;
}

// pre-cache the branch (step with idle dwells so neuroglancer loads each frustum), then glide
async function bufferAndPlay(pid: number) {
  const token = ++bufferToken;
  phase = "buffer";
  s = 0;
  dir = 1;
  const coverStep = Math.max(1, Math.round(1200 / stepNm));
  const stepNodes = Math.max(coverStep, Math.ceil(ptsVox.length / 90));
  for (let i = 0; i < ptsVox.length; i += stepNodes) {
    if (token !== bufferToken) return; // a newer branch took over
    setPosition(ptsVox[i]);
    const pct = Math.round((i / Math.max(1, ptsVox.length - 1)) * 100);
    status(`buffering branch ${pid} ${pct}% — caching…`);
    await sleep(120);
  }
  if (token !== bufferToken) return;
  setPosition(ptsVox[0]);
  await sleep(400);
  if (token !== bufferToken) return;
  phase = "play";
  status(`branch ${pid}: gliding (ping-pong) — ${ptsVox.length} nodes`, "ok");
}

async function loadBranch(pid: number) {
  bufferToken++; // stop any current buffering immediately
  phase = "buffer";
  status(`branch ${pid}: fetching camera path (building tube if first visit)…`);
  let cam: Camera;
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/branches/${pid}/camera`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    cam = await r.json();
  } catch (e) {
    status(`✗ camera fetch failed for branch ${pid}: ${e}`, "warn");
    return;
  }
  if (!cam.points_nm || cam.points_nm.length < 2) {
    status(`✗ branch ${pid} has <2 camera nodes`, "warn");
    return;
  }
  if (!viewer) setupViewer(cam);
  setBranch(cam);
  ($("branch") as HTMLSelectElement).value = String(pid);
  bufferAndPlay(pid);
}

async function main() {
  speed = parseFloat(($("speed") as HTMLInputElement).value);

  // HUD controls
  ($("toggle") as HTMLButtonElement).onclick = (e) => {
    running = !running;
    (e.target as HTMLButtonElement).textContent = running ? "⏸ pause cam" : "▶ play cam";
  };
  ($("speed") as HTMLInputElement).oninput = (e) =>
    (speed = parseFloat((e.target as HTMLInputElement).value));
  ($("resetmin") as HTMLButtonElement).onclick = () => (fpsMin = Infinity);

  // open / resume the cell
  status(`opening cell ${ROOT_ID}…`);
  try {
    const hr = await fetch(`${API}/api/cells`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ root_id: ROOT_ID, datastack: DATASTACK }),
    });
    if (!hr.ok) throw new Error(`HTTP ${hr.status}`);
    const header = await hr.json();
    console.log("[em] cell header", header);
  } catch (e) {
    status(`✗ couldn't open cell — is the backend up? (uv run --extra em --extra serve python -m proofreading.em.serve)  ${e}`, "warn");
    return;
  }

  // branch checklist -> picker
  let branches: Branch[];
  try {
    const br = await fetch(`${API}/api/cells/${ROOT_ID}/branches`);
    branches = (await br.json()).branches;
  } catch (e) {
    status(`✗ couldn't list branches: ${e}`, "warn");
    return;
  }
  const sel = $("branch") as HTMLSelectElement;
  sel.innerHTML = branches
    .map(
      (b) =>
        `<option value="${b.path_id}">#${b.path_id} · ${b.state} · ${b.compartment} · ${b.n_nodes}n${b.built ? " ✓" : ""}</option>`,
    )
    .join("");
  sel.onchange = () => loadBranch(parseInt(sel.value, 10));

  // start on the first to-review branch (fallback: first branch)
  const first = (branches.find((b) => b.state === "to_review") || branches[0]).path_id;
  loadBranch(first);
}

main();
