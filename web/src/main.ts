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
import { makeLayer } from "neuroglancer/unstable/layer/index.js";
import {
  CredentialsProvider,
  makeCredentialsGetter,
} from "neuroglancer/unstable/credentials_provider/index.js";
import { registerDefaultCredentialsProvider } from "neuroglancer/unstable/credentials_provider/default_manager.js";

// Red overlay for the target-mask layer — mirrors proofreading/em/tube.py `_TINT`.
const TINT = `void main() {
  float v = toNormalized(getDataValue());
  emitRGBA(vec4(1.0, 0.2, 0.2, v > 0.5 ? 0.6 : 0.0));
}`;

// annotation tags: key -> tag, and tag -> color (mirrors proofreading/em/viewer.py TAG_COLORS)
const TAG_KEYS: Record<string, string> = {
  m: "merge error",
  s: "split error",
  e: "extend",
  q: "question",
};
const TAG_COLORS: Record<string, string> = {
  "merge error": "#ff3333",
  "split error": "#33aaff",
  extend: "#33ff66",
  question: "#ffcc00",
};
const annLayerName = (tag: string) => "ann:" + tag.replace(/ /g, "_");

// M4.2: skeleton guidance markers (guidebook). Branch points = merge/split candidates; end
// points/tips = extend candidates. Distinct from the 4 tag colors; local point layers, always-on.
const SKEL_BRANCH_LAYER = "skel:branch";
const SKEL_END_LAYER = "skel:end";
// const SKEL_HIGHLIGHT_LAYER = "skel:highlight"; // M4.5 hidden — does not redraw on branch switch
const SKEL_BRANCH_COLOR = "#cc66ff"; // branch points — magenta
const SKEL_END_COLOR = "#ffffff"; // end points / tips — white
// const SKEL_HIGHLIGHT_COLOR = "#ffff00"; // M4.5 hidden — bright yellow; re-enable with layer

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
  prebuilding?: number[];
}
interface Branch {
  path_id: number;
  state: string;
  n_nodes: number;
  length_nm: number;
  compartment: string;
  built: boolean;
  dist_to_root_nm: number;
}

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "http://localhost:8000").replace(/\/$/, "");
const ROOT_ID = params.get("root") || "864691135413357554";
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
let resNm: [number, number, number] = [16, 16, 40]; // tube voxel size (nm); set in setupViewer
let s = 0; // current arc position (nm) along the branch
let phase: "buffer" | "play" = "buffer";
let running = true;
let speed = 1000;
let bufferToken = 0; // cancels an in-flight buffering sweep when the branch changes
let scrubbing = false; // user is dragging the progress slider
let currentPid: number | null = null; // branch currently loaded (for `x` mark-done)
let prebuildingHint: number[] = []; // branches the backend is pre-building (for the HUD hint)

// uuid -> {tag, nm} for every drawn mark, so `d` can find the nearest one to the cursor (M2.2)
const annIndex = new Map<string, { tag: string; nm: [number, number, number] }>();

// M3: pause -> live full-res layers. `live` holds the source URLs (+ token, M3.2); on PAUSE we
// show the live EM (and graphene seg, M3.2) and hide the tube, on PLAY we swap back.
interface LiveSources {
  root_id: string;
  image_source: string;
  segmentation_source: string;
  skeleton_source: string | null;
  viewer_resolution_nm: number[];
  token: string | null;
}
let live: LiveSources | null = null;
let liveShown = false;
let liveEmLayer: any = null; // ManagedUserLayer refs; held only while paused (removed on play)
let liveSegLayer: any = null;
// M4.4: skeleton markers (branch/end points) for click-to-set-root; xyz in nm
let skelFeatures: { xyz_nm: number[]; path_id: number | null; kind: string }[] = [];
let rootArmed = false; // set-root mode: armed by the button, consumes ONE marker click
// let highlightIds: string[] = []; // M4.5 hidden

let last = performance.now();
let frames = 0;

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


const frame = (now: number) => {
  const dt = (now - last) / 1000;
  last = now;
  frames++;

  if (phase === "play" && running && dt > 0 && ptsVox.length >= 2) {
    s += speed * dt; // forward only
    if (s >= totalArc) {
      s = totalArc; // stop at the branch end (no auto-rewind)
      running = false;
      updatePlayButton();
    }
    setPosition(posAt(s));
  }

  // M3: live full-res layers only while truly paused on a branch (idle) — not during the
  // buffering sweep or a scrubber drag (camera is moving then), and not before live loaded.
  const wantLive = !!live && phase === "play" && !running && !scrubbing;
  if (wantLive !== liveShown) setLive(wantLive);

  if (frames % 6 === 0) {
    const frac = totalArc > 0 ? s / totalArc : 0;
    $("progresspct").textContent = phase === "buffer" ? "buffering" : `${(frac * 100).toFixed(0)}%`;
    if (!scrubbing) ($("progress") as HTMLInputElement).value = String(Math.round(frac * 1000));
  }
  requestAnimationFrame(frame);
};

function setupViewer(cam: Camera) {
  const res = cam.resolution_nm;
  resNm = res;
  const start = [
    cam.points_nm[0][0] / res[0],
    cam.points_nm[0][1] / res[1],
    cam.points_nm[0][2] / res[2],
  ];
  // M4.1: a skeleton-only segmentation layer = the whole-cell skeleton in the 3D panel. It's
  // skeleton-only (no volumetric) so it renders ONLY in 3D, nothing in the 2D cross-section, and
  // shares the graphene origin so the injected middleauth token (M3.2) authorizes it. Always-on.
  const skelLayers = live?.skeleton_source
    ? [{
        type: "segmentation", name: "skeleton", source: live.skeleton_source, segments: [live.root_id],
        selectedAlpha: 0.2,           // 2D cross-section opacity
        objectAlpha: 0.8,             // 3D mesh opacity
        meshSilhouetteRendering: 1.7, // silhouette strength
      }]
    : [];
  const state = {
    dimensions: { x: [res[0] * 1e-9, "m"], y: [res[1] * 1e-9, "m"], z: [res[2] * 1e-9, "m"] },
    position: start,
    // zoom tight on the neurite so the cross-section stays inside the ~1 µm tube radius
    crossSectionScale: 0.20,
    // 3D panel zoomed way out to frame the whole cell (~hundreds of µm); independent of the 2D zoom
    projectionScale: live?.skeleton_source ? 10000 : 6000,
    layers: [
      { type: "image", name: "em", source: cam.em_source, shaderControls: { normalized: { range: [100, 155] } } },
      { type: "image", name: "tgt", source: cam.tgt_source, shader: TINT, opacity: 0.2 },
      ...skelLayers,
    ],
    layout: live?.skeleton_source ? "xy-3d" : "xy",
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

  // Add the annotation layers only ONCE the global coordinate space is rank-3 (i.e. after the
  // em/tgt sources load). A local annotation layer created before then captures rank 0, and a
  // 3-D point then overflows on render ("offset is out of bounds").
  const rankWait = window.setInterval(() => {
    const v = viewer?.navigationState?.pose?.position?.value;
    if (v && v.length === 3) {
      window.clearInterval(rankWait);
      addAnnotationLayers();
    }
  }, 200);
}

let annAdded = false;
function addAnnotationLayers() {
  if (annAdded || !viewer) return;
  try {
    // Build each layer NOW (global space is rank-3) via the programmatic layer API — NOT
    // restoreState, which would re-create the image layers and re-race the local annotation
    // source back to rank-0. makeLayer's local source reads the current (rank-3) global space.
    for (const [tag, color] of Object.entries(TAG_COLORS)) {
      const name = annLayerName(tag);
      if (viewer.layerManager.getLayerByName(name)) continue;
      const managed = makeLayer(viewer.layerSpecification, name, {
        type: "annotation",
        source: "local://annotations",
        annotationColor: color,
      });
      viewer.layerManager.addManagedLayer(managed);
    }
    // M4.2: skeleton guidance marker layers (branch + end points), always-on point overlays
    for (const [name, color] of [
      [SKEL_BRANCH_LAYER, SKEL_BRANCH_COLOR],
      [SKEL_END_LAYER, SKEL_END_COLOR],
    ] as const) {
      if (viewer.layerManager.getLayerByName(name)) continue;
      const managed = makeLayer(viewer.layerSpecification, name, {
        type: "annotation",
        source: "local://annotations",
        annotationColor: color,
      });
      viewer.layerManager.addManagedLayer(managed);
    }
    // M4.5: highlight layer hidden pending branch-switch bug fix (polyline does not redraw on change)
    // if (!viewer.layerManager.getLayerByName(SKEL_HIGHLIGHT_LAYER)) {
    //   viewer.layerManager.addManagedLayer(
    //     makeLayer(viewer.layerSpecification, SKEL_HIGHLIGHT_LAYER, {
    //       type: "annotation",
    //       source: "local://annotations",
    //       annotationColor: SKEL_HIGHLIGHT_COLOR,
    //     }),
    //   );
    // }
    annAdded = true;
    console.log("[em] annotation layers added (rank-3)");
    restoreAnnotations(); // redraw any prior-session marks from the WAL
    drawSkeletonFeatures(); // M4.2: branch + end point guidance markers
    // if (ptsVox.length >= 2) drawBranchHighlight(ptsVox); // M4.5 hidden
  } catch (e) {
    console.warn("[em] addAnnotationLayers failed", e);
  }
}

// M4.2: draw branch points + end points (from the skeleton) onto their guidance layers
function drawSkelPoint(layerName: string, posVox: number[], id: string) {
  try {
    const layer: any = viewer.layerManager.getLayerByName(layerName);
    const src = layer?.layer?.localAnnotations;
    if (!src) return;
    src.add({ type: 0, id, point: Float32Array.of(posVox[0], posVox[1], posVox[2]), properties: [] });
  } catch (e) {
    console.warn("[em] drawSkelPoint failed", e);
  }
}

// M4.5 hidden — polyline does not redraw when branch changes; re-investigate before re-enabling.
// async function drawBranchHighlight(pts: [number, number, number][]) {
//   try {
//     let src: any = null;
//     for (let i = 0; i < 50; i++) {
//       const layer: any = viewer?.layerManager?.getLayerByName(SKEL_HIGHLIGHT_LAYER);
//       src = layer?.layer?.localAnnotations;
//       if (src) break;
//       await sleep(100);
//     }
//     if (!src) return;
//     for (const id of highlightIds) {
//       try { const ref = src.getReference(id); src.delete(ref); ref.dispose?.(); } catch { /* gone */ }
//     }
//     highlightIds = [];
//     for (let i = 0; i < pts.length - 1; i++) {
//       const id = `hl${i}`;
//       src.add({ type: 1, id,
//         pointA: Float32Array.of(pts[i][0], pts[i][1], pts[i][2]),
//         pointB: Float32Array.of(pts[i + 1][0], pts[i + 1][1], pts[i + 1][2]),
//         properties: [] });
//       highlightIds.push(id);
//     }
//   } catch (e) { console.warn("[em] drawBranchHighlight failed", e); }
// }

async function drawSkeletonFeatures() {
  let feats: { branch_points?: any[]; end_points?: any[] };
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/skeleton-features`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    feats = await r.json();
  } catch (e) {
    console.warn("[em] skeleton-features fetch failed", e);
    return;
  }
  indexSkelFeatures(feats); // remember for click-to-set-root (M4.4)
  // wait until the local annotation source has loaded (init is async after addManagedLayer)
  for (let i = 0; i < 50; i++) {
    const s: any = viewer.layerManager.getLayerByName(SKEL_END_LAYER);
    if (s?.layer?.localAnnotations) break;
    await sleep(100);
  }
  const toVox = (p: any): [number, number, number] => [
    p.xyz_nm[0] / resNm[0],
    p.xyz_nm[1] / resNm[1],
    p.xyz_nm[2] / resNm[2],
  ];
  (feats.branch_points || []).forEach((p, i) => drawSkelPoint(SKEL_BRANCH_LAYER, toVox(p), `bp${i}`));
  (feats.end_points || []).forEach((p, i) => drawSkelPoint(SKEL_END_LAYER, toVox(p), `ep${i}`));
  console.log(
    `[em] skeleton features: ${(feats.branch_points || []).length} branch, ${(feats.end_points || []).length} end`,
  );
}

function indexSkelFeatures(feats: { branch_points?: any[]; end_points?: any[] }) {
  skelFeatures = [
    ...(feats.branch_points || []).map((p) => ({ xyz_nm: p.xyz_nm, path_id: p.path_id, kind: "branch" })),
    ...(feats.end_points || []).map((p) => ({ xyz_nm: p.xyz_nm, path_id: p.path_id, kind: "end" })),
  ];
}

// M4.4: arm/disarm the one-shot set-root mode (deliberate, so a stray click can't re-root)
function setRootArmed(on: boolean) {
  rootArmed = on;
  const btn = $("setroot") as HTMLButtonElement;
  btn.textContent = on ? "● click a marker" : "set root";
  btn.style.background = on ? "#5a3a1c" : "";
  if (on) status("set-root armed — click a branch/end marker on the skeleton", "ok");
}

// M4.4: re-root the review at the clicked skeleton marker (somaless cells have no real soma, so
// the skeleton service's default root is an arbitrary tip). Re-derives the branch decomposition +
// proximal->distal order; markers are undirected (root-invariant) so they don't move; coverage is
// preserved (L2-keyed). Restarts review at the new most-proximal branch.
async function setRoot(xyz_nm: number[], kind: string) {
  status(`setting root at ${kind} point…`);
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/root`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ xyz_nm }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    indexSkelFeatures(resp.skeleton_features || {}); // refresh path_ids (positions unchanged)
    renderBranches(resp.branches);
    renderSummary(resp.summary);
    const next = (resp.branches as Branch[]).find((b) => b.state === "to_review") || resp.branches[0];
    status(`root set — review re-derived; starting at branch #${next.path_id}`, "ok");
    loadBranch(next.path_id);
  } catch (e) {
    status(`✗ set-root failed: ${e}`, "warn");
  }
}

// M2.2 resume: redraw the cell's annotations (from the WAL) onto the layers
async function restoreAnnotations() {
  let anns: any[] = [];
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/annotations`);
    anns = (await r.json()).annotations || [];
  } catch (e) {
    console.warn("[em] restore fetch failed", e);
    return;
  }
  if (!anns.length) return;
  // wait until the local annotation sources have loaded (they init async after addManagedLayer)
  for (let i = 0; i < 50; i++) {
    const sample: any = viewer.layerManager.getLayerByName(annLayerName("question"));
    if (sample?.layer?.localAnnotations) break;
    await sleep(100);
  }
  for (const a of anns) {
    drawPoint(a.tag, [a.xyz[0] / resNm[0], a.xyz[1] / resNm[1], a.xyz[2] / resNm[2]], a.uuid);
  }
  status(`resumed ${anns.length} annotation${anns.length === 1 ? "" : "s"}`, "ok");
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

// --- annotations (M2.1: drop a colored mark at the cursor) ---
function drawPoint(tag: string, posVox: number[], id: string) {
  try {
    const layer: any = viewer.layerManager.getLayerByName(annLayerName(tag));
    const src = layer?.layer?.localAnnotations; // the annotation UserLayer's LocalAnnotationSource
    if (!src) {
      console.warn("[em] annotation layer not ready for", tag);
      return;
    }
    src.add({
      type: 0, // AnnotationType.POINT
      id, // reuse the WAL uuid so we can find/remove it on delete (M2.2)
      point: Float32Array.of(posVox[0], posVox[1], posVox[2]),
      properties: [],
    });
    // remember it in nm so `d` can find the mark nearest the cursor regardless of branch/zoom
    annIndex.set(id, { tag, nm: [posVox[0] * resNm[0], posVox[1] * resNm[1], posVox[2] * resNm[2]] });
  } catch (e) {
    console.warn("[em] drawPoint failed", e);
  }
}

// remove a mark from its local annotation layer + the index (mirrors the WAL tombstone)
function removePoint(uuid: string, tag: string) {
  try {
    const layer: any = viewer.layerManager.getLayerByName(annLayerName(tag));
    const src = layer?.layer?.localAnnotations;
    if (src) {
      const ref = src.getReference(uuid); // AnnotationReference; addRef'd, so dispose after
      src.delete(ref);
      ref.dispose?.();
    }
  } catch (e) {
    console.warn("[em] removePoint failed", e);
  }
  annIndex.delete(uuid);
}

// M2.2 delete: tombstone the mark nearest the cursor (in nm), then drop it from the viewer
async function deleteNearest() {
  const ms = viewer?.mouseState;
  if (!ms?.active || !ms.position || ms.position.length < 3) {
    status("hover over a mark, then press d to delete", "warn");
    return;
  }
  const cx = ms.position[0] * resNm[0],
    cy = ms.position[1] * resNm[1],
    cz = ms.position[2] * resNm[2];
  let best: string | null = null;
  let bestTag = "";
  let bestD = Infinity;
  for (const [uuid, a] of annIndex) {
    const d = Math.hypot(a.nm[0] - cx, a.nm[1] - cy, a.nm[2] - cz);
    if (d < bestD) {
      bestD = d;
      best = uuid;
      bestTag = a.tag;
    }
  }
  const THRESH_NM = 2500; // "under the cursor" — generous vs the ~1 µm tube radius
  if (!best || bestD > THRESH_NM) {
    status(
      annIndex.size ? `no mark near cursor (nearest ${Math.round(bestD)} nm)` : "no marks to delete",
      "warn",
    );
    return;
  }
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/annotations/${best}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    removePoint(best, bestTag);
    // deleting a merge mark un-omits its branch -> repaint the checklist/summary
    if (resp.branches) renderBranches(resp.branches);
    if (resp.summary) renderSummary(resp.summary);
    status(`deleted ${bestTag} (${Math.round(bestD)} nm away)`, "ok");
  } catch (e) {
    status(`✗ delete failed: ${e}`, "warn");
  }
}

async function annotate(tag: string) {
  const ms = viewer?.mouseState;
  if (!ms || !ms.active || !ms.position || ms.position.length < 3) {
    status("hover over the image, then press the key to annotate", "warn");
    return;
  }
  const p = ms.position; // global voxels (same space as nav position)
  const xyz_nm = [p[0] * resNm[0], p[1] * resNm[1], p[2] * resNm[2]];
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/annotations`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tag, xyz_nm }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    drawPoint(tag, [p[0], p[1], p[2]], resp.annotation.uuid);
    status(`${tag} @ [${xyz_nm.map((x) => Math.round(x)).join(", ")}] nm`, "ok");
  } catch (e) {
    status(`✗ annotate failed: ${e}`, "warn");
  }
}

function updatePlayButton() {
  ($("toggle") as HTMLButtonElement).textContent = running ? "⏸ pause cam" : "▶ play cam";
}
function togglePlay() {
  running = !running;
  updatePlayButton();
}

// pre-cache the branch (step with idle dwells so neuroglancer loads each frustum), then glide
async function bufferAndPlay(pid: number) {
  const token = ++bufferToken;
  phase = "buffer";
  s = 0;
  // Visit every ~1200 nm along the branch so the local precomputed is warm for every frustum.
  // No upper-count cap: long branches need proportionally more dwell positions.
  const coverStep = Math.max(1, Math.round(1200 / stepNm));
  for (let i = 0; i < ptsVox.length; i += coverStep) {
    if (token !== bufferToken) return; // a newer branch took over
    setPosition(ptsVox[i]);
    const pct = Math.round((i / Math.max(1, ptsVox.length - 1)) * 100);
    status(`buffering branch ${pid} ${pct}% — caching…`);
    await sleep(120);
  }
  if (token !== bufferToken) return;
  s = 0;
  setPosition(ptsVox[0]);
  await sleep(400);
  if (token !== bufferToken) return;
  phase = "play";
  running = false; // stay paused after buffering; user presses play to start
  updatePlayButton();
  const hint = prebuildingHint.length ? ` · ↻ pre-building #${prebuildingHint.join(", #")}` : "";
  status(`branch ${pid}: ready — ${ptsVox.length} nodes${hint}`, "ok");
}

// --- coverage (M2.3: mark branch done + advance) ---
// list arrives proximal -> distal (M4.3); show soma-distance so the ordering is legible
const branchOption = (b: Branch) =>
  `<option value="${b.path_id}">#${b.path_id} · ${b.state} · ${b.compartment} · ${b.n_nodes}n · ${Math.round(b.dist_to_root_nm / 1000)}µm${b.built ? " ✓" : ""}</option>`;

// repaint the branch dropdown from a fresh checklist, keeping the current selection
function renderBranches(branches: Branch[]) {
  const sel = $("branch") as HTMLSelectElement;
  const prev = sel.value;
  sel.innerHTML = branches.map(branchOption).join("");
  if (prev && branches.some((b) => String(b.path_id) === prev)) sel.value = prev;
}

function renderSummary(summary: Record<string, number>) {
  $("coverage").textContent =
    `to_review ${summary.to_review ?? 0} · covered ${summary.covered ?? 0} · omitted ${summary.omitted ?? 0}`;
}

// mark the loaded branch reviewed (durably), repaint coverage, advance to the next to-review
async function markDone() {
  if (currentPid === null) return;
  const pid = currentPid;
  status(`marking branch ${pid} done…`);
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/branches/${pid}/done`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    renderBranches(resp.branches);
    renderSummary(resp.summary);
    const next = resp.next_path_id;
    if (next === null || next === undefined) {
      running = false;
      updatePlayButton();
      status(`✓ branch ${pid} done — cell complete (nothing left to review)`, "ok");
    } else {
      status(`✓ branch ${pid} done — advancing to #${next}`, "ok");
      loadBranch(next);
    }
  } catch (e) {
    status(`✗ mark-done failed: ${e}`, "warn");
  }
}

// M3.2: authenticate the graphene segmentation with our CAVE token. The graphene datasource asks
// the credentials manager for a "middleauthapp" provider (keyed by the seg server origin); we
// OVERRIDE it to hand back the token directly — no /auth_info fetch, no OAuth popup, no
// localStorage. The default viewer builds its credentials manager from this global registry at
// creation, so registration MUST run before setupDefaultViewer(). The token stays on localhost
// (the backend binds 127.0.0.1) and is sent only to the graphene server over HTTPS as a Bearer.
let middleauthRegistered = false;
function registerMiddleAuthToken(token: string) {
  if (middleauthRegistered) return;
  class MiddleAuthTokenProvider extends CredentialsProvider<any> {
    get = makeCredentialsGetter(async () => ({ tokenType: "Bearer", accessToken: token }));
  }
  registerDefaultCredentialsProvider("middleauthapp", () => new MiddleAuthTokenProvider());
  middleauthRegistered = true;
  console.log("[em] middleauth token provider registered");
}

// --- live full-res layers (M3: swap tube <-> live mip0 EM + graphene seg on play/pause) ---
// The live layers are ADDED only while paused and REMOVED on play. Hiding (setVisible) isn't
// enough: a hidden layer keeps its chunk sources resident and the graphene seg keeps doing
// background work, both of which compete with the tube's buffering sweep for the deliberately
// bounded chunk cache + download queue -> the next branch builds/buffers slower. Removing frees
// them (removeManagedLayer disposes the layer), so motion buffers as fast as before M3.
function addLiveLayers() {
  if (liveSegLayer || !viewer || !live) return;
  try {
    liveEmLayer = makeLayer(viewer.layerSpecification, "live_em", {
      type: "image",
      source: live.image_source,
      shaderControls: { normalized: { range: [100, 155] } },
    });
    viewer.layerManager.addManagedLayer(liveEmLayer);
    // the real graphene segmentation, with only this cell's root selected (string: >2^53)
    liveSegLayer = makeLayer(viewer.layerSpecification, "live_seg", {
      type: "segmentation",
      source: live.segmentation_source,
      segments: [live.root_id],
      selectedAlpha: 0.2,
      objectAlpha: 0.8,
      meshSilhouetteRendering: 1.7,
    });
    viewer.layerManager.addManagedLayer(liveSegLayer);
    console.log("[em] live layers added (em + seg)");
  } catch (e) {
    console.warn("[em] addLiveLayers failed", e);
  }
}

function removeLiveLayers() {
  for (const l of [liveSegLayer, liveEmLayer]) {
    try {
      if (l && viewer.layerManager.has(l)) viewer.layerManager.removeManagedLayer(l);
    } catch (e) {
      console.warn("[em] removeLiveLayers failed", e);
    }
  }
  liveEmLayer = liveSegLayer = null;
}

// paused: add the live layers + hide the tube. playing: remove the live layers + show the tube.
function setLive(on: boolean) {
  try {
    const lm = viewer.layerManager;
    if (on) addLiveLayers();
    else removeLiveLayers();
    lm.getLayerByName("em")?.setVisible(!on); // tube EM
    lm.getLayerByName("tgt")?.setVisible(!on); // tube mask — graphene seg replaces it when live
  } catch (e) {
    console.warn("[em] setLive failed", e);
    return;
  }
  liveShown = on;
  if (on) status("paused — live full-res EM + segmentation", "ok");
}

// M2.4: omit the current branch + its distal descendants (a foreign segment to be split off). At
// an X crossing the cell's continuation is a sibling branch, so it stays to-review. Reversible by
// deleting the red merge mark dropped at the junction (`d`).
async function omitBranch() {
  if (currentPid === null) return;
  const pid = currentPid;
  status(`omitting branch ${pid}…`);
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/branches/${pid}/omit`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    const a = resp.annotation;
    if (a) drawPoint(a.tag, [a.xyz[0] / resNm[0], a.xyz[1] / resNm[1], a.xyz[2] / resNm[2]], a.uuid);
    renderBranches(resp.branches);
    renderSummary(resp.summary);
    const next = resp.next_path_id;
    if (next === null || next === undefined) {
      running = false;
      updatePlayButton();
      status(`⊘ branch ${pid} omitted (${resp.omitted_l2_count} L2) — nothing left to review`, "ok");
    } else {
      status(`⊘ branch ${pid} omitted (${resp.omitted_l2_count} L2) — advancing to #${next}`, "ok");
      loadBranch(next);
    }
  } catch (e) {
    status(`✗ omit failed: ${e}`, "warn");
  }
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
  // drawBranchHighlight(ptsVox); // M4.5 hidden — does not redraw on branch switch
  currentPid = pid;
  prebuildingHint = cam.prebuilding || [];
  ($("branch") as HTMLSelectElement).value = String(pid);
  bufferAndPlay(pid);
}

async function main() {
  speed = parseFloat(($("speed") as HTMLInputElement).value);

  // HUD controls
  ($("toggle") as HTMLButtonElement).onclick = () => togglePlay();
  ($("speed") as HTMLInputElement).oninput = (e) =>
    (speed = parseFloat((e.target as HTMLInputElement).value));
  ($("setroot") as HTMLButtonElement).onclick = () => setRootArmed(!rootArmed);
  ($("omit") as HTMLButtonElement).onclick = () => omitBranch();

  // cell-id input: load a different cell by reloading with ?root=<id> (the page re-opens it;
  // each cell resumes its own WAL on the backend). Other params (datastack/api) are preserved.
  const cellInput = $("cellid") as HTMLInputElement;
  cellInput.value = ROOT_ID;
  const loadCell = () => {
    const id = cellInput.value.trim();
    if (!id || id === ROOT_ID) return;
    const p = new URLSearchParams(location.search);
    p.set("root", id);
    location.search = p.toString(); // reload with the new cell
  };
  ($("loadcell") as HTMLButtonElement).onclick = loadCell;
  cellInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadCell();
  });

  // progress scrubber: drag to move the camera along the branch (pauses playback so it
  // doesn't fight the drag). The frame loop updates the slider only when not scrubbing.
  const prog = $("progress") as HTMLInputElement;
  const scrubTo = () => {
    if (phase !== "play" || totalArc <= 0) return;
    running = false;
    updatePlayButton();
    s = (parseFloat(prog.value) / 1000) * totalArc;
    setPosition(posAt(s));
  };
  prog.addEventListener("pointerdown", () => (scrubbing = true));
  prog.addEventListener("pointerup", () => (scrubbing = false));
  prog.addEventListener("input", scrubTo);

  // annotation keys (capture phase + stopImmediatePropagation so they beat neuroglancer's
  // own m/s/e/q/x/n bindings). Only fire when the cursor is over a data panel.
  window.addEventListener(
    "keydown",
    (e) => {
      if (!viewer) return;
      // ignore while typing in a form field (cell-id input, branch dropdown)
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      // space = play/pause (works anywhere, not cursor-dependent)
      if (e.key === " ") {
        e.preventDefault();
        e.stopImmediatePropagation();
        togglePlay();
        return;
      }
      // x = mark the current branch done + advance (branch-level, not cursor-dependent)
      if ((e.key === "x" || e.key === "X") && currentPid !== null) {
        e.preventDefault();
        e.stopImmediatePropagation();
        markDone();
        return;
      }
      // o = omit the current branch + its distal descendants (foreign / to be split off)
      if ((e.key === "o" || e.key === "O") && currentPid !== null) {
        e.preventDefault();
        e.stopImmediatePropagation();
        omitBranch();
        return;
      }
      // annotation keys only fire when the cursor is over a data panel
      if (!viewer.mouseState?.active) return;
      // d / Backspace / Delete = remove the mark nearest the cursor (M2.2)
      if (e.key === "d" || e.key === "D" || e.key === "Backspace" || e.key === "Delete") {
        e.preventDefault();
        e.stopImmediatePropagation();
        deleteNearest();
        return;
      }
      const tag = TAG_KEYS[e.key.toLowerCase()];
      if (!tag) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      annotate(tag);
    },
    true,
  );

  // M4.4: set-root mode. Only fires when ARMED by the "set root" button (so a stray click can't
  // re-root). Armed + click on a data panel = one-shot: snap to the nearest marker, then disarm.
  // Off-panel clicks (e.g. the HUD button) are ignored so they stay armed / can cancel.
  window.addEventListener(
    "click",
    (e) => {
      if (!rootArmed || !viewer || !skelFeatures.length) return;
      const ms = viewer.mouseState;
      if (!ms?.active || !ms.position || ms.position.length < 3) return; // off-panel — stay armed
      e.preventDefault();
      e.stopImmediatePropagation();
      const cx = ms.position[0] * resNm[0],
        cy = ms.position[1] * resNm[1],
        cz = ms.position[2] * resNm[2];
      let best: (typeof skelFeatures)[number] | null = null;
      let bestD = Infinity;
      for (const f of skelFeatures) {
        const d = Math.hypot(f.xyz_nm[0] - cx, f.xyz_nm[1] - cy, f.xyz_nm[2] - cz);
        if (d < bestD) {
          bestD = d;
          best = f;
        }
      }
      setRootArmed(false); // one-shot
      if (!best || bestD > 2000) {
        status("no skeleton marker near the click — root unchanged", "warn");
        return;
      }
      setRoot(best.xyz_nm, best.kind);
    },
    true,
  );

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

  // M3: prefetch the live full-res sources (shown on pause). Non-fatal — the tube still works.
  try {
    const lr = await fetch(`${API}/api/cells/${ROOT_ID}/live-sources`);
    if (lr.ok) live = await lr.json();
    console.log("[em] live sources", live);
  } catch (e) {
    console.warn("[em] live-sources fetch failed (tube only)", e);
  }
  // register the graphene credentials BEFORE the viewer is created (setupViewer runs in loadBranch)
  if (live?.token) registerMiddleAuthToken(live.token);

  // branch checklist -> picker
  let branches: Branch[];
  try {
    const br = await fetch(`${API}/api/cells/${ROOT_ID}/branches`);
    const data = await br.json();
    branches = data.branches;
    renderSummary(data.summary || {});
  } catch (e) {
    status(`✗ couldn't list branches: ${e}`, "warn");
    return;
  }
  const sel = $("branch") as HTMLSelectElement;
  renderBranches(branches);
  sel.onchange = () => loadBranch(parseInt(sel.value, 10));

  // start on the first to-review branch (fallback: first branch)
  const first = (branches.find((b) => b.state === "to_review") || branches[0]).path_id;
  loadBranch(first);
}

main();
