// Axon myelination fly-through -- per-node tagging.
//
// Flies axon-compartment branches only (via `branches(compartment="axon")`) and lets the user
// tag individual skeleton NODES as myelinated (key `t`, hover + press, same interaction shape
// as main.ts's merge/split/extend/question point-tags -- see CONTEXT.md), defaulting to
// unmyelinated (absence of a tag). This replaced an earlier continuous on/off toggle design:
// a stray keypress could silently start a durable recording, and neither the toggle intervals
// nor their single-node overrides were real neuroglancer annotations, so they weren't visible
// or deletable in neuroglancer's own Annotations panel. Tags are.
//
// Skeleton context comes from the whole-cell layers (a 3D skeleton mesh + branch-point/tip
// markers, both modelled on main.ts), not a per-branch overlay. Tags still snap to the TRUE,
// sparse skeleton vertices (`cam.nodes_nm`, kept in `currentNodesNm`) -- NOT the arc-length-
// resampled flight path (`cam.points_nm`, used only for camera smoothness) -- server-side.

import { makeLayer } from "neuroglancer/unstable/layer/index.js";
import {
  CredentialsProvider,
  makeCredentialsGetter,
} from "neuroglancer/unstable/credentials_provider/index.js";
import { registerDefaultCredentialsProvider } from "neuroglancer/unstable/credentials_provider/default_manager.js";
import { createFlyKernel } from "./flykernel";

interface Branch {
  path_id: number;
  state: string; // error-review coverage -- IRRELEVANT here, see myelin_state
  myelin_state: string; // "to_review" | "covered" -- this tool's own coverage dimension
  n_nodes: number;
  length_nm: number;
  compartment: string;
  built: boolean;
  dist_to_root_nm: number;
}

interface MyelinTag {
  uuid: string;
  xyz_nm: number[];
  path_id: number | null;
}

// live full-res sources (same endpoint main.ts uses); we only need the skeleton source + token
// here -- the myelin tool doesn't do main.ts's pause->live-EM swap.
interface LiveSources {
  root_id: string;
  image_source: string;
  segmentation_source: string;
  skeleton_source: string | null;
  viewer_resolution_nm: number[];
  token: string | null;
}
let live: LiveSources | null = null;

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "http://localhost:8000").replace(/\/$/, "");
// default has confirmed axon branches (57) -- 864691135413357554, the prior default, has none,
// which made an un-parameterized first open look broken (empty branch picker, nothing to fly).
const ROOT_ID = params.get("root") || "864691136335553971";
const DATASTACK = params.get("datastack") || "minnie65_public";

const $ = (id: string) => document.getElementById(id)!;
const status = (msg: string, cls = "") => {
  $("status").textContent = msg;
  $("status").className = cls;
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// whole-cell skeleton guidance markers (mirrors main.ts's M4.2 skel:branch / skel:end layers):
// branch points + tips for the ENTIRE cell, fetched once per cell-open from /skeleton-features,
// so you can see where the current branch sits in the whole arbor. The 3D skeleton mesh layer
// (see buildViewerState) covers the arbor's shape, so there's no per-branch line/node overlay.
const SKEL_BRANCH_LAYER = "myelin:branchpt";
const SKEL_END_LAYER = "myelin:endpt";
const SKEL_BRANCH_COLOR = "#cc66ff"; // branch points -- magenta (matches main.ts)
const SKEL_END_COLOR = "#ffffff"; // tips -- white (matches main.ts)
let skelMarkerLayersAdded = false;

// the actual myelin data: one green dot per tagged node.
const TAG_LAYER = "myelin:tag";
const TAG_COLOR = "#33ff99";
let tagLayerAdded = false;
let tagDeleteSignalWired = false;
let drawnTagIds: string[] = [];
let currentTags: MyelinTag[] = [];

// ids we are ABOUT to remove ourselves (redraw housekeeping or our own `d`-key delete) --
// consulted by the childDeleted signal handler so we don't double-fire the backend DELETE for
// removals we already know about. Populated right before every `src.delete(ref)` call below.
const selfDeleting = new Set<string>();

// "paint while flying" (key `p`) -- auto-tags every true node the camera passes during
// playback, so a long myelinated stretch doesn't need one `t` press per node. Reset on every
// branch load / markDone, mirroring how the old toggle auto-closed on branch switch.
let paintMode = false;
let lastPaintedNodeIdx: number | null = null;
let currentNodesNm: number[][] = [];

function updatePlayButton() {
  ($("toggle") as HTMLButtonElement).textContent = kernel.isRunning() ? "pause cam" : "play cam";
}

function updatePaintUI() {
  const el = $("paintstate");
  el.textContent = paintMode ? "● PAINTING (p to stop)" : "";
  el.className = paintMode ? "warn" : "";
}

function setPaintMode(on: boolean) {
  paintMode = on;
  lastPaintedNodeIdx = null;
  updatePaintUI();
}

function addSkelMarkerLayers() {
  if (skelMarkerLayersAdded) return;
  const viewer = kernel.getViewer();
  try {
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
    skelMarkerLayersAdded = true;
  } catch (e) {
    console.warn("[myelin] addSkelMarkerLayers failed", e);
  }
}

// draw the WHOLE cell's branch points + tips once (they're root-invariant positions, so unlike
// the per-branch overlays these never need redrawing on branch switch).
async function drawSkeletonFeatures() {
  let feats: { branch_points?: any[]; end_points?: any[] };
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/skeleton-features`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    feats = await r.json();
  } catch (e) {
    console.warn("[myelin] skeleton-features fetch failed", e);
    return;
  }
  const viewer = kernel.getViewer();
  const resNm = kernel.getResNm();
  for (let i = 0; i < 50; i++) {
    const s: any = viewer?.layerManager?.getLayerByName(SKEL_END_LAYER);
    if (s?.layer?.localAnnotations) break;
    await sleep(100);
  }
  const draw = (layerName: string, pts: any[], prefix: string) => {
    const layer: any = viewer?.layerManager?.getLayerByName(layerName);
    const src = layer?.layer?.localAnnotations;
    if (!src) return;
    pts.forEach((p, i) => {
      src.add({
        type: 0, // AnnotationType.POINT
        id: `${prefix}${i}`,
        point: Float32Array.of(
          p.xyz_nm[0] / resNm[0], p.xyz_nm[1] / resNm[1], p.xyz_nm[2] / resNm[2],
        ),
        properties: [],
      });
    });
  };
  draw(SKEL_BRANCH_LAYER, feats.branch_points || [], "bp");
  draw(SKEL_END_LAYER, feats.end_points || [], "ep");
}

function addTagLayer() {
  if (!tagLayerAdded) {
    const viewer = kernel.getViewer();
    try {
      if (!viewer.layerManager.getLayerByName(TAG_LAYER)) {
        const managed = makeLayer(viewer.layerSpecification, TAG_LAYER, {
          type: "annotation",
          source: "local://annotations",
          annotationColor: TAG_COLOR,
        });
        viewer.layerManager.addManagedLayer(managed);
      }
      tagLayerAdded = true;
    } catch (e) {
      console.warn("[myelin] addTagLayer failed", e);
    }
  }
  wireTagDeleteSignal();
}

// subscribe to the tag layer's own delete signal -- fires for BOTH our programmatic deletes
// (redraw housekeeping, the `d` key) AND deletes made through neuroglancer's native Annotations
// panel. The `selfDeleting` guard tells the two apart: if WE already know about this id, just
// consume the guard entry; otherwise the user deleted it natively, so sync it to the backend.
async function wireTagDeleteSignal() {
  if (tagDeleteSignalWired) return;
  const viewer = kernel.getViewer();
  let src: any = null;
  for (let i = 0; i < 50; i++) {
    const layer: any = viewer?.layerManager?.getLayerByName(TAG_LAYER);
    src = layer?.layer?.localAnnotations;
    if (src) break;
    await sleep(100);
  }
  if (!src) return;
  src.childDeleted.add((id: string) => {
    if (selfDeleting.delete(id)) return;
    currentTags = currentTags.filter((t) => t.uuid !== id);
    fetch(`${API}/api/cells/${ROOT_ID}/myelin/tag/${id}`, { method: "DELETE" })
      .then(() => status("removed tag via annotation panel", "ok"))
      .catch((e: unknown) => console.warn("[myelin] native delete sync failed", e));
  });
  tagDeleteSignalWired = true;
}

// wait for (and cache the lookup of) the tag layer's LocalAnnotationSource.
async function getTagSource(): Promise<any> {
  const viewer = kernel.getViewer();
  let src: any = null;
  for (let i = 0; i < 50; i++) {
    const layer: any = viewer?.layerManager?.getLayerByName(TAG_LAYER);
    src = layer?.layer?.localAnnotations;
    if (src) break;
    await sleep(100);
  }
  return src;
}

// add exactly one tag dot, without touching any other drawn tag -- used both by the full
// redraw below and by paintTagNode's incremental add (redrawing everything on every auto-tag
// during a paint pass would tear down/rebuild dozens of dots and flicker).
function addTagDot(src: any, tag: MyelinTag) {
  const resNm = kernel.getResNm();
  const p = tag.xyz_nm;
  src.add({
    type: 0, // AnnotationType.POINT
    id: tag.uuid,
    point: Float32Array.of(p[0] / resNm[0], p[1] / resNm[1], p[2] / resNm[2]),
    properties: [],
  });
  drawnTagIds.push(tag.uuid);
}

async function redrawTags(tags: MyelinTag[]) {
  const src = await getTagSource();
  if (!src) return;
  for (const id of drawnTagIds) {
    selfDeleting.add(id);
    try {
      const ref = src.getReference(id);
      src.delete(ref);
      ref.dispose?.();
    } catch {
      selfDeleting.delete(id); // never actually removed -- don't leave a stale guard entry
    }
  }
  drawnTagIds = [];
  for (const tag of tags) addTagDot(src, tag);
}

// fetch this branch's live tags and redraw them (authoritative full resync -- branch load,
// or after a `d`-key delete).
async function refreshTags(pid: number) {
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/myelin/tags?path_id=${pid}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    currentTags = data.tags || [];
    await redrawTags(currentTags);
  } catch (e) {
    console.warn("[myelin] refreshTags failed", e);
  }
}

function nearestNodeIndex(posNm: number[], nodesNm: number[][]): number {
  let best = 0;
  let bestD = Infinity;
  nodesNm.forEach((p, i) => {
    const d = Math.hypot(p[0] - posNm[0], p[1] - posNm[1], p[2] - posNm[2]);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

// auto-tag one node crossed while painting -- skips nodes already tagged (flying back and
// forth over a painted stretch shouldn't pile up duplicate tags at the same spot) and adds the
// new dot incrementally rather than doing a full refetch+redraw per node.
const PAINT_DEDUPE_EPS_NM = 10;
async function paintTagNode(pid: number, nodeXyzNm: number[]) {
  const alreadyTagged = currentTags.some(
    (t) => Math.hypot(t.xyz_nm[0] - nodeXyzNm[0], t.xyz_nm[1] - nodeXyzNm[1], t.xyz_nm[2] - nodeXyzNm[2]) < PAINT_DEDUPE_EPS_NM,
  );
  if (alreadyTagged) return;
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/myelin/tag`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ xyz_nm: nodeXyzNm, path_id: pid }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    const tag: MyelinTag = { uuid: resp.uuid, xyz_nm: resp.xyz_nm, path_id: pid };
    currentTags.push(tag);
    const src = await getTagSource();
    if (src) addTagDot(src, tag);
    status(resp.warning ? `painting... (${resp.warning})` : "painting...", resp.warning ? "warn" : "ok");
  } catch (e) {
    console.warn("[myelin] paintTagNode failed", e);
  }
}

// tag the node nearest the cursor as myelinated (key `t`).
async function tagNodeAtCursor() {
  const viewer = kernel.getViewer();
  const resNm = kernel.getResNm();
  const pid = kernel.getCurrentPid();
  const ms = viewer?.mouseState;
  if (!ms?.active || !ms.position || ms.position.length < 3 || pid === null) {
    status("hover over a skeleton node, then press t to tag it myelinated", "warn");
    return;
  }
  const posNm = [ms.position[0] * resNm[0], ms.position[1] * resNm[1], ms.position[2] * resNm[2]];
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/myelin/tag`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ xyz_nm: posNm, path_id: pid }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    status(
      resp.warning ? `tagged myelinated (${resp.warning})` : "tagged myelinated",
      resp.warning ? "warn" : "ok",
    );
    await refreshTags(pid);
  } catch (e) {
    status(`tag failed: ${e}`, "warn");
  }
}

// delete the tag nearest the cursor (key `d`) -- mirrors main.ts's delete-nearest for tags.
// (Deleting via neuroglancer's native Annotations panel also works -- see wireTagDeleteSignal.)
async function deleteNearestTag() {
  const viewer = kernel.getViewer();
  const resNm = kernel.getResNm();
  const pid = kernel.getCurrentPid();
  const ms = viewer?.mouseState;
  if (!ms?.active || !ms.position || ms.position.length < 3) {
    status("hover over a tag, then press d to delete it", "warn");
    return;
  }
  const cx = ms.position[0] * resNm[0],
    cy = ms.position[1] * resNm[1],
    cz = ms.position[2] * resNm[2];
  let best: string | null = null;
  let bestD = Infinity;
  for (const tag of currentTags) {
    const d = Math.hypot(tag.xyz_nm[0] - cx, tag.xyz_nm[1] - cy, tag.xyz_nm[2] - cz);
    if (d < bestD) {
      bestD = d;
      best = tag.uuid;
    }
  }
  const THRESH_NM = 2500; // matches main.ts's delete-nearest threshold
  if (!best || bestD > THRESH_NM) {
    status(
      currentTags.length ? `no tag near cursor (nearest ${Math.round(bestD)} nm)` : "no tags on this branch",
      "warn",
    );
    return;
  }
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/myelin/tag/${best}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    status(`deleted tag (${Math.round(bestD)} nm away)`, "ok");
    if (pid !== null) await refreshTags(pid);
  } catch (e) {
    status(`delete failed: ${e}`, "warn");
  }
}

// Buffer depth: how much loaded EM sits ahead of the camera, and whether that's currently holding
// the fly-through back. Worth showing rather than hiding -- if the camera is crawling you want to
// know it's waiting on chunks (and not, say, that the speed slider moved).
function renderBufferDepth(phase: "buffer" | "play") {
  const d = kernel.getBufferDepth();
  const el = $("bufdepth");
  if (phase !== "play" || d.targetNm <= 0) {
    el.textContent = "";
    return;
  }
  const pct = Math.round((d.aheadNm / d.targetNm) * 100);
  const slowed = d.speedFraction < 0.95;
  el.textContent = slowed
    ? `buffer ${pct}% -- slowed to ${Math.round(d.speedFraction * 100)}%${d.precise ? "" : " (approx)"}`
    : `buffer ${pct}%`;
  el.className = slowed ? "warn" : "";
}

function onProgress(frac: number, phase: "buffer" | "play") {
  $("progresspct").textContent = phase === "buffer" ? "buffering" : `${(frac * 100).toFixed(0)}%`;
  ($("progress") as HTMLInputElement).value = String(Math.round(frac * 1000));
  renderBufferDepth(phase);
  if (paintMode && phase === "play") {
    const pos = kernel.getCurrentPositionNm();
    const pid = kernel.getCurrentPid();
    if (pos && pid !== null && currentNodesNm.length) {
      const idx = nearestNodeIndex(pos, currentNodesNm);
      if (idx !== lastPaintedNodeIdx) {
        lastPaintedNodeIdx = idx;
        paintTagNode(pid, currentNodesNm[idx]);
      }
    }
  }
}

// M4.1-style whole-cell skeleton: a skeleton-ONLY segmentation layer, so it renders as a mesh in
// the 3D panel and is nearly invisible in the 2D cross-section. Must be in the INITIAL viewer
// state (hence onBuildViewerState, not onRankReady), and its graphene credentials must already be
// registered -- both handled by fetching live-sources before the first loadBranch in main().
function buildViewerState(state: any): any {
  if (!live?.skeleton_source) return state;
  state.layers.push({
    type: "segmentation",
    name: "skeleton",
    source: live.skeleton_source,
    segments: [live.root_id],
    selectedAlpha: 0.2, // 2D cross-section opacity -- keep the EM readable
    objectAlpha: 0.8, // 3D mesh opacity
    meshSilhouetteRendering: 1.7,
  });
  state.layout = "xy-3d"; // 2D fly-through + 3D whole-cell context, side by side
  state.projectionScale = 10000; // frame the whole arbor in 3D (independent of the 2D zoom)
  return state;
}

const kernel = createFlyKernel({
  api: API,
  rootId: ROOT_ID,
  compartment: "axon", // scopes background pre-build to the myelin tool's own axon sequence
  onStatus: status,
  onProgress,
  onBuildViewerState: buildViewerState,
  onRankReady: () => {
    addSkelMarkerLayers(); // added first so the tag layer renders on top of the markers
    addTagLayer();
    drawSkeletonFeatures(); // whole-cell branch/end markers (once; positions are root-invariant)
  },
});

// `flyCache()` in the devtools console: dumps cache pressure + resident/expired chunk counts, to
// tell a look-ahead problem from an eviction problem when a branch keeps slowing down.
// `__fly` exposes the kernel for ad-hoc inspection (`__fly.getBufferDepth()`) the same way
// flykernel already publishes `window.viewer`.
(window as any).flyCache = () => kernel.logCacheDiagnostic();
(window as any).__fly = kernel;

async function loadBranchAndRefresh(pid: number) {
  setPaintMode(false); // leaving this branch -- don't carry painting into the next one
  try {
    const cam = await kernel.loadBranch(pid);
    currentNodesNm = cam.nodes_nm; // still the paint/tag snap targets, just no longer drawn
    ($("branch") as HTMLSelectElement).value = String(pid);
    updatePlayButton();
    await refreshTags(pid);
  } catch (e) {
    status(`branch load failed: ${e}`, "warn");
  }
}

// mark the current branch myelin-reviewed (durably), repaint the dropdown + summary, advance
// to the next to-review AXON branch.
async function markDone() {
  const pid = kernel.getCurrentPid();
  if (pid === null) return;
  setPaintMode(false); // finishing review of this branch implies leaving it
  status(`marking branch ${pid} myelin-reviewed...`);
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/branches/${pid}/myelin-done`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    renderBranches(resp.branches);
    renderMyelinSummary(resp.myelin_summary, resp.branches);
    const next = resp.next_path_id;
    if (next === null || next === undefined) {
      status(`branch ${pid} done -- all axon branches reviewed`, "ok");
    } else {
      status(`branch ${pid} done -- advancing to #${next}`, "ok");
      await loadBranchAndRefresh(next);
    }
  } catch (e) {
    status(`mark-done failed: ${e}`, "warn");
  }
}

const branchOption = (b: Branch) =>
  `<option value="${b.path_id}">#${b.path_id} -- ${b.myelin_state} -- ${b.n_nodes}n -- ${Math.round(b.dist_to_root_nm / 1000)}um${b.built ? " (built)" : ""}</option>`;

function renderBranches(branches: Branch[]) {
  const sel = $("branch") as HTMLSelectElement;
  const prev = sel.value;
  sel.innerHTML = branches.map(branchOption).join("");
  if (prev && branches.some((b) => String(b.path_id) === prev)) sel.value = prev;
}

// last-known myelin coverage counts, so the build-progress refresh can repaint the whole line
// without needing a fresh summary from the server.
let lastSummary: Record<string, number> = {};

function renderMyelinSummary(summary?: Record<string, number>, branches?: Branch[]) {
  if (summary) lastSummary = summary;
  if (branches) {
    builtCount = branches.filter((b) => b.built).length;
    branchCount = branches.length;
  }
  const cov = `to_review ${lastSummary.to_review ?? 0} -- covered ${lastSummary.covered ?? 0}`;
  const warm = branchCount ? ` | cached ${builtCount}/${branchCount}` : "";
  $("coverage").textContent = cov + warm;
}

// background warm-up progress (how many axon branches already have their tube built)
let builtCount = 0;
let branchCount = 0;

// A read that hasn't advanced a chunk in this long is almost certainly wedged rather than slow
// (chunks normally land every few hundred ms), so surface it instead of spinning silently.
const STALL_WARN_S = 25;

// render the live chunk-level caching bar for whatever branch is filling right now
function renderWarmStatus(st: { active?: any[]; in_flight?: number; chunks_cached?: number }) {
  const bar = $("warmbar") as HTMLElement;
  const fill = $("warmfill") as HTMLElement;
  const label = $("warmlabel") as HTMLElement;
  const active = st.active || [];
  const totalCached = st.chunks_cached ?? 0;
  if (!active.length) {
    bar.style.display = "none";
    label.textContent = st.in_flight
      ? `${st.in_flight} branch(es) queued -- ${totalCached} chunks cached`
      : totalCached
        ? `${totalCached} chunks cached`
        : "";
    label.className = "";
    return;
  }
  // Several fills can run at once (background warm-up + any on-demand /camera the user triggered
  // by switching branches). Show them individually: summing them would make the number DROP each
  // time one finishes and leaves the set, and each one's `done` also restarts at the em->tgt
  // phase switch. `chunks_cached` is the monotonic figure, so lead with that.
  const worstStall = Math.max(...active.map((a) => a.stalled_s));
  const detail = active
    .map((a) => `#${a.path_id} ${a.phase} ${a.done}/${a.total}`)
    .join(" · ");
  // the bar tracks the single furthest-along fill; it's a liveness cue, not a completion estimate
  const pct = Math.max(...active.map((a) => (a.total ? (a.done / a.total) * 100 : 0)));
  bar.style.display = "";
  fill.style.width = `${Math.round(pct)}%`;
  const stalled = worstStall >= STALL_WARN_S;
  fill.style.background = stalled ? "#ff5566" : "#7fa8ff";
  label.textContent = stalled
    ? `${totalCached} chunks cached -- STALLED ${Math.round(worstStall)}s (${detail})`
    : `${totalCached} chunks cached -- ${detail} -- ${st.in_flight} queued`;
  label.className = stalled ? "warn" : "";
}

// poll the branch list + live fill progress while the background warm-up runs, so `cached N/M`
// and the caching bar climb on their own. Stops once everything is built (or on error).
async function pollWarmProgress() {
  for (let i = 0; i < 5000; i++) {
    try {
      const sr = await fetch(`${API}/api/cells/${ROOT_ID}/warm-status`);
      if (!sr.ok) return;
      renderWarmStatus(await sr.json());
    } catch {
      return; // backend went away; stop polling rather than spamming
    }
    // the branch list is heavier, so refresh the built-count less often than the live bar
    if (i % 5 === 0) {
      try {
        const r = await fetch(`${API}/api/cells/${ROOT_ID}/branches?compartment=axon`);
        if (r.ok) {
          const data = await r.json();
          renderMyelinSummary(data.myelin_summary, data.branches);
        }
      } catch {
        /* transient; the next tick retries */
      }
    }
    if (branchCount && builtCount >= branchCount) return; // whole cell cached
    await sleep(2000);
  }
}

// Hand our CAVE token to the graphene skeleton source (mirrors main.ts's M3.2): the datasource
// asks the credentials manager for a "middleauthapp" provider; we answer with the token directly
// -- no OAuth popup. MUST run before the viewer is created (i.e. before the first loadBranch).
let middleauthRegistered = false;
function registerMiddleAuthToken(token: string) {
  if (middleauthRegistered) return;
  class MiddleAuthTokenProvider extends CredentialsProvider<any> {
    get = makeCredentialsGetter(async () => ({ tokenType: "Bearer", accessToken: token }));
  }
  registerDefaultCredentialsProvider("middleauthapp", () => new MiddleAuthTokenProvider());
  middleauthRegistered = true;
}

async function main() {
  const cellInput = $("cellid") as HTMLInputElement;
  cellInput.value = ROOT_ID;
  const loadCell = () => {
    const id = cellInput.value.trim();
    if (!id || id === ROOT_ID) return;
    const p = new URLSearchParams(location.search);
    p.set("root", id);
    location.search = p.toString();
  };
  ($("loadcell") as HTMLButtonElement).onclick = loadCell;
  cellInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadCell();
  });

  ($("toggle") as HTMLButtonElement).onclick = () => {
    kernel.togglePlay();
    updatePlayButton();
  };
  ($("markdone") as HTMLButtonElement).onclick = () => markDone();
  ($("speed") as HTMLInputElement).oninput = (e) =>
    kernel.setSpeed(parseFloat((e.target as HTMLInputElement).value));

  const prog = $("progress") as HTMLInputElement;
  prog.addEventListener("pointerdown", () => kernel.setScrubbing(true));
  prog.addEventListener("pointerup", () => kernel.setScrubbing(false));
  prog.addEventListener("input", () => kernel.scrubTo(parseFloat(prog.value) / 1000));

  window.addEventListener(
    "keydown",
    (e) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === " ") {
        e.preventDefault();
        e.stopImmediatePropagation();
        kernel.togglePlay();
        updatePlayButton();
        return;
      }
      if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        e.stopImmediatePropagation();
        tagNodeAtCursor();
        return;
      }
      if (e.key === "x" || e.key === "X") {
        e.preventDefault();
        e.stopImmediatePropagation();
        markDone();
        return;
      }
      if (e.key === "d" || e.key === "D" || e.key === "Backspace" || e.key === "Delete") {
        e.preventDefault();
        e.stopImmediatePropagation();
        deleteNearestTag();
        return;
      }
      if (e.key === "p" || e.key === "P") {
        e.preventDefault();
        e.stopImmediatePropagation();
        setPaintMode(!paintMode);
        return;
      }
    },
    true,
  );

  status(`opening cell ${ROOT_ID}...`);
  try {
    const hr = await fetch(`${API}/api/cells`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      // warm_compartment kicks off a background tube build for EVERY remaining to-review axon
      // branch, so later branches are already cached by the time the sweep reaches them.
      body: JSON.stringify({ root_id: ROOT_ID, datastack: DATASTACK, warm_compartment: "axon" }),
    });
    if (!hr.ok) throw new Error(`HTTP ${hr.status}`);
  } catch (e) {
    status(
      `couldn't open cell -- is the backend up? (uv run --extra em --extra serve python -m proofreading.em.serve)  ${e}`,
      "warn",
    );
    return;
  }

  // whole-cell 3D skeleton context. Non-fatal: without it the tool still works, just 2D-only.
  // Must complete BEFORE the first loadBranch (which constructs the viewer via buildViewerState).
  try {
    const lr = await fetch(`${API}/api/cells/${ROOT_ID}/live-sources`);
    if (lr.ok) live = await lr.json();
  } catch (e) {
    console.warn("[myelin] live-sources fetch failed (2D only)", e);
  }
  if (live?.token) registerMiddleAuthToken(live.token);

  let branches: Branch[];
  try {
    const br = await fetch(`${API}/api/cells/${ROOT_ID}/branches?compartment=axon`);
    if (!br.ok) throw new Error(`HTTP ${br.status}`);
    const data = await br.json();
    branches = data.branches;
    renderMyelinSummary(data.myelin_summary || {}, branches);
  } catch (e) {
    status(`couldn't list axon branches: ${e}`, "warn");
    return;
  }
  if (!branches.length) {
    status("no axon-compartment branches found on this cell", "warn");
    return;
  }
  renderBranches(branches);
  ($("branch") as HTMLSelectElement).onchange = () =>
    loadBranchAndRefresh(parseInt(($("branch") as HTMLSelectElement).value, 10));

  // resume where myelin review left off, NOT the error-review coverage's to-review branch
  // (that's a different, irrelevant dimension here -- see the Branch interface comment)
  // Start the caching poller BEFORE the first branch load, NOT after: loadBranchAndRefresh awaits
  // /camera, which builds that branch's tube server-side and can take minutes -- exactly the
  // window the progress bar exists for. Fire-and-forget so it runs during that wait.
  pollWarmProgress();

  const first = (branches.find((b) => b.myelin_state === "to_review") || branches[0]).path_id;
  await loadBranchAndRefresh(first);
}

main();
