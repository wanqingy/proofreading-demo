// Simplified proofreading recorder — a static embedded neuroglancer viewer (no fly-through,
// no tube cache, no branch camera) plus one "Record" button.
//
// Workflow: manually select all segments belonging to one neurite in the segmentation layer,
// optionally drop annotation points linked to a segment (with a description) to leave a note,
// then hit Record. It reads three things straight off the live viewer state and posts them in
// one shot to the backend, which assigns the neurite id (see proofreading/annotate/api.py):
//   1. visibleSegments from the segmentation layer's displayState.segmentationGroupState
//   2. every local annotation's relatedSegments + description (segment -> note)
//   3. a link built fresh from viewer.state.toJSON() at capture time (see buildLink below) --
//      NOT location.href, which neuroglancer's own UrlHashBinding only updates on a 200ms
//      debounce (400ms max-wait), so it can lag behind whatever was just selected/typed.

import "neuroglancer/unstable/ui/default_viewer.css";
import "neuroglancer/unstable/main_module.js";
import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";
import { makeLayer } from "neuroglancer/unstable/layer/index.js";

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "http://localhost:8001").replace(/\/$/, "");
// Default to the Brainmaps volume being tested (Google OAuth via the "Brain Maps" popup --
// see vite.config.ts for the client id). Override with ?img=&seg= for any other source.
const IMG_SOURCE =
  params.get("img") ||
  "brainmaps://73265790802:M808144_LH_S31a_s3_260305a_confocal:subvol1_raw";
const SEG_SOURCE =
  params.get("seg") ||
  "brainmaps://73265790802:M808144_LH_S31a_s3_260305a_confocal:subvol1_clahe3d_secgan297000_seg260713";

const SEG_LAYER = "seg";
const IMG_LAYER = "em";
const ANN_LAYER = "notes";

const $ = (id: string) => document.getElementById(id)!;
const status = (msg: string, cls = "") => {
  $("status").textContent = msg;
  $("status").className = cls;
};

// self-reported display name for attribution (not verified -- see auth discussion for a
// future hardening pass). Persisted so you only type it once per browser.
const USER_KEY = "proofreading-annotate-user";
function getUser(): string {
  return localStorage.getItem(USER_KEY) || "";
}
function setUser(name: string) {
  localStorage.setItem(USER_KEY, name);
}

let viewer: any = null;

function setupViewer() {
  // setupDefaultViewer() already restores state from the URL hash (via its internal
  // UrlHashBinding.updateFromUrlHash()) if this page was opened via a captured "open ↗"
  // link. Only seed the default em/seg layers on a FRESH visit -- otherwise this would
  // immediately clobber the just-restored captured state with the generic default.
  const freshVisit = !location.hash || location.hash === "#";
  viewer = setupDefaultViewer();
  (window as any).viewer = viewer;
  if (freshVisit) {
    viewer.state.restoreState({
      layers: [
        { type: "image", name: IMG_LAYER, source: IMG_SOURCE },
        { type: "segmentation", name: SEG_LAYER, source: SEG_SOURCE },
      ],
      layout: "xy-3d",
    });
  }

  // the local annotation layer needs a rank-3 global coordinate space (set once the image/
  // segmentation sources load) or a 3-D point overflows on render — same guard as main.ts.
  const rankWait = window.setInterval(() => {
    const v = viewer?.navigationState?.pose?.position?.value;
    if (v && v.length === 3) {
      window.clearInterval(rankWait);
      addAnnotationLayer();
    }
  }, 200);
}

let annAdded = false;
function addAnnotationLayer() {
  if (annAdded || !viewer) return;
  try {
    // a reopened capture link already has a "notes" layer restored from its own state --
    // don't create a second one.
    if (viewer.layerManager.getLayerByName(ANN_LAYER)) {
      annAdded = true;
      status("ready — select segments, then hit Record", "ok");
      return;
    }
    const managed = makeLayer(viewer.layerSpecification, ANN_LAYER, {
      type: "annotation",
      source: "local://annotations",
      annotationColor: "#ffcc00",
    });
    viewer.layerManager.addManagedLayer(managed);
    annAdded = true;
    status("ready — select segments, then hit Record", "ok");
  } catch (e) {
    console.warn("[annotate] addAnnotationLayer failed", e);
  }
}

// --- capture: read visible segments + per-segment notes off the live viewer state ---
function captureVisibleSegments(): string[] {
  const layer: any = viewer.layerManager.getLayerByName(SEG_LAYER);
  const visible = layer?.layer?.displayState?.segmentationGroupState?.value?.visibleSegments;
  if (!visible) return [];
  return visible.toJSON(); // Uint64Set.toJSON() -> decimal string[]
}

function captureNotesBySegment(): Record<string, string> {
  const layer: any = viewer.layerManager.getLayerByName(ANN_LAYER);
  const src = layer?.layer?.localAnnotations;
  if (!src) return {};
  const notes: Record<string, string> = {};
  for (const ann of src) {
    const desc = ann.description;
    if (!desc) continue;
    const related = ann.relatedSegments as BigUint64Array[] | undefined;
    if (!related || !related.length) continue;
    for (const segId of related[0]) {
      notes[segId.toString()] = desc;
    }
  }
  return notes;
}

// mirrors neuroglancer's own UrlHashBinding.setUrlHash encoding (ui/url_hash_binding.js),
// but built synchronously from the current state -- not the debounced address-bar hash.
function encodeFragment(fragment: string): string {
  return encodeURI(fragment).replace(/[!'()*;,]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase());
}

function buildLink(): string {
  const json = viewer.state.toJSON();
  const str = JSON.stringify(json, (_key, value) => (typeof value === "bigint" ? value.toString() : value));
  const frag = encodeFragment(str);
  return `${location.origin}${location.pathname}${location.search}#!${frag}`;
}

async function postRecord(
  link: string,
  ids: Record<string, string>,
  resolutions: Record<string, string> = {},
  neuriteId?: number,
) {
  const r = await fetch(`${API}/api/record`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ link, ids, resolutions, neurite_id: neuriteId ?? null, user: getUser() }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// --- edit mode: reopening a chip's "update" button loads its captured state into THIS tab
// (via ?edit=<id> + its stored link's hash) and the Record button becomes "Update #<id>",
// so the next capture overwrites that neurite instead of allocating a new one.
let editingNeuriteId: number | null = null;

function setEditing(id: number | null) {
  editingNeuriteId = id;
  const btn = $("record") as HTMLButtonElement;
  const cancel = $("canceledit") as HTMLButtonElement;
  btn.textContent = id !== null ? `● Update #${id}` : "● Record";
  cancel.style.display = id !== null ? "" : "none";
}

function clearEditParam() {
  const url = new URL(location.href);
  url.searchParams.delete("edit");
  history.replaceState(null, "", url.toString());
}

async function recordNeurite() {
  const btn = $("record") as HTMLButtonElement;
  btn.disabled = true;
  try {
    const segIds = captureVisibleSegments();
    if (!segIds.length) {
      status("no segments selected — select segments in the seg layer first", "warn");
      return;
    }
    const notes = captureNotesBySegment();
    const ids: Record<string, string> = {};
    for (const seg of segIds) ids[seg] = notes[seg] || "";
    const link = buildLink();
    const targetId = editingNeuriteId ?? undefined;

    status(`recording ${segIds.length} segment${segIds.length === 1 ? "" : "s"}…`);
    let resp = await postRecord(link, ids, {}, targetId);

    // resolve any cross-neurite conflicts one at a time via a confirm dialog, then re-submit
    const resolutions: Record<string, string> = {};
    while (resp.status === "conflict") {
      for (const [segId, ownerId] of Object.entries(resp.conflicts)) {
        const keep = window.confirm(
          `Segment ${segId} is already recorded under neurite ${ownerId}.\n` +
            `Is this an intentionally shared segment? OK = keep (record under both), Cancel = drop it from this capture.`,
        );
        resolutions[segId] = keep ? "keep" : "drop";
      }
      resp = await postRecord(link, ids, resolutions, targetId);
    }

    status(`✓ recorded as neurite ${resp.neurite_id} — ${Object.keys(resp.ids).length} segments`, "ok");
    if (targetId !== undefined) {
      setEditing(null);
      clearEditParam();
    }
    await refreshNeuriteList();
  } catch (e) {
    status(`✗ record failed: ${e}`, "warn");
  } finally {
    btn.disabled = false;
  }
}

// --- review list: every recorded neurite, with a link back to its captured view ---
interface NeuriteSummary {
  id: number;
  link: string;
  seg_count: number;
  note_count: number;
  ts: string;
  user: string;
}

let neuriteMap: Record<number, NeuriteSummary> = {};

function selectedNeuriteId(): number | null {
  const sel = $("neuriteselect") as HTMLSelectElement;
  const v = sel.value;
  return v ? parseInt(v, 10) : null;
}

function renderNeuriteList(items: NeuriteSummary[]) {
  neuriteMap = {};
  for (const n of items) neuriteMap[n.id] = n;
  const sel = $("neuriteselect") as HTMLSelectElement;
  const prev = sel.value;
  const openBtn = $("neuriteopen") as HTMLButtonElement;
  const editBtn = $("neuriteedit") as HTMLButtonElement;
  const delBtn = $("neuritedelete") as HTMLButtonElement;

  if (!items.length) {
    sel.innerHTML = `<option>no neurites recorded yet</option>`;
    sel.disabled = true;
    openBtn.disabled = editBtn.disabled = delBtn.disabled = true;
    return;
  }
  // most recently recorded first (backend already returns this order)
  sel.innerHTML = items
    .map(
      (n) =>
        `<option value="${n.id}">#${n.id} -- ${n.seg_count}seg ${n.note_count}note -- ${n.ts}${n.user ? " -- " + n.user : ""}</option>`,
    )
    .join("");
  sel.disabled = false;
  openBtn.disabled = editBtn.disabled = delBtn.disabled = false;
  if (prev && items.some((n) => String(n.id) === prev)) sel.value = prev;
}

function openSelected() {
  const id = selectedNeuriteId();
  const n = id !== null ? neuriteMap[id] : null;
  if (!n || !n.link) {
    status(`✗ neurite ${id} has no captured link`, "warn");
    return;
  }
  window.open(n.link, "_blank", "noopener");
}

function editSelected() {
  const id = selectedNeuriteId();
  const n = id !== null ? neuriteMap[id] : null;
  if (!n || !n.link) {
    status(`✗ neurite ${id} has no captured link to reopen for editing`, "warn");
    return;
  }
  const url = new URL(n.link);
  url.searchParams.set("edit", String(id));
  location.href = url.toString();
}

async function deleteSelected() {
  const id = selectedNeuriteId();
  const n = id !== null ? neuriteMap[id] : null;
  if (id === null || !n) return;
  const who = n.user ? ` (recorded by ${n.user})` : "";
  if (!window.confirm(`Delete neurite #${id}${who}? This cannot be undone.`)) return;
  try {
    const r = await fetch(`${API}/api/neurites/${id}?user=${encodeURIComponent(getUser())}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (editingNeuriteId === id) {
      setEditing(null);
      clearEditParam();
    }
    await refreshNeuriteList();
  } catch (e) {
    status(`✗ delete failed: ${e}`, "warn");
  }
}

async function refreshNeuriteList() {
  try {
    const r = await fetch(`${API}/api/neurites`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderNeuriteList(data.neurites || []);
  } catch (e) {
    console.warn("[annotate] refreshNeuriteList failed", e);
  }
}

async function main() {
  const userInput = $("username") as HTMLInputElement;
  userInput.value = getUser();
  userInput.addEventListener("change", () => setUser(userInput.value.trim()));

  ($("record") as HTMLButtonElement).onclick = () => recordNeurite();
  ($("help") as HTMLButtonElement).onclick = () => $("legend").classList.toggle("shown");
  ($("canceledit") as HTMLButtonElement).onclick = () => {
    setEditing(null);
    clearEditParam();
  };
  ($("neuriteopen") as HTMLButtonElement).onclick = openSelected;
  ($("neuriteedit") as HTMLButtonElement).onclick = editSelected;
  ($("neuritedelete") as HTMLButtonElement).onclick = deleteSelected;

  const editParam = params.get("edit");
  if (editParam) setEditing(parseInt(editParam, 10));

  status("loading viewer…");
  setupViewer();
  await refreshNeuriteList();
}

main();
