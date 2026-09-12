// Proofreading recorder -- injectable version for pages we DON'T control (e.g. the public
// https://neuroglancer-demo.appspot.com), where our own OAuth client can't get BrainMaps
// access but the page's own already-authenticated session can.
//
// Usage: paste this whole file into the browser DevTools console on a neuroglancer page that
// has already loaded (so `window.viewer` exists), and press Enter. Best saved as a Chrome
// DevTools "Snippet" (Sources tab -> Snippets -> New snippet -> paste -> Ctrl+Enter to run
// anytime) so you don't have to re-paste it every session.
//
// NOT delivered as a `<script src="http://localhost:.../ng-recorder.js">` tag -- that would be
// blocked as mixed content on an https:// page fetching an http:// dev-server script. Paste-to-
// console (or a DevTools Snippet) runs directly in the page's JS context, no network fetch.
//
// Reads the same three things our own web/src/annotate.ts does, but by DUCK-TYPING layers
// (not fixed names, since we don't control layer names on someone else's page) and unioning
// across every matching layer found:
//   1. visibleSegments from every segmentation-type layer
//   2. every local annotation's relatedSegments + description, from every annotation layer
//   3. a link built fresh from viewer.state.toJSON() at capture time
// POSTs to the same backend as annotate.ts (proofreading/annotate/api.py), CORS-open by design.

(function () {
  if (window.__ngRecorderInjected) {
    const el = document.getElementById("ng-recorder-hud");
    if (el) el.scrollIntoView?.({ block: "nearest" });
    return;
  }
  window.__ngRecorderInjected = true;

  const API = window.__ngRecorderApi || "http://localhost:8001";
  const viewer = window.viewer;
  if (!viewer) {
    alert("[ng-recorder] window.viewer not found -- wait for the page to finish loading, then rerun.");
    window.__ngRecorderInjected = false;
    return;
  }

  const style = document.createElement("style");
  style.textContent = `
    #ng-recorder-hud { position:fixed; left:10px; right:10px; bottom:10px; z-index:2147483647;
      display:flex; align-items:center; gap:10px; background:rgba(0,0,0,0.85); color:#d7e3ff;
      padding:6px 10px; border:1px solid #2c3a5a; border-radius:8px;
      font:12px ui-monospace,SFMono-Regular,Menlo,monospace; height:34px; user-select:none; }
    #ng-recorder-hud h1 { font-size:11px; margin:0; color:#7fa8ff; letter-spacing:.04em; white-space:nowrap; }
    #ng-recorder-hud button { background:#1c2740; color:#d7e3ff; border:1px solid #3a4d78; border-radius:5px;
      padding:4px 10px; cursor:pointer; font:inherit; white-space:nowrap; }
    #ng-recorder-hud button:hover { background:#28365a; }
    #ng-recorder-hud button:disabled { opacity:.5; cursor:default; }
    #ng-recorder-user { width:90px; flex:none; background:#1c2740; color:#d7e3ff; border:1px solid #3a4d78;
      border-radius:5px; padding:4px 6px; font:inherit; font-size:11px; }
    #ng-recorder-record { font-weight:bold; background:#1a3a22; border-color:#3a784a; color:#8aff9c; }
    #ng-recorder-status { color:#9fb0d0; font-size:11px; white-space:nowrap; max-width:260px;
      overflow:hidden; text-overflow:ellipsis; }
    #ng-recorder-status.ok { color:#8aff9c; }
    #ng-recorder-status.warn { color:#ff8a8a; }
    #ng-recorder-strip { display:flex; align-items:center; gap:6px; overflow-x:auto; flex:1;
      min-width:0; height:100%; }
    #ng-recorder-neurites { display:flex; align-items:center; gap:6px; flex:1; min-width:0; }
    #ng-recorder-select { flex:1; min-width:0; background:#1c2740; color:#d7e3ff; border:1px solid #3a4d78;
      border-radius:5px; padding:4px 6px; font:inherit; font-size:11px; }
    #ng-recorder-select:disabled { opacity:.5; }
    #ng-recorder-neurites button[data-action="delete"] { color:#ff8a8a; }
    #ng-recorder-showallpanel { display:none; position:fixed; right:10px; bottom:52px; z-index:2147483647;
      background:rgba(0,0,0,0.9); color:#7f8db0; border:1px solid #2c3a5a; border-radius:8px;
      padding:10px 12px; font-size:11px; line-height:1.5; width:340px;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    #ng-recorder-showallpanel.shown { display:block; }
    #ng-recorder-showallcount { color:#8aff9c; margin-bottom:6px; }
    #ng-recorder-showalllink { user-select:text; color:#d7e3ff; overflow-wrap:anywhere; max-height:30vh;
      overflow-y:auto; border-top:1px solid #2c3a5a; border-bottom:1px solid #2c3a5a; padding:6px 0;
      margin-bottom:6px; }
    #ng-recorder-showallfoot { display:flex; gap:6px; justify-content:flex-end; }
  `;
  document.head.appendChild(style);

  const hud = document.createElement("div");
  hud.id = "ng-recorder-hud";
  hud.innerHTML = `
    <h1>PROOFREADING</h1>
    <input id="ng-recorder-user" placeholder="your name" title="attributed on each recording (not verified)" />
    <button id="ng-recorder-record">[REC] Record</button>
    <button id="ng-recorder-cancel" style="display:none;color:#ff8a8a;border-color:#6a3a3a">cancel edit</button>
    <div id="ng-recorder-status">ready</div>
    <div style="width:1px;align-self:stretch;background:#2c3a5a"></div>
    <div id="ng-recorder-neurites">
      <select id="ng-recorder-select" disabled><option>no neurites recorded yet</option></select>
      <button id="ng-recorder-open" data-action="open" title="open in new tab">open</button>
      <button id="ng-recorder-edit" data-action="edit" title="edit in this tab">edit</button>
      <button id="ng-recorder-delete" data-action="delete" title="delete">del</button>
      <button id="ng-recorder-showall" title="open a NG view with every recorded segment highlighted" disabled>show all</button>
    </div>
  `;
  document.body.appendChild(hud);

  const showallPanel = document.createElement("div");
  showallPanel.id = "ng-recorder-showallpanel";
  showallPanel.innerHTML = `
    <div id="ng-recorder-showallcount"></div>
    <div id="ng-recorder-showalllink"></div>
    <div id="ng-recorder-showallfoot">
      <button id="ng-recorder-showallcopy" title="copy this link">copy link</button>
      <button id="ng-recorder-showallreopen" title="open in new tab">open &#8599;</button>
    </div>
  `;
  document.body.appendChild(showallPanel);

  const $ = (id) => document.getElementById(id);
  const setStatus = (msg, cls) => {
    const el = $("ng-recorder-status");
    el.textContent = msg;
    el.className = cls || "";
  };

  // self-reported display name for attribution (not verified). Persisted per-origin, so it's
  // remembered across re-injections on this same site but separate from our own app's storage.
  const USER_KEY = "proofreading-annotate-user";
  const getUser = () => localStorage.getItem(USER_KEY) || "";
  const setUserPref = (name) => localStorage.setItem(USER_KEY, name);

  // duck-type layers by behavior, not name/constructor (production builds are minified) --
  // union across every matching layer found, since we don't control how many there are here.
  function findLayers() {
    const segLayers = [];
    const annLayers = [];
    for (const ml of viewer.layerManager.managedLayers) {
      const layer = ml.layer;
      if (!layer) continue;
      if (layer.displayState && layer.displayState.segmentationGroupState) segLayers.push(layer);
      if (layer.localAnnotations) annLayers.push(layer);
    }
    return { segLayers, annLayers };
  }

  function captureVisibleSegments(segLayers) {
    const ids = new Set();
    for (const layer of segLayers) {
      const visible = layer.displayState.segmentationGroupState.value.visibleSegments;
      if (!visible) continue;
      for (const id of visible.toJSON()) ids.add(id);
    }
    return [...ids];
  }

  function captureNotesBySegment(annLayers) {
    // a segment can be linked from more than one annotation point (across layers too) --
    // concatenate all their notes instead of the last one silently clobbering the rest.
    const bySeg = {};
    for (const layer of annLayers) {
      const src = layer.localAnnotations;
      if (!src) continue;
      for (const ann of src) {
        const desc = ann.description;
        if (!desc) continue;
        const related = ann.relatedSegments;
        if (!related || !related.length) continue;
        for (const segId of related[0]) {
          const key = segId.toString();
          (bySeg[key] || (bySeg[key] = [])).push(desc);
        }
      }
    }
    const notes = {};
    for (const [seg, list] of Object.entries(bySeg)) notes[seg] = list.join("; ");
    return notes;
  }

  // mirrors neuroglancer's own UrlHashBinding.setUrlHash encoding, built synchronously from
  // the current state at capture time (not the debounced address-bar hash).
  function encodeFragment(fragment) {
    return encodeURI(fragment).replace(/[!'()*;,]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase());
  }

  function buildLink(mutate) {
    const json = viewer.state.toJSON();
    if (mutate) mutate(json);
    const str = JSON.stringify(json, (_k, v) => (typeof v === "bigint" ? v.toString() : v));
    const frag = encodeFragment(str);
    return `${location.origin}${location.pathname}${location.search}#!${frag}`;
  }

  async function postRecord(link, ids, resolutions, neuriteId) {
    const r = await fetch(`${API}/api/record`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ link, ids, resolutions: resolutions || {}, neurite_id: neuriteId ?? null, user: getUser() }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  let editingNeuriteId = null;
  function setEditing(id) {
    editingNeuriteId = id;
    $("ng-recorder-record").textContent = id != null ? `[REC] Update #${id}` : "[REC] Record";
    $("ng-recorder-cancel").style.display = id != null ? "" : "none";
  }

  async function recordNeurite() {
    const btn = $("ng-recorder-record");
    btn.disabled = true;
    try {
      const { segLayers, annLayers } = findLayers();
      if (!segLayers.length) {
        setStatus("no segmentation layer found on this page", "warn");
        return;
      }
      const segIds = captureVisibleSegments(segLayers);
      if (!segIds.length) {
        setStatus("no segments selected -- select segments first", "warn");
        return;
      }
      const notes = captureNotesBySegment(annLayers);
      const ids = {};
      for (const seg of segIds) ids[seg] = notes[seg] || "";
      const link = buildLink();
      const targetId = editingNeuriteId ?? undefined;

      setStatus(`recording ${segIds.length} segment${segIds.length === 1 ? "" : "s"}...`);
      let resp = await postRecord(link, ids, {}, targetId);
      const resolutions = {};
      while (resp.status === "conflict") {
        for (const [segId, ownerId] of Object.entries(resp.conflicts)) {
          const keep = window.confirm(
            `Segment ${segId} is already recorded under neurite ${ownerId}.\n` +
              `Is this an intentionally shared segment? OK = keep, Cancel = drop.`,
          );
          resolutions[segId] = keep ? "keep" : "drop";
        }
        resp = await postRecord(link, ids, resolutions, targetId);
      }
      setStatus(`OK: recorded as neurite ${resp.neurite_id} -- ${Object.keys(resp.ids).length} segments`, "ok");
      if (targetId !== undefined) setEditing(null);
      await refreshList();
    } catch (e) {
      setStatus(`FAIL: record failed: ${e}`, "warn");
    } finally {
      btn.disabled = false;
    }
  }

  let neuriteMap = {};
  // whether the last refreshList() actually succeeded -- so showAll() never opens a link that
  // (falsely) highlights nothing just because the backend happened to be down.
  let neuritesLoaded = false;

  function selectedNeuriteId() {
    const sel = $("ng-recorder-select");
    return sel.value ? parseInt(sel.value, 10) : null;
  }

  function renderList(items) {
    neuriteMap = {};
    for (const n of items) neuriteMap[n.id] = n;
    const sel = $("ng-recorder-select");
    const prev = sel.value;
    const openBtn = $("ng-recorder-open");
    const editBtn = $("ng-recorder-edit");
    const delBtn = $("ng-recorder-delete");
    const showAllBtn = $("ng-recorder-showall");
    if (!items.length) {
      sel.innerHTML = "<option>no neurites recorded yet</option>";
      sel.disabled = true;
      openBtn.disabled = editBtn.disabled = delBtn.disabled = showAllBtn.disabled = true;
      return;
    }
    sel.innerHTML = items
      .map(
        (n) =>
          `<option value="${n.id}">#${n.id} -- ${n.seg_count}seg ${n.note_count}note -- ${n.ts}${n.user ? " -- " + n.user : ""}</option>`,
      )
      .join("");
    sel.disabled = false;
    openBtn.disabled = editBtn.disabled = delBtn.disabled = showAllBtn.disabled = false;
    if (prev && items.some((n) => String(n.id) === prev)) sel.value = prev;
  }

  function openSelected() {
    const id = selectedNeuriteId();
    const n = id !== null ? neuriteMap[id] : null;
    if (!n || !n.link) {
      setStatus(`FAIL: neurite ${id} has no captured link`, "warn");
      return;
    }
    window.open(n.link, "_blank", "noopener");
  }

  function editSelected() {
    const id = selectedNeuriteId();
    const n = id !== null ? neuriteMap[id] : null;
    if (!n || !n.link) {
      setStatus(`FAIL: neurite ${id} has no captured link`, "warn");
      return;
    }
    // this page IS the captured state's own origin -- navigating to its link reloads THIS
    // same page with that state restored. The injected HUD does not survive a full reload;
    // re-run this script afterward (it'll pick up __ngRecorderEdit and start in edit mode).
    const url = new URL(n.link);
    url.searchParams.set("__ngRecorderEdit", String(id));
    if (
      window.confirm(
        `Reopening neurite #${id} for editing requires a full page reload (this HUD doesn't ` +
          `survive that). After it reloads, RERUN this script (paste again, or your saved ` +
          `Snippet) to get the HUD back in "Update #${id}" mode. Continue?`,
      )
    ) {
      location.href = url.toString();
    }
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
      if (editingNeuriteId === id) setEditing(null);
      await refreshList();
    } catch (e) {
      setStatus(`FAIL: delete failed: ${e}`, "warn");
    }
  }

  async function refreshList() {
    try {
      const r = await fetch(`${API}/api/neurites`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      renderList(data.neurites || []);
      neuritesLoaded = true;
    } catch (e) {
      neuritesLoaded = false;
      setStatus("FAIL: neurite list unavailable -- backend down?", "warn");
      console.warn("[ng-recorder] refreshList failed", e);
    }
  }

  // --- "show all": one NG view with every recorded segment highlighted ---
  //
  // No new endpoint and no extra fetch: GET /api/neurites already returns the full
  // seg_id -> note map per neurite, so neuriteMap holds everything already.
  //
  // There is no declared list of TARGET segment ids in this repo, so "not annotated yet"
  // has no universe to diff against as a list. The viewer answers it instead: every
  // recorded segment selected (bright), everything else left unselected but faintly
  // visible via notSelectedAlpha -- the contrast IS the answer. No findLayers() duck-typing
  // needed here (unlike captureVisibleSegments) -- the state-JSON walk keys off
  // type === "segmentation", which is already name-agnostic.

  // seg ids are decimal uint64 strings -- up to 20 digits, so Number(a) - Number(b) loses
  // precision past 2^53, and a bare .sort() puts "10149" before "1041". Non-negative decimals
  // with no leading zeros order correctly by (digit count, then lexicographic).
  const bySegId = (a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);

  // dedupe across neurites: a cross-neurite conflict resolved "keep" is recorded under BOTH
  // (proofreading/annotate/api.py), so summing seg_count over-counts -- this is the distinct set.
  function allRecordedIds() {
    const ids = new Set();
    for (const n of Object.values(neuriteMap)) {
      for (const seg of Object.keys(n.ids || {})) ids.add(seg);
    }
    return [...ids].sort(bySegId);
  }

  function buildShowAllLink(ids) {
    return buildLink((state) => {
      for (const layer of state.layers || []) {
        if (layer.type !== "segmentation") continue;
        layer.segments = ids; // plain (unprefixed) ids -- "!"-prefixed means HIDDEN, not this
        // notSelectedAlpha defaults to 0 (fully invisible) -- floor it so unrecorded segments
        // are faintly visible instead of just black. Floor, don't overwrite: never dims a
        // higher value the user already had.
        layer.notSelectedAlpha = Math.max(layer.notSelectedAlpha ?? 0, 0.15);
      }
    });
  }

  function showAll() {
    if (!neuritesLoaded) {
      setStatus("FAIL: can't show all -- neurite list hasn't loaded (backend down?)", "warn");
      return;
    }
    const ids = allRecordedIds();
    if (!ids.length) {
      setStatus("no neurites recorded yet", "warn");
      return;
    }
    const link = buildShowAllLink(ids);
    window.open(link, "_blank", "noopener"); // new tab -- never touches this page's own selection
    $("ng-recorder-showallcount").textContent =
      `${ids.length} segment${ids.length === 1 ? "" : "s"} across ${Object.keys(neuriteMap).length} neurite${Object.keys(neuriteMap).length === 1 ? "" : "s"}`;
    $("ng-recorder-showalllink").textContent = link;
    showallPanel.classList.add("shown");
  }

  async function copyShowAllLink() {
    const link = $("ng-recorder-showalllink").textContent || "";
    if (!link) return;
    try {
      // a programmatic clipboard write, NOT a synthetic "copy" event -- neuroglancer's own
      // document-level copy handler replaces the clipboard with the whole viewer-state JSON
      // unless a real text Range is already selected.
      await navigator.clipboard.writeText(link);
      setStatus("OK: copied link", "ok");
    } catch (e) {
      setStatus(`FAIL: copy failed: ${e}`, "warn");
    }
  }

  function reopenShowAllLink() {
    const link = $("ng-recorder-showalllink").textContent || "";
    if (link) window.open(link, "_blank", "noopener");
  }

  const userInput = $("ng-recorder-user");
  userInput.value = getUser();
  userInput.addEventListener("change", () => setUserPref(userInput.value.trim()));

  $("ng-recorder-record").onclick = recordNeurite;
  $("ng-recorder-cancel").onclick = () => setEditing(null);
  $("ng-recorder-open").onclick = openSelected;
  $("ng-recorder-edit").onclick = editSelected;
  $("ng-recorder-delete").onclick = deleteSelected;
  $("ng-recorder-showall").onclick = showAll;
  $("ng-recorder-showallcopy").onclick = copyShowAllLink;
  $("ng-recorder-showallreopen").onclick = reopenShowAllLink;

  const params = new URLSearchParams(location.search);
  const editParam = params.get("__ngRecorderEdit");
  if (editParam) setEditing(parseInt(editParam, 10));

  refreshList();
  setStatus("ready -- select segments, then hit Record", "ok");
  console.log("[ng-recorder] injected. API =", API);
})();
