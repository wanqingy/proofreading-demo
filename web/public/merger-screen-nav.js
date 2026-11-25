// Merger-screening navigator -- injectable, for the batch links built by
// proofreading/annotate/merger_screen.py (paste into DevTools console on a loaded batch link,
// same usage as ng-recorder.js: Sources tab -> Snippets -> run with Ctrl+Enter).
//
// Each batch link has 200 segments, all selected-but-hidden ("!"-prefixed) in its segmentation
// layer -- nothing visible until this script's first advance. Neuroglancer has no built-in
// "next in list" command (only a per-row right-click -> moveToSegment, tedious at 200 rows), so
// this drives it directly: on each advance it hides the previous segment, shows only the new
// one, and recenters the camera (displayState.moveToSegment -- the same method a manual
// right-click calls).
//
// Order comes from decoding location.hash itself, NOT from iterating the live
// visibleSegments/selectedSegments -- those are backed by a hash table (Uint64Set), not an
// order-preserving structure, so the generator's size-descending order wouldn't survive a live
// re-read. Decoding the URL text directly sidesteps that.

(function () {
  if (window.__mergerNavInjected) {
    const el = document.getElementById("merger-nav-hud");
    if (el) el.scrollIntoView?.({ block: "nearest" });
    return;
  }
  window.__mergerNavInjected = true;

  const viewer = window.viewer;
  if (!viewer) {
    alert("[merger-nav] window.viewer not found -- wait for the page to finish loading, then rerun.");
    window.__mergerNavInjected = false;
    return;
  }

  // decode this page's own captured link the same way review_sheet.py's decode_link() does
  // (urllib.parse.unquote + json.loads there; decodeURIComponent + JSON.parse here -- same
  // percent-encoding, encodeURI on write is a strict subset decodeURIComponent fully reverses).
  function decodeOwnLink() {
    const hash = location.hash;
    if (!hash || !hash.startsWith("#!")) return null;
    try {
      return JSON.parse(decodeURIComponent(hash.slice(2)));
    } catch (e) {
      console.warn("[merger-nav] failed to decode location.hash", e);
      return null;
    }
  }

  const state = decodeOwnLink();
  const segLayerSpec = state && state.layers.find((l) => l.type === "segmentation");
  if (!segLayerSpec) {
    alert("[merger-nav] no segmentation layer found in this page's own link -- is this a batch link from merger_screen.py?");
    window.__mergerNavInjected = false;
    return;
  }
  // this is the AUTHORITATIVE, order-preserving id list -- strip the "!" (hidden) marker the
  // generator wrote; order here is exactly the size-descending order from the source CSV.
  const ids = segLayerSpec.segments.map((s) => s.replace(/^!/, ""));

  const segManagedLayer = viewer.layerManager.getLayerByName(segLayerSpec.name);
  if (!segManagedLayer || !segManagedLayer.layer) {
    alert(`[merger-nav] segmentation layer "${segLayerSpec.name}" not found in the live viewer -- has it finished loading?`);
    window.__mergerNavInjected = false;
    return;
  }
  const displayState = segManagedLayer.layer.displayState;
  const visibleSegments = displayState.segmentationGroupState.value.visibleSegments;

  const style = document.createElement("style");
  style.textContent = `
    #merger-nav-hud { position:fixed; right:10px; bottom:10px; z-index:2147483647;
      display:flex; align-items:center; gap:10px; background:rgba(0,0,0,0.85); color:#d7e3ff;
      padding:6px 10px; border:1px solid #2c3a5a; border-radius:8px;
      font:12px ui-monospace,SFMono-Regular,Menlo,monospace; height:34px; user-select:none; }
    #merger-nav-hud h1 { font-size:11px; margin:0; color:#7fa8ff; letter-spacing:.04em; white-space:nowrap; }
    #merger-nav-hud button { background:#1c2740; color:#d7e3ff; border:1px solid #3a4d78; border-radius:5px;
      padding:4px 10px; cursor:pointer; font:inherit; white-space:nowrap; }
    #merger-nav-hud button:hover { background:#28365a; }
    #merger-nav-hud button:disabled { opacity:.5; cursor:default; }
    #merger-nav-status { color:#9fb0d0; font-size:11px; white-space:nowrap; }
  `;
  document.head.appendChild(style);

  const hud = document.createElement("div");
  hud.id = "merger-nav-hud";
  hud.innerHTML = `
    <h1>MERGER SCREEN</h1>
    <button id="merger-nav-prev" title="previous segment ([)">&#8592; prev</button>
    <div id="merger-nav-status"></div>
    <button id="merger-nav-next" title="next segment (])">next &#8594;</button>
  `;
  document.body.appendChild(hud);

  const $ = (id) => document.getElementById(id);
  const statusEl = $("merger-nav-status");

  let index = -1; // -1 = nothing shown yet

  function render() {
    statusEl.textContent =
      index < 0 ? `not started -- press ] or "next" (${ids.length} segments)` : `${index + 1} / ${ids.length} -- seg ${ids[index]}`;
    $("merger-nav-prev").disabled = index <= 0;
    $("merger-nav-next").disabled = index >= ids.length - 1;
  }

  function advance(delta) {
    const newIndex = Math.max(0, Math.min(ids.length - 1, index + delta));
    if (newIndex === index) return;
    if (index >= 0) visibleSegments.delete(BigInt(ids[index])); // hide the one we're leaving
    index = newIndex;
    const id = ids[index];
    visibleSegments.add(BigInt(id)); // show only the new one
    displayState.moveToSegment(BigInt(id)); // recenter the camera, same as a manual right-click
    render();
  }

  $("merger-nav-prev").onclick = () => advance(-1);
  $("merger-nav-next").onclick = () => advance(1);

  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (e.key === "[") advance(-1);
    else if (e.key === "]") advance(1);
  });

  render();
  console.log(`[merger-nav] injected. ${ids.length} segments in this batch (${segLayerSpec.name}).`);
})();
