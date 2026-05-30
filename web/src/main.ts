// Phase 0 spike — browser-native fly-through.
//
// Hypothesis under test: an embedded neuroglancer whose camera is animated entirely
// CLIENT-SIDE (a requestAnimationFrame loop mutating `navigationState.pose.position`)
// glides smoothly over a served local tube AND does NOT degrade over a long session —
// the thing the python<->neuroglancer 30fps full-state-sync path could not do.
//
// There is NO python in the animation loop. Python only (a) served the precomputed tube
// and (b) dumped the camera path as static JSON (see web/spike_export.py). Everything
// below runs in the browser.

import "neuroglancer/unstable/ui/default_viewer.css";
// Side-effect registration of layer types (image/segmentation), datasources (precomputed,
// ...) and kvstores (http, gcs, s3, ...). `default_viewer_setup` builds only the viewer
// SHELL — without this, layers load but have no renderer/data backend (image never shows).
import "neuroglancer/unstable/main_module.js";
import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";

// Red overlay for the target-mask layer — mirrors proofreading/em/tube.py `_TINT`.
const TINT = `void main() {
  float v = toNormalized(getDataValue());
  emitRGBA(vec4(1.0, 0.2, 0.2, v > 0.5 ? 0.6 : 0.0));
}`;

interface CameraPath {
  resolution_nm: [number, number, number];
  em_source: string;
  tgt_source: string;
  points_nm: [number, number, number][];
  root_id: number;
  paths: number[];
}

const $ = (id: string) => document.getElementById(id)!;
const status = (msg: string) => ($("status").textContent = msg);

async function main() {
  let data: CameraPath;
  try {
    const resp = await fetch("/camera_path.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (e) {
    status(
      `✗ couldn't load /camera_path.json — run:  uv run --extra em python web/spike_export.py  (${e})`,
    );
    return;
  }

  const res = data.resolution_nm;
  // camera path in absolute voxels (nm / resolution) — neuroglancer's position space.
  const ptsVox = data.points_nm.map(
    (p) => [p[0] / res[0], p[1] / res[1], p[2] / res[2]] as [number, number, number],
  );
  if (ptsVox.length < 2) {
    status("✗ camera path has < 2 points");
    return;
  }

  // cumulative arc length in nm, so the speed slider is physical (nm/s).
  const cum = [0];
  for (let i = 1; i < data.points_nm.length; i++) {
    const a = data.points_nm[i - 1];
    const b = data.points_nm[i];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    cum.push(cum[i - 1] + d);
  }
  const totalArc = cum[cum.length - 1];

  // position (voxels) at arc length s (nm) along the polyline.
  const posAt = (s: number): [number, number, number] => {
    s = Math.max(0, Math.min(totalArc, s));
    // binary search the segment containing s
    let lo = 0,
      hi = cum.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] <= s) lo = mid;
      else hi = mid;
    }
    const seg = cum[hi] - cum[lo] || 1;
    const f = (s - cum[lo]) / seg;
    const a = ptsVox[lo];
    const b = ptsVox[hi];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };

  // --- initial neuroglancer state (layers + coordinate space + start position) ---
  const start = ptsVox[0];
  const state = {
    dimensions: {
      x: [res[0] * 1e-9, "m"],
      y: [res[1] * 1e-9, "m"],
      z: [res[2] * 1e-9, "m"],
    },
    position: start,
    // zoom tight on the neurite so the cross-section stays inside the ~1 µm tube radius
    // (a wide FOV requests off-tube chunks that don't exist). Scroll in-viewer to adjust.
    crossSectionScale: 0.12,
    projectionScale: 6000,
    layers: [
      { type: "image", name: "em", source: data.em_source },
      { type: "image", name: "tgt", source: data.tgt_source, shader: TINT, opacity: 0.85 },
    ],
    layout: "xy",
    showDefaultAnnotations: false,
  };
  status("camera path ok — starting neuroglancer…");
  let viewer: any;
  try {
    viewer = setupDefaultViewer();
  } catch (e: any) {
    status(`✗ setupDefaultViewer threw: ${e?.stack || e?.message || e}`);
    console.error("[spike] setupDefaultViewer", e);
    return;
  }

  // load layers + coordinate space programmatically (robust; no URL-hash encoding issues)
  try {
    viewer.state.restoreState(state);
  } catch (e: any) {
    status(`✗ restoreState threw: ${e?.stack || e?.message || e}`);
    console.error("[spike] restoreState", e);
  }
  console.log("[spike] state applied:", state);
  (window as any).viewer = viewer; // for poking in DevTools

  // Raise chunk-cache limits + enable prefetch so the whole (small) per-branch tube can be
  // pre-loaded and *stay* cached. The smooth glide then replays cached chunks — neuroglancer
  // only finishes loading chunks while navigation is idle, so motion over *uncached* data
  // blanks; pre-caching removes the need to load anything during motion.
  try {
    const cq = viewer.dataContext.chunkQueueManager;
    cq.capacities.gpuMemory.sizeLimit.value = 3e9;
    cq.capacities.gpuMemory.itemLimit.value = 1e7;
    cq.capacities.systemMemory.sizeLimit.value = 6e9;
    cq.capacities.systemMemory.itemLimit.value = 1e7;
    cq.capacities.download.itemLimit.value = 32;
    cq.enablePrefetch.value = true;
    console.log("[spike] raised chunk-cache limits + prefetch on");
  } catch (e) {
    console.warn("[spike] could not raise cache limits", e);
  }

  // --- camera state machine: 'buffer' (pre-cache sweep) -> 'play' (smooth ping-pong) ---
  let phase: "buffer" | "play" = "buffer";
  let running = true; // play/pause within the 'play' phase
  let s = 0;
  let dir = 1; // ping-pong direction
  let speed = parseFloat(($("speed") as HTMLInputElement).value); // nm/s

  // fps / degradation tracking
  const t0 = performance.now();
  let last = t0;
  let frames = 0;
  let fpsSmooth = 0;
  let fpsMin = Infinity;

  const setPosition = (vox: [number, number, number]) => {
    try {
      const pos = viewer.navigationState.pose.position;
      const cur = pos.value;
      if (cur && cur.length === 3) pos.value = Float32Array.of(vox[0], vox[1], vox[2]);
    } catch {
      /* coordinate space not ready yet; try next frame */
    }
  };

  const fmt = (ms: number) => {
    const t = Math.floor(ms / 1000);
    return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
  };

  const frame = (now: number) => {
    const dt = (now - last) / 1000;
    last = now;
    frames++;

    if (phase === "play" && running && dt > 0) {
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

    // smoothed fps; only track the min during playback, after a short warmup
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
  requestAnimationFrame(frame);

  // --- buffering pre-pass: step along the branch with short idle dwells so neuroglancer
  // loads each frustum's chunks into its cache, then hand off to the smooth glide. ---
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const stepNm = data.step_nm || 500;
  const coverStep = Math.max(1, Math.round(1200 / stepNm)); // ~frustum coverage between dwells
  const stepNodes = Math.max(coverStep, Math.ceil(ptsVox.length / 90)); // cap ~90 dwell stops
  (async () => {
    for (let i = 0; i < ptsVox.length; i += stepNodes) {
      setPosition(ptsVox[i]);
      const pct = Math.round((i / Math.max(1, ptsVox.length - 1)) * 100);
      $("status").className = "";
      $("status").textContent = `buffering ${pct}% — caching branch into neuroglancer…`;
      await sleep(130);
    }
    setPosition(ptsVox[0]);
    await sleep(500); // let the start settle before playback
    phase = "play";
    $("status").className = "ok";
    $("status").textContent = `buffered ${ptsVox.length} nodes — gliding (ping-pong). Watch fps + sharpness.`;
  })();

  // --- HUD controls ---
  const toggle = $("toggle") as HTMLButtonElement;
  toggle.onclick = () => {
    running = !running;
    toggle.textContent = running ? "⏸ pause cam" : "▶ play cam";
  };
  ($("speed") as HTMLInputElement).oninput = (e) => {
    speed = parseFloat((e.target as HTMLInputElement).value);
  };
  ($("resetmin") as HTMLButtonElement).onclick = () => {
    fpsMin = Infinity;
  };
}

main();
