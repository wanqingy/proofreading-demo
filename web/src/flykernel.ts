// Shared fly-through kernel: viewer bootstrap + arc-length camera animation + branch loading.
//
// Adapted from (not a live extraction of) web/src/main.ts's M1 camera/buffering machinery --
// main.ts is left completely untouched. This module intentionally owns ONLY the tag-agnostic
// mechanics (viewer construction, cache limits, the requestAnimationFrame loop, per-branch
// buffering/glide, camera fetch); it owns no DOM rendering and no annotation-layer specifics,
// since those differ per consumer (main.ts's merge/split/extend/question tags vs. myelin.ts's
// toggle track). A factory function (not module-level state) so each page gets its own instance
// without any of the ES-module live-binding awkwardness a true multi-consumer refactor would need.

import "neuroglancer/unstable/ui/default_viewer.css";
import "neuroglancer/unstable/main_module.js";
import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";

// Red overlay for the target-mask layer -- mirrors proofreading/em/tube.py `_TINT`.
const TINT = `void main() {
  float v = toNormalized(getDataValue());
  emitRGBA(vec4(1.0, 0.2, 0.2, v > 0.5 ? 0.6 : 0.0));
}`;

export interface Camera {
  path_id: number;
  root_id: string;
  resolution_nm: [number, number, number];
  points_nm: [number, number, number][];
  nodes_nm: [number, number, number][]; // TRUE sparse skeleton vertices (vs. resampled points_nm)
  orientations: number[][] | null;
  step_nm: number;
  em_source: string;
  tgt_source: string;
  build: { cached: boolean; seconds: number };
  prebuilding?: number[];
}

export interface FlyKernelOptions {
  api: string;
  rootId: string;
  onStatus?: (msg: string, cls?: string) => void;
  onProgress?: (frac: number, phase: "buffer" | "play") => void;
  // fired once the global coordinate space becomes rank-3 (i.e. after em/tgt load) -- a local
  // annotation layer created before then captures rank 0 and overflows on render.
  onRankReady?: () => void;
  // fired before a branch switch begins
  onBeforeLoadBranch?: (pid: number) => void | Promise<void>;
  // scopes the camera fetch's background pre-build to this compartment's own review sequence
  // (e.g. "axon" for the myelin tool) instead of the unfiltered, error-review-coverage default
  // -- see proofreading/em/service.py's _queue_prebuild docstring.
  compartment?: string;
  // last chance to extend/replace the viewer state before it is restored. Layers that must
  // exist at viewer-construction time (e.g. a live segmentation/skeleton source, which also
  // needs its credentials registered BEFORE setupDefaultViewer runs) belong here rather than
  // in onRankReady, which fires only after the viewer is already built.
  onBuildViewerState?: (state: any) => any;
}

export interface FlyKernel {
  getViewer(): any;
  getResNm(): [number, number, number];
  getCurrentPid(): number | null;
  getPhase(): "buffer" | "play";
  isRunning(): boolean;
  togglePlay(): void;
  setSpeed(v: number): void;
  scrubTo(frac: number): void; // 0..1 along the current branch
  setScrubbing(v: boolean): void;
  getCurrentPositionNm(): [number, number, number] | null;
  loadBranch(pid: number): Promise<Camera>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function createFlyKernel(opts: FlyKernelOptions): FlyKernel {
  const status = (msg: string, cls = "") => opts.onStatus?.(msg, cls);

  let viewer: any = null;
  let ptsVox: [number, number, number][] = [];
  let cum: number[] = [0];
  let totalArc = 0;
  let stepNm = 500;
  let resNm: [number, number, number] = [16, 16, 40];
  let s = 0; // current arc position (nm) along the branch
  let phase: "buffer" | "play" = "buffer";
  let running = true;
  let speed = 1000;
  let bufferToken = 0;
  let scrubbing = false;
  let currentPid: number | null = null;
  let last = performance.now();
  let frames = 0;
  let annRankNotified = false;

  const setPosition = (vox: [number, number, number]) => {
    try {
      const pos = viewer.navigationState.pose.position;
      const cur = pos.value;
      if (cur && cur.length === 3) pos.value = Float32Array.of(vox[0], vox[1], vox[2]);
    } catch {
      /* coordinate space not ready yet; try next frame */
    }
  };

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
        s = totalArc;
        running = false;
      }
      setPosition(posAt(s));
    }

    if (frames % 6 === 0 && !scrubbing) {
      const frac = totalArc > 0 ? s / totalArc : 0;
      opts.onProgress?.(frac, phase);
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
    const state = {
      dimensions: { x: [res[0] * 1e-9, "m"], y: [res[1] * 1e-9, "m"], z: [res[2] * 1e-9, "m"] },
      position: start,
      crossSectionScale: 0.2,
      projectionScale: 6000,
      layers: [
        { type: "image", name: "em", source: cam.em_source, shaderControls: { normalized: { range: [100, 155] } } },
        { type: "image", name: "tgt", source: cam.tgt_source, shader: TINT, opacity: 0.2 },
      ],
      layout: "xy",
      showDefaultAnnotations: false,
    };
    const finalState = opts.onBuildViewerState ? opts.onBuildViewerState(state) : state;
    viewer = setupDefaultViewer();
    viewer.state.restoreState(finalState);
    (window as any).viewer = viewer;

    // Cache limits -- bound the working set to roughly ONE branch so switching branches EVICTS
    // the previous one (LRU) rather than accumulating until the renderer process is OOM-killed.
    try {
      const cq = viewer.dataContext.chunkQueueManager;
      cq.capacities.gpuMemory.sizeLimit.value = 1e9;
      cq.capacities.gpuMemory.itemLimit.value = 1e6;
      cq.capacities.systemMemory.sizeLimit.value = 1.5e9;
      cq.capacities.systemMemory.itemLimit.value = 1e6;
      cq.capacities.download.itemLimit.value = 16;
      cq.enablePrefetch.value = false;
    } catch (e) {
      console.warn("[flykernel] could not set cache limits", e);
    }

    try {
      const canvas = viewer.display?.canvas as HTMLCanvasElement | undefined;
      canvas?.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        status("WebGL context lost (GPU memory) -- reload the page", "warn");
        console.error("[flykernel] webglcontextlost");
      });
    } catch {
      /* ignore */
    }
    requestAnimationFrame(frame);

    const rankWait = window.setInterval(() => {
      const v = viewer?.navigationState?.pose?.position?.value;
      if (v && v.length === 3) {
        window.clearInterval(rankWait);
        if (!annRankNotified) {
          annRankNotified = true;
          opts.onRankReady?.();
        }
      }
    }, 200);
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

  async function bufferAndPlay(pid: number) {
    const token = ++bufferToken;
    phase = "buffer";
    s = 0;
    const coverStep = Math.max(1, Math.round(1200 / stepNm));
    for (let i = 0; i < ptsVox.length; i += coverStep) {
      if (token !== bufferToken) return;
      setPosition(ptsVox[i]);
      const pct = Math.round((i / Math.max(1, ptsVox.length - 1)) * 100);
      status(`buffering branch ${pid} ${pct}% -- caching...`);
      await sleep(120);
    }
    if (token !== bufferToken) return;
    s = 0;
    setPosition(ptsVox[0]);
    await sleep(400);
    if (token !== bufferToken) return;
    phase = "play";
    running = false; // stay paused after buffering; caller presses play to start
    status(`branch ${pid}: ready -- ${ptsVox.length} nodes`, "ok");
  }

  async function loadBranch(pid: number): Promise<Camera> {
    await opts.onBeforeLoadBranch?.(pid);
    bufferToken++; // stop any current buffering immediately
    phase = "buffer";
    status(`branch ${pid}: fetching camera path (building tube if first visit)...`);
    const url = new URL(`${opts.api}/api/cells/${opts.rootId}/branches/${pid}/camera`);
    if (opts.compartment) url.searchParams.set("compartment", opts.compartment);
    const r = await fetch(url.toString());
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const cam: Camera = await r.json();
    if (!cam.points_nm || cam.points_nm.length < 2) {
      throw new Error(`branch ${pid} has <2 camera nodes`);
    }
    if (!viewer) setupViewer(cam);
    setBranch(cam);
    currentPid = pid;
    bufferAndPlay(pid);
    return cam;
  }

  return {
    getViewer: () => viewer,
    getResNm: () => resNm,
    getCurrentPid: () => currentPid,
    getPhase: () => phase,
    isRunning: () => running,
    togglePlay: () => {
      running = !running;
    },
    setSpeed: (v: number) => {
      speed = v;
    },
    scrubTo: (frac: number) => {
      if (phase !== "play" || totalArc <= 0) return;
      running = false;
      s = frac * totalArc;
      setPosition(posAt(s));
    },
    setScrubbing: (v: boolean) => {
      scrubbing = v;
    },
    getCurrentPositionNm: () => {
      try {
        const v = viewer?.navigationState?.pose?.position?.value;
        if (!v || v.length !== 3) return null;
        return [v[0] * resNm[0], v[1] * resNm[1], v[2] * resNm[2]];
      } catch {
        return null;
      }
    },
    loadBranch,
  };
}
