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
import {
  ChunkMemoryStatistics,
  ChunkState,
  getChunkStateStatisticIndex,
  numChunkMemoryStatistics,
} from "neuroglancer/unstable/chunk_manager/base.js";
import { startCrashWatch, type CrashWatch } from "./crashwatch";

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
  // May be empty at construction: the page can start with no cell open (nothing reviewed yet, or
  // resolving which cell was last worked on takes a fetch). Set it later with setRootId.
  rootId: string;
  onStatus?: (msg: string, cls?: string) => void;
  onProgress?: (frac: number, phase: "buffer" | "play") => void;
  // fired once the global coordinate space becomes rank-3 (i.e. after em/tgt load) -- a local
  // annotation layer created before then captures rank 0 and overflows on render.
  onRankReady?: () => void;
  // fired before a branch switch begins
  onBeforeLoadBranch?: (pid: number) => void | Promise<void>;
  // scopes the camera fetch's background pre-build to this compartment's own review sequence
  // ("axon" or "all" for the myelin tool) instead of the unfiltered, error-review-coverage default
  // -- see proofreading/em/service.py's _scope / _queue_prebuild docstrings. Like rootId, it may
  // not be known at construction (it's recorded per cell), so it can be set later.
  compartment?: string;
  // Neuroglancer's velocity-based prefetch (default on -- see setupViewer). Exposed so it can be
  // turned off WITHOUT a rebuild when diagnosing a renderer kill: prefetch adds download/decode
  // churn on top of the chunk caps, which makes it the first suspect to rule out.
  prefetch?: boolean;
  // last chance to extend/replace the viewer state before it is restored. Layers that must
  // exist at viewer-construction time (e.g. a live segmentation/skeleton source, which also
  // needs its credentials registered BEFORE setupDefaultViewer runs) belong here rather than
  // in onRankReady, which fires only after the viewer is already built.
  onBuildViewerState?: (state: any) => any;
}

// How much loaded path we want in front of the camera, expressed as seconds of travel at the
// current speed. Neuroglancer's own prefetch predicts PREFETCH_MS=2000ms ahead
// (sliceview/backend.js), so asking for ~2s keeps the two mechanisms aimed at the same window.
const LOOKAHEAD_SECONDS = 2;
// Never stop completely: a frozen camera can't trigger the velocity estimator that drives
// prefetch, so it would have to wait for plain visible-tier loading to dig it out.
const MIN_SPEED_FRACTION = 0.08;

// Zoom threshold (nm per screen pixel) past which neuroglancer's prefetch stops paying for itself.
//
// Measured, because the two regimes pull in opposite directions. The tube is a SINGLE-SCALE volume,
// so zooming out cannot switch to a coarser level -- it just multiplies the number of full-res tiles
// the view needs, and nearly all of the extra ones lie outside the tube and 404. Prefetch makes that
// worse, because it sprays up to 32 tiles per axis along straight lines that the curving axon leaves
// almost immediately.
// Measured as time for the VISIBLE footprint to reach 95% of the real (non-empty) chunks it will
// ever get, with the camera FLYING -- the actual use case. Earlier attempts at this measurement were
// biased and are worth naming so they aren't repeated: "time until the resident-chunk count
// plateaus" penalises prefetch by construction (prefetch deliberately loads beyond the view), and
// neuroglancer's own numVisibleChunksNeeded == numVisibleChunksAvailable ALWAYS, because a 404
// counts as available -- so its completeness signal cannot see holes at all.
//   at the default zoom (3.2 nm/px): prefetch HELPS -- look-ahead is what keeps a moving camera fed
//     (speed fraction 0.68 with it vs 0.46 without, at 2500 nm/s).
//   at 8x out (25.6 nm/px), flying: prefetch HURTS 4x -- 10132ms to fill the view with it on vs
//     2504ms with it off. Stationary at the same zoom it makes no difference (402ms either way):
//     the damage is specific to a moving camera, which keeps entering new territory where prefetch
//     extrapolates outward past the strip and the doomed requests crowd out the real ones.
// `download.itemLimit` was measured to be irrelevant in both directions (5032 vs 5030 ms at 16 vs 6,
// and 1006 vs 1008 with prefetch off): the real ceiling is Chrome's 6 HTTP/1.1 connections per
// origin -- confirmed by counting sockets, exactly 6 -- so the setting cannot buy concurrency that
// does not exist. Hence prefetch, not concurrency, is the lever.
const PREFETCH_MAX_NM_PER_PX = 8;
const PREFETCH_HYSTERESIS = 1.25; // re-enable only well below the threshold, so it can't flap

// Measured across-path width of the cached tube, for radius_nm = 1000 (which no client overrides --
// checked: service.py, api.py and both frontends all leave it at the default). tube_chunks() unions
// +/-1000nm CUBES snapped outward to 64^3 chunk edges, which measured as -1856..+1216 nm around the
// centerline -- asymmetric, because the centerline sits at an arbitrary offset within its chunk.
// Take the narrow side: this is used to warn, so erring small errs safe.
const TUBE_HALF_WIDTH_NM = 1216;

// The warning measures this central fraction of the 2D panel, not the whole thing -- see viewInfo.
const VIEW_CENTRE_FRACTION = 0.5;

export interface BufferDepth {
  // arc nm of contiguously-TRAVERSABLE path ahead of the camera (capped at the target window).
  // "Traversable" counts known-absent chunks as passable -- they are never going to load, so the
  // camera must not wait for them. This is what gates the speed.
  aheadNm: number;
  // arc nm of contiguously-REAL (actual bytes) path ahead. <= aheadNm; the gap between them is
  // path the camera will fly over with nothing to look at. HUD only -- never gates the speed.
  realAheadNm: number;
  // the window we're trying to keep loaded (arc nm); aheadNm/targetNm drives the speed
  targetNm: number;
  // speed multiplier actually applied this frame, 1 = unimpeded
  speedFraction: number;
  // false once the precise per-chunk probe is unavailable and we're on the ratio fallback
  precise: boolean;
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
  getBufferDepth(): BufferDepth;
  /** Summary of the previous session if it died without a clean exit (blank page / renderer
   * kill), else null. Read at startup -- the evidence is in localStorage, not the console. */
  getCrashReport(): string | null;
  /** Point the kernel at a (different) cell. Needed because the kernel is constructed before the
   * page knows which cell to open -- see FlyKernelOptions.rootId. */
  setRootId(id: string): void;
  /** Scope the camera fetch's background pre-build -- same reason as setRootId: the cell's
   * recorded scope isn't known until it's been looked up. See FlyKernelOptions.compartment. */
  setCompartment(compartment: string): void;
  /** What is actually on screen right now: chunks in the 2D panel's footprint classified into real
   * (has image), absent (404 -- black forever) and pending (still coming). `pastStrip` is true once
   * a meaningful share of the view is absent, i.e. black for want of DATA rather than of time. */
  getViewInfo(): {
    nmPerPx: number;
    halfViewNm: number;
    real: number;
    absent: number;
    pending: number;
    holeFrac: number;
    pastStrip: boolean;
  };
  // one-shot console dump: per-source chunk counts + cache pressure, to tell "chunks were never
  // requested" apart from "chunks were loaded and then evicted"
  logCacheDiagnostic(): Promise<void>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

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
  let rootId = opts.rootId;
  let compartment = opts.compartment;

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

  // ---- buffer depth: how much loaded path is in front of the camera --------------------- //
  //
  // Measured by asking the `em` layer's chunk cache directly whether the chunk containing each
  // upcoming path position is resident in GPU memory. Keys are `curPositionInChunks.join()` and
  // residency is `state === ChunkState.GPU_MEMORY` -- the same lookup neuroglancer's own
  // SliceView.isReady does (sliceview/frontend.js). We deliberately do NOT call
  // viewer.isReady(): it flushes ALL pending chunk updates past the 30ms frame deadline
  // (chunk_manager/frontend.js), and it also aggregates the 3D panel's live graphene skeleton
  // layer, which would let a slow network source throttle a fly-through over local chunks.
  //
  // The containing chunk is a PROXY for the full visible footprint (the 2D window spans a couple
  // of chunks at crossSectionScale 0.2), so this can read "loaded" for a view still missing edge
  // chunks. Good enough for a depth metric; the speed law only needs a monotone signal.
  interface ChunkProbe {
    // `data` is null for a chunk the server 404'd (see classifyChunk) and a typed array otherwise.
    chunks: Map<string, { state: number; data?: unknown }>;
    chunkDataSize: number[];
    // The chunk grid is anchored at the volume's voxel_offset, NOT at global origin: neuroglancer
    // builds a chunk's voxel bounds as `gridPosition * chunkDataSize + baseVoxelOffset`
    // (sliceview/volume/backend.js computeChunkBounds), and the precomputed datasource keeps the
    // spec 0-based (`upperVoxelBound: scaleInfo.size`) with the offset held separately
    // (datasource/precomputed/frontend.js). So this has to come off the position before dividing --
    // verified against the on-disk chunk filenames, which are exactly those offset bounds.
    baseVoxelOffset: number[];
    lowerChunkBound: number[];
    upperChunkBound: number[];
    minSpanNm: number; // smallest chunk extent in nm, sets the probe step
  }
  let probe: ChunkProbe | null = null;
  let probeAttempts = 0;
  let probeUsable = true; // false -> permanently on the ratio fallback
  let probeVerified = false;
  let depth: BufferDepth = {
    aheadNm: 0, realAheadNm: 0, targetNm: 0, speedFraction: 1, precise: false,
  };
  let speedFraction = 1; // smoothed, so motion doesn't judder as chunks land
  let branchLoads = 0; // cumulative -- the old renderer kill grew with branch SWITCHES, not time

  // Cache pressure, refreshed on a slow cadence (see sampleAsyncStats) rather than read fresh each
  // tick: getStatistics() is an RPC to the worker, and polling it every second would perturb the
  // very memory behavior this is trying to measure.
  let asyncStats: { gpuMB: number; systemMB: number; expired: number } | null = null;

  async function sampleAsyncStats() {
    try {
      const cq = viewer?.dataContext?.chunkQueueManager;
      if (!cq) return;
      const stats = await cq.getStatistics();
      // Same indexing as logCacheDiagnostic below: [state][tier][ChunkMemoryStatistics], group
      // scaled by numChunkMemoryStatistics then offset to the field (ui/statistics.js).
      const stat = (arr: Float64Array, state: number, field: number) => {
        let sum = 0;
        for (let tier = 0; tier < 3; tier++) {
          sum += arr[getChunkStateStatisticIndex(state, tier) * numChunkMemoryStatistics + field] ?? 0;
        }
        return sum;
      };
      let gpuBytes = 0;
      let systemBytes = 0;
      let expired = 0;
      for (const [, arr] of stats) {
        gpuBytes += stat(arr, ChunkState.GPU_MEMORY, ChunkMemoryStatistics.gpuMemoryBytes);
        systemBytes += stat(arr, ChunkState.GPU_MEMORY, ChunkMemoryStatistics.systemMemoryBytes);
        expired += stat(arr, ChunkState.EXPIRED, ChunkMemoryStatistics.numChunks);
      }
      asyncStats = { gpuMB: Math.round(gpuBytes / 1e6), systemMB: Math.round(systemBytes / 1e6), expired };
    } catch {
      /* best-effort -- diagnostics must never throw into the caller */
    }
  }
  window.setInterval(sampleAsyncStats, 10000);

  // Records a rolling trail to localStorage so a renderer kill (blank page) leaves evidence.
  // Per-second fields are cheap, already-maintained numbers; gpuMB/systemMB/expired come from the
  // slow sampler above instead of calling getStatistics() here directly.
  const crash: CrashWatch = startCrashWatch({
    getStats: () => ({
      chunks: probe?.chunks.size ?? -1,
      branchLoads,
      pid: currentPid,
      phase,
      speed,
      aheadNm: Math.round(depth.aheadNm),
      frac: Number(depth.speedFraction.toFixed(2)),
      layers: viewer?.layerManager?.managedLayers?.length ?? -1,
      gpuMB: asyncStats?.gpuMB ?? -1,
      systemMB: asyncStats?.systemMB ?? -1,
      expired: asyncStats?.expired ?? -1,
    }),
  });

  function emUserLayer(): any {
    const managed = viewer?.layerManager?.managedLayers ?? [];
    return managed.find((l: any) => l.name === "em")?.layer ?? null;
  }

  function resolveProbe(): ChunkProbe | null {
    if (probe || !probeUsable) return probe;
    // visibleSourcesList only fills once the layer is actually being rendered, so this legitimately
    // fails for the first frames after a branch loads. Keep retrying indefinitely (one find + a
    // property read per call is nothing) and just say so once -- giving up permanently would
    // downgrade the metric for the whole session because one cold branch was slow to first paint.
    if (++probeAttempts === 300) {
      console.warn("[flykernel] chunk probe still unresolved; on view-completeness fallback for now");
    }
    try {
      const em = emUserLayer();
      for (const rl of em?.renderLayers ?? []) {
        // visibleSourcesList entries are {source: TransformedSource, ...}; the chunk map and spec
        // live on that entry's `source` (verified against the live viewer -- `.source.source` is
        // undefined here, so this is the level that owns `chunks`).
        const src = rl.visibleSourcesList?.[0]?.source;
        const spec = src?.spec;
        if (!src?.chunks || !spec?.chunkDataSize) continue;
        const cds: number[] = Array.from(spec.chunkDataSize);
        probe = {
          chunks: src.chunks,
          chunkDataSize: cds,
          baseVoxelOffset: Array.from(spec.baseVoxelOffset ?? cds.map(() => 0)),
          lowerChunkBound: Array.from(spec.lowerChunkBound ?? cds.map(() => 0)),
          upperChunkBound: Array.from(spec.upperChunkBound ?? cds.map(() => Infinity)),
          minSpanNm: Math.min(...cds.map((n, i) => n * resNm[i])),
        };
        return probe;
      }
    } catch (e) {
      console.warn("[flykernel] chunk probe resolve failed; using fallback", e);
      probeUsable = false;
    }
    return null;
  }

  // chunk-grid key for a GLOBAL voxel position, matching updateFixedCurPositionInChunks' clamped
  // floor (sliceview/base.js) once the position is put in the source's own 0-based chunk space.
  // Assumes the em layer's voxel grid is the viewer's global grid, which holds because the viewer
  // dimensions are constructed from this very volume's `resolution_nm` (see setupViewer).
  function chunkKey(p: [number, number, number], pr: ChunkProbe): string {
    const k: number[] = [];
    for (let d = 0; d < 3; d++) {
      const c = Math.floor((p[d] - pr.baseVoxelOffset[d]) / pr.chunkDataSize[d]);
      k.push(clamp(c, pr.lowerChunkBound[d], pr.upperChunkBound[d] - 1));
    }
    return k.join();
  }

  // Three outcomes, and the distinction matters because two of them look identical on screen:
  //
  //   "real"    bytes arrived; this is reviewable EM
  //   "absent"  the server answered 404, which neuroglancer treats as a SUCCESSFUL empty download:
  //             `data === null`, promoted to GPU_MEMORY at 0 bytes, painted as the shader fill value
  //             (black). Nothing will ever arrive here -- the tube is a narrow strip around the
  //             centerline and this position is outside it.
  //   "pending" not in the map at all. The frontend map only ever holds SYSTEM_MEMORY/GPU_MEMORY
  //             (chunk_manager/frontend.js applyChunkUpdate throws on anything else), so NEW /
  //             QUEUED / DOWNLOADING / FAILED are all simply absent keys -- i.e. still coming.
  //
  // There is deliberately no ChunkState.FAILED case: the frontend never sees FAILED (it is a
  // worker-only state), and a 404 is not a failure anyway. The old code tested for it, which was
  // unreachable -- the intended "don't wait for a hole" behaviour was happening by accident via the
  // GPU_MEMORY branch, which is exactly what makes `absent` indistinguishable from `real` today.
  type ChunkClass = "real" | "absent" | "pending";
  function classifyChunk(p: [number, number, number], pr: ChunkProbe): ChunkClass {
    const c = pr.chunks.get(chunkKey(p, pr));
    if (c === undefined) return "pending";
    if (c.state !== ChunkState.GPU_MEMORY) return "pending"; // SYSTEM_MEMORY: awaiting GPU upload
    return c.data == null ? "absent" : "real";
  }

  // Fallback metric: completeness of the CURRENT view only, from the counters the worker keeps
  // updated every ~200ms (the pattern ui/layer_bar.js uses). Reactive rather than predictive --
  // it can only notice we're already in an incomplete view, not that one is coming.
  //
  // Returns null for "no information": a layer that isn't rendering yet reports needed === 0,
  // which is NOT the same as "everything is loaded". Conflating the two is actively harmful --
  // it let the priming dwell return instantly and made the probe self-check run against an empty
  // chunk map, falsely concluding the key math was broken. Callers decide what no-information
  // means for them (keep waiting vs. don't stall the camera).
  function viewCompleteness(): number | null {
    try {
      let needed = 0;
      let available = 0;
      for (const rl of emUserLayer()?.renderLayers ?? []) {
        const info = rl.layerChunkProgressInfo;
        if (!info) continue;
        needed += info.numVisibleChunksNeeded;
        available += info.numVisibleChunksAvailable;
      }
      return needed === 0 ? null : available / needed;
    } catch {
      return null;
    }
  }

  // Two different questions about the path ahead, answered in ONE walk because they need different
  // stopping rules:
  //
  //   traversableNm -- stops only at "pending". Drives the camera speed. A hole must NOT stop the
  //                    camera: nothing is ever going to arrive there, so waiting would park us at
  //                    the slowdown floor forever.
  //   realNm        -- stops at "pending" OR "absent". Drives the HUD only. This is the honest
  //                    "how much reviewable EM is ahead of me" number, and it is the one that must
  //                    never claim coverage we do not have.
  function aheadNm(fromArc: number, capNm: number): { traversableNm: number; realNm: number } {
    const pr = resolveProbe();
    if (!pr) {
      // Fallback: scale the window by how complete the current view is. No telemetry -> assume
      // clear, so a missing metric can never be the thing that slows the fly-through down.
      const f = capNm * (viewCompleteness() ?? 1);
      return { traversableNm: f, realNm: f };
    }
    const step = Math.max(1, pr.minSpanNm / 2); // half a chunk: can't skip a boundary
    let realNm = -1;
    for (let d = 0; d <= capNm; d += step) {
      const k = classifyChunk(posAt(fromArc + d), pr);
      if (realNm < 0 && k !== "real") realNm = d; // first non-real position ends the real run
      if (k === "pending") return { traversableNm: d, realNm: realNm < 0 ? d : realNm };
    }
    return { traversableNm: capNm, realNm: realNm < 0 ? capNm : realNm };
  }

  // Turn prefetch off once zoomed out past the point where it pays for itself, and back on when
  // zoomed back in. See PREFETCH_MAX_NM_PER_PX for the measurements behind the threshold.
  //
  // `zoomFactor` is canonical voxels per screen pixel, and the canonical voxel here is the finest
  // axis of the tube's own resolution (the viewer's dimensions are built from `resolution_nm`), so
  // nm/px is zoomFactor * min(resNm). Read rather than watched: one property read on an existing
  // 6-frame cadence is cheaper than owning a listener's lifetime.
  // What the user is ACTUALLY looking at: classify every chunk inside the 2D panel's footprint.
  //
  // Deliberately measured rather than predicted from radius_nm. A geometric prediction cries wolf:
  // at the default zoom the view half-width (~1280nm) already exceeds the straight-path tube
  // half-width (1216nm), yet the view measures ZERO holes -- because the tube is a union of boxes
  // ALONG the path, so wherever the axon runs across the viewing plane the real coverage is much
  // wider than the worst case. Counting beats predicting.
  //
  // Cost is one Map lookup per chunk in the footprint (8 at the default zoom, a few hundred zoomed
  // far out), on the same 6-frame cadence as everything else here.
  function viewInfo(): {
    nmPerPx: number;
    halfViewNm: number;
    real: number;
    absent: number;
    pending: number;
    holeFrac: number;
    pastStrip: boolean;
  } {
    let nmPerPx = 0;
    let panelPx = window.innerWidth / 2;
    let panelPy = window.innerHeight;
    try {
      const zoom = viewer?.navigationState?.zoomFactor?.value;
      if (zoom > 0) nmPerPx = zoom * Math.min(...resNm);
      const c = viewer?.display?.canvas as HTMLCanvasElement | undefined;
      if (c?.clientWidth) panelPx = c.clientWidth / 2; // xy-3d splits horizontally
      if (c?.clientHeight) panelPy = c.clientHeight;
    } catch {
      /* fall back to window size */
    }
    const halfViewNm = (nmPerPx * panelPx) / 2;
    const none = { nmPerPx, halfViewNm, real: 0, absent: 0, pending: 0, holeFrac: 0, pastStrip: false };
    const pr = resolveProbe();
    const posVox = viewer?.navigationState?.pose?.position?.value;
    if (!pr || !posVox || posVox.length !== 3 || !nmPerPx) return none;
    // half-extent of the panel in VOXELS on each display axis (zoom is voxels per pixel), scaled to
    // the CENTRAL region rather than the whole panel.
    //
    // Why the centre: the axon is centred in frame and that is where myelination is judged, so a
    // hole there is what could actually be misread as unmyelinated. Measuring the full panel makes
    // the warning fire on edge chunks that are outside the strip by chunk-grid rounding -- measured
    // 25% "holes" at the DEFAULT zoom at one position, which would put a permanent warning on screen
    // and train the user to ignore it. A warning that is always on is worth nothing.
    const zoom = nmPerPx / Math.min(...resNm);
    const hx = (zoom * panelPx * VIEW_CENTRE_FRACTION) / 2;
    const hy = (zoom * panelPy * VIEW_CENTRE_FRACTION) / 2;
    const cIdx = (v: number, d: number) =>
      clamp(
        Math.floor((v - pr.baseVoxelOffset[d]) / pr.chunkDataSize[d]),
        pr.lowerChunkBound[d],
        pr.upperChunkBound[d] - 1,
      );
    const x0 = cIdx(posVox[0] - hx, 0), x1 = cIdx(posVox[0] + hx, 0);
    const y0 = cIdx(posVox[1] - hy, 1), y1 = cIdx(posVox[1] + hy, 1);
    const z0 = cIdx(posVox[2], 2);
    let real = 0, absent = 0, pending = 0;
    for (let x = x0; x <= x1; x++) {
      for (let y = y0; y <= y1; y++) {
        const c = pr.chunks.get([x, y, z0].join());
        if (c === undefined || c.state !== ChunkState.GPU_MEMORY) pending++;
        else if (c.data == null) absent++;
        else real++;
      }
    }
    const known = real + absent;
    const holeFrac = known > 0 ? absent / known : 0;
    // 10%: below that it's chunk-grid rounding at the panel edge, not a coverage problem worth
    // interrupting the user about.
    return { nmPerPx, halfViewNm, real, absent, pending, holeFrac, pastStrip: holeFrac > 0.1 };
  }

  let prefetchOn: boolean | null = null;
  function updatePrefetchForZoom() {
    if (opts.prefetch === false) return; // explicitly forced off (?prefetch=0) -- never re-enable
    try {
      const cq = viewer?.dataContext?.chunkQueueManager;
      const zoom = viewer?.navigationState?.zoomFactor?.value;
      if (!cq || !(zoom > 0)) return;
      const nmPerPx = zoom * Math.min(...resNm);
      // hysteresis: drop out above the threshold, come back only well below it
      const want =
        prefetchOn === false
          ? nmPerPx < PREFETCH_MAX_NM_PER_PX / PREFETCH_HYSTERESIS
          : nmPerPx <= PREFETCH_MAX_NM_PER_PX;
      if (want !== prefetchOn) {
        prefetchOn = want;
        cq.enablePrefetch.value = want;
        console.debug(
          `[flykernel] prefetch ${want ? "on" : "off"} (${nmPerPx.toFixed(1)} nm/px` +
            `${want ? "" : " -- zoomed out past the cached strip; look-ahead would mostly miss"})`,
        );
      }
    } catch {
      /* never let a zoom read break the frame loop */
    }
  }

  // Re-measure buffer depth and update the smoothed speed multiplier. Cheap (a handful of Map
  // lookups) but pointless to redo every frame at 60fps, so it runs on the same 6-frame cadence
  // as the progress callback below.
  function updateBufferDepth() {
    const window = Math.max(LOOKAHEAD_SECONDS * speed, 2 * (probe?.minSpanNm ?? 1024));
    // Near the end of a branch there is less path left than the window -- measure against what
    // remains, or we'd read "shallow buffer" and crawl over the last stretch of every branch.
    const remaining = Math.max(0, totalArc - s);
    const target = Math.min(window, remaining);
    if (target <= 0) {
      depth = { aheadNm: 0, realAheadNm: 0, targetNm: 0, speedFraction: 1, precise: !!probe };
      speedFraction = 1;
      return;
    }
    const { traversableNm, realNm } = aheadNm(s, target);
    // Speed is gated on TRAVERSABLE, not real: see aheadNm. A stretch that is genuinely absent must
    // not brake the camera, or the camera would never get past it.
    const wanted = clamp(traversableNm / target, MIN_SPEED_FRACTION, 1);
    // ease toward the target so chunks landing mid-glide don't snap the speed
    speedFraction += (wanted - speedFraction) * 0.25;
    depth = {
      aheadNm: traversableNm,
      realAheadNm: realNm,
      targetNm: target,
      speedFraction,
      precise: !!probe,
    };
  }

  const frame = (now: number) => {
    const dt = (now - last) / 1000;
    last = now;
    frames++;

    if (phase === "play" && running && dt > 0 && ptsVox.length >= 2) {
      if (frames % 6 === 0) updateBufferDepth();
      // Advance proportionally to how much loaded path is ahead: full speed when well buffered,
      // creeping when the loader is barely keeping up. This is what stops the camera from gliding
      // over EM that hasn't arrived -- an unloaded stretch of axon otherwise looks exactly like
      // an unmyelinated one.
      s += speed * speedFraction * dt; // forward only
      if (s >= totalArc) {
        s = totalArc;
        running = false;
      }
      setPosition(posAt(s));
    }

    // Outside the play branch on purpose: the user can zoom while PAUSED, which is exactly when
    // they're inspecting, and that is the case where a wrong prefetch setting costs the most.
    if (frames % 6 === 0) updatePrefetchForZoom();

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
      // Prefetch ON: neuroglancer's prefetch is VELOCITY-based -- it estimates camera velocity and
      // requests up to PREFETCH_MS=2000ms ahead along the direction of travel
      // (sliceview/backend.js), which is exactly the access pattern of a constant-speed
      // fly-through. It used to be off because the old buffer-the-whole-branch pre-pass already
      // touched every chunk, so prefetch only added contention for the 16 download slots; now that
      // playback is gated on a rolling look-ahead window instead, prefetch is what keeps that
      // window full. Caveat worth knowing: MAX_PREFETCH_VELOCITY=0.1 global-voxels/ms means
      // prefetch quietly disengages per-dimension above ~1600nm/s at 16nm voxels, so at the top of
      // the speed slider the proportional slowdown carries it alone.
      cq.enablePrefetch.value = opts.prefetch !== false;
    } catch (e) {
      console.warn("[flykernel] could not set cache limits", e);
    }

    try {
      const canvas = viewer.display?.canvas as HTMLCanvasElement | undefined;
      canvas?.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        status("WebGL context lost (GPU memory) -- reload the page", "warn");
        console.error("[flykernel] webglcontextlost");
        // Into the breadcrumb too: a context loss that PRECEDES a blank page distinguishes a
        // GPU-side failure from a renderer OOM, which leaves no such note.
        crash.note("webglcontextlost");
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

  // Dwell at the current camera position until the em layer reports the view complete, or
  // `timeoutMs` elapses. The minimum dwell exists because the completeness counters are refreshed
  // by the worker only every ~200ms (chunk_manager/backend.js): sampled immediately after moving,
  // they can still describe the PREVIOUS position and read as a spurious "ready".
  async function dwellUntilLoaded(timeoutMs: number, minMs = 250): Promise<boolean> {
    const t0 = performance.now();
    for (;;) {
      await sleep(50);
      const elapsed = performance.now() - t0;
      // Require positive evidence (needed > 0 AND all of it available). A null reading means the
      // layer isn't rendering yet, so there is nothing to conclude -- keep waiting until the
      // timeout rather than declaring the view loaded.
      const c = viewCompleteness();
      if (elapsed >= minMs && c !== null && c >= 1) return true;
      if (elapsed >= timeoutMs) return false;
    }
  }

  // Prime only the START of the branch, then hand off to the proportional gate in frame().
  //
  // This used to sweep the ENTIRE branch before playing, which on a long branch was both slow to
  // get going and self-defeating: a branch whose chunks exceed the 1GB GPU cap evicts its own
  // early chunks before the sweep reaches the end, so "buffering 100%" still meant an unloaded
  // start. Priming a couple of look-ahead windows is enough to begin smoothly; the rolling
  // look-ahead keeps it that way, and prefetch does the fetching in front of the camera.
  async function bufferAndPlay(pid: number) {
    const token = ++bufferToken;
    phase = "buffer";
    s = 0;
    speedFraction = 1; // start optimistic; the gate corrects within a few frames
    const primeNm = Math.min(totalArc, 2 * LOOKAHEAD_SECONDS * speed);
    const stops = Math.max(1, Math.ceil(primeNm / Math.max(1, stepNm * 2)));
    for (let i = 0; i <= stops; i++) {
      if (token !== bufferToken) return;
      setPosition(posAt((primeNm * i) / stops));
      status(`priming branch ${pid} ${Math.round((i / stops) * 100)}% -- caching...`);
      if (!(await dwellUntilLoaded(1500)) && token === bufferToken) {
        // Not fatal: the gate will simply hold the camera back here instead. Worth a console note
        // because a stop that can't complete in 1.5s from LOCAL disk usually means either eviction
        // pressure or a chunk the server never wrote.
        console.debug(`[flykernel] prime stop ${i}/${stops} incomplete after 1.5s`);
      }
    }
    if (token !== bufferToken) return;
    s = 0;
    setPosition(ptsVox[0]);
    const startLoaded = await dwellUntilLoaded(1500);
    if (token !== bufferToken) return;

    // Self-check the precise probe exactly once, at the one moment we have an independent answer:
    // the em layer just reported this view complete, so the chunk under the camera MUST read as
    // resident. If it doesn't, our grid math disagrees with this layer's transform -- fall back to
    // the ratio metric rather than crawl the whole branch at the slowdown floor on a bad key.
    if (startLoaded && !probeVerified && probeUsable) {
      const pr = resolveProbe();
      // Only conclude anything when the map has content to disagree with us: an empty map means
      // the layer hasn't populated it yet, which says nothing about our key math. Concluding
      // "broken" there is a false negative that permanently downgrades the metric -- exactly the
      // bug that made this check disable itself on the first run.
      if (pr && pr.chunks.size > 0) {
        probeVerified = true;
        // Only "pending" indicts the key math: it means our computed key found NOTHING in the map,
        // even though the layer just reported this view complete. "absent" is a real answer (the
        // key WAS found, holding a 404's null data), so it proves the math works.
        if (classifyChunk(ptsVox[0], pr) === "pending") {
          console.warn(
            "[flykernel] chunk-probe self-check failed (view complete but containing chunk is not " +
              "in the chunk map at all); falling back to view-completeness metric",
          );
          probe = null;
          probeUsable = false;
        }
      }
    }
    updateBufferDepth();
    phase = "play";
    running = false; // stay paused after priming; caller presses play to start
    status(`branch ${pid}: ready -- ${ptsVox.length} nodes`, "ok");
  }

  async function loadBranch(pid: number): Promise<Camera> {
    branchLoads++;
    crash.note(`loadBranch ${pid} (#${branchLoads})`);
    await opts.onBeforeLoadBranch?.(pid);
    bufferToken++; // stop any current buffering immediately
    phase = "buffer";
    status(`branch ${pid}: fetching camera path (building tube if first visit)...`);
    const url = new URL(`${opts.api}/api/cells/${rootId}/branches/${pid}/camera`);
    if (compartment) url.searchParams.set("compartment", compartment);
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
    setRootId: (id: string) => {
      rootId = id;
    },
    setCompartment: (c: string) => {
      compartment = c;
    },
    getBufferDepth: () => depth,
    getCrashReport: () => crash.report(),
    getViewInfo: viewInfo,
    // Answers "were these chunks never requested, or were they loaded and then thrown away?" --
    // the two causes need opposite fixes (more look-ahead vs. a bigger cache), and guessing
    // between them is how you end up tuning the wrong knob.
    logCacheDiagnostic: async () => {
      try {
        const cq = viewer?.dataContext?.chunkQueueManager;
        if (!cq) return;
        // The frontend capacity objects carry only the LIMITS (sizeLimit/itemLimit) -- there is no
        // current-usage field to read, so actual pressure has to come from getStatistics() below.
        const gpu = cq.capacities.gpuMemory;
        console.log(
          `[flykernel] limits: gpu ${(gpu.sizeLimit.value / 1e9).toFixed(2)}GB / ` +
            `${gpu.itemLimit.value} items, system ` +
            `${(cq.capacities.systemMemory.sizeLimit.value / 1e9).toFixed(2)}GB, ` +
            `${cq.capacities.download.itemLimit.value} concurrent downloads, ` +
            `prefetch ${cq.enablePrefetch.value ? "on" : "off"}`,
        );
        const stats = await cq.getStatistics();
        // The statistics array is [state][tier][ChunkMemoryStatistics], so the group index has to
        // be scaled by numChunkMemoryStatistics and then offset to the field you want -- indexing
        // the group directly reads `numChunks` for one state as a BYTE count of another
        // (ui/statistics.js is the reference for this).
        const stat = (arr: Float64Array, state: number, field: number) => {
          let sum = 0;
          for (let tier = 0; tier < 3; tier++) {
            sum += arr[getChunkStateStatisticIndex(state, tier) * numChunkMemoryStatistics + field] ?? 0;
          }
          return sum;
        };
        let residentTotal = 0;
        let queuedTotal = 0;
        let gpuBytes = 0;
        for (const [source, arr] of stats) {
          const resident = stat(arr, ChunkState.GPU_MEMORY, ChunkMemoryStatistics.numChunks);
          // The real eviction signal. ChunkState.EXPIRED is NEVER assigned in neuroglancer 2.41.2
          // (grep: only the enum, the wire message, and the frontend delete) so its statistics
          // buckets are permanently zero -- this diagnostic used to report it and advise on it,
          // which could never fire. Eviction actually moves a chunk back to QUEUED, so a QUEUED
          // count that climbs while the camera stalls is what "the cache is thrashing" looks like.
          const queued = stat(arr, ChunkState.QUEUED, ChunkMemoryStatistics.numChunks);
          const bytes = stat(arr, ChunkState.GPU_MEMORY, ChunkMemoryStatistics.gpuMemoryBytes);
          residentTotal += resident;
          queuedTotal += queued;
          gpuBytes += bytes;
          const name = (source as any)?.constructor?.name ?? "source";
          console.log(
            `[flykernel]   ${name}: ${resident} chunks resident ` +
              `(${(bytes / 1e6).toFixed(1)}MB gpu), ${queued} queued`,
          );
        }
        const gpuPct = ((gpuBytes / gpu.sizeLimit.value) * 100).toFixed(0);
        const vi = viewInfo();
        console.log(
          `[flykernel] total ${residentTotal} resident / ${queuedTotal} queued, ` +
            `${(gpuBytes / 1e6).toFixed(0)}MB gpu = ${gpuPct}% of cap; buffer depth ` +
            `${Math.round(depth.aheadNm)}nm traversable / ${Math.round(depth.realAheadNm)}nm with ` +
            `data, of ${Math.round(depth.targetNm)}nm wanted; ` +
            `speedFraction=${depth.speedFraction.toFixed(2)} precise=${depth.precise}`,
        );
        console.log(
          `[flykernel] zoom ${vi.nmPerPx.toFixed(1)} nm/px -> view half-width ` +
            `${Math.round(vi.halfViewNm)}nm vs cached strip ~${TUBE_HALF_WIDTH_NM}nm` +
            (vi.pastStrip
              ? " -- PAST THE STRIP: the periphery is black for want of data, not time, and " +
                "prefetch is disabled here because its look-ahead would mostly miss"
              : " -- view fits inside the strip"),
        );
        console.log(
          "[flykernel] a climbing 'queued' while the camera stalls = eviction thrashing; " +
            "steady queued with a stalled camera = the loader isn't keeping up (speed / zoom)",
        );
      } catch (e) {
        console.warn("[flykernel] cache diagnostic failed", e);
      }
    },
  };
}
