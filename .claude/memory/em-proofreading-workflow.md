---
name: em-proofreading-workflow
description: EM proofreading workflow — Phase A BUILT on branch proofreading-package-and-em-design; design in repo docs
metadata: 
  node_type: memory
  type: project
  originSessionId: f6dfc60d-4097-495d-abd5-e0caa7815b6d
---

EM proofreading workflow: neuroglancer fly-through from the python API over MICrONS
`minnie65_phase3_v1` (`minnie65_public` = read-only dev sandbox).

**Phase A is IMPLEMENTED** (not merged to master) on branch
`proofreading-package-and-em-design`: engine in `proofreading/em/` (skeleton_tree,
path, wal, coverage, client) + interactive session (viewer.py, annotator.py
`ProofreadSession`) + `demo_em_phase_a.ipynb`/`.py`. Commits 63f9e5e (engine),
15f5977 (session). Verified against the live cell 864691135572530981; basic flow
works in-browser per the user. Phase B (manual edits) / Phase C (re-entry reconcile)
not built yet.

Design docs in the repo: `docs/proofreading-workflow.md` (plan), `CONTEXT.md`
(vocabulary), `docs/adr/0001..0003` (decisions).

**REVIEW ARCHITECTURE — VALIDATED (2026-05-29): sparse-tube local precomputed, served.**
seg-only-renders-when-idle is solved by serving a LOCAL precomputed EM(+target) layer that
neuroglancer streams from localhost — it renders SHARP DURING MOTION (confirmed in-browser:
"flythrough good"). Stays entirely in neuroglancer (user's idea; beats screenshot +
microviewer + in-memory-LocalVolume detours, all tried same day).
WHY in-memory LocalVolume failed: a neuron is a thin tree in a huge box, so a DENSE mip1
cutout is infeasible — even per-branch the bbox is median 833 MB / up to 5 TB (whole cell
917×724×520µm, 15mm cable, dense mip1 = 34 TB). The SPARSE TUBE is tiny: median 14 MB / max
273 MB per branch.
HOW (`em/tube.py`): `tube_chunks()` = unique 64³ chunks within radius_nm of the resampled
path; `build_local_volume(src, points, radius, cache_dir, name, transform=)` downloads only
those chunks from a CloudVolume `src` (PARALLEL ThreadPoolExecutor — serial was 161s/133MB
for one branch, latency-bound) and writes a single-scale local precomputed (`CloudVolume.
create_new_info`, file://, encoding raw, compress False) mirroring src res/offset → world-
aligned. `serve_dir()` = CORS ThreadingHTTPServer over the cache → neuroglancer
`precomputed://localhost/.../name` ImageLayer. EM layer grayscale; target layer =
`agg_seg==root_id` mask (transform) w/ red `_TINT` shader. Added as `PREVIEW_EM`/`PREVIEW_TGT`
so `viewer.set_preview_mode` toggles tube(in motion) vs live img+seg(on pause).
`em.tube_prototype(client, root_id, path_id)` = one-branch end-to-end → (viewer, fly).
Knobs: mip(1=16nm, user's floor; ≈133MB/branch raw), radius_nm(1000), step_nm. Cache under
`<wal_dir>/tube_cache/<datastack>/<root_id>/`.
STATUS (committed c2120da, branch proofreading-package-and-em-design): tube is the DEFAULT
review (`review_path(pid, mode='tube'|'preview'|'live')`), backed by ONE shared per-cell
`tube.CellTube` (EM+target, full bounds, filled lazily per branch via `_fill_chunks`, single
persistent layer pair — fixed the per-branch source-swap that churned the chunk cache).
Parallel + concurrent build (161s→~10+20s/branch), `.tube_done` cache = instant revisit.
Panel widget fixed: single synced controls via `_on_fly_change` hook (Review / Mark done /
`x` all rebuild + advance), `panel()` returns (no double render), `sess.diagnostics()` added.
`em_phase_a.ipynb` is the clean canonical notebook; old `demo_em_phase_a.ipynb` deleted.

**HARD CEILING (the reason for the next pivot):** driving review from the neuroglancer PYTHON
API degrades the longer a session runs — per-frame full-state `deepcopy`+`set_state` at 30fps
(grows with annotations) + browser-side accumulation (reopening the tab restores speed). The
sparse tube fixed RENDERING, not the python↔ngl INTERACTION model.
**AGREED-BUT-DEFERRED DIRECTION (user wants a shared tool + auto-glide is essential):** go
BROWSER-NATIVE — embed neuroglancer directly, animate `navigationState.pose` client-side in a
requestAnimationFrame loop (NOT the state-prop `react-neuroglancer` wrapper — too coarse),
with the python `proofreading/em` engine as a FastAPI DATA backend (serve tube precomputed +
camera paths, record annotations→WAL). Phase 0 SPIKE first: embed ngl npm Viewer + rAF camera
over one served tube, confirm smooth + non-degrading. Interim option to keep the notebook
usable: kill the per-frame deepcopy in `flythrough._interpolate_nav`. Docs: limitation +
direction written to `docs/proofreading-workflow.md`; full plan in
`~/.claude/plans/jolly-wondering-feather.md`. Weigh the build vs existing tools (NeuVue).

**PHASE 0 SPIKE — STARTED (2026-05-29), uncommitted in `web/`.** Goal: prove client-side
rAF camera over an embedded neuroglancer is smooth + non-degrading (the python↔ngl ceiling).
Stack: Vite 6 + `neuroglancer` npm **2.41.2** (matches python side). Embed via
`setupDefaultViewer()` from `neuroglancer/unstable/ui/default_viewer_setup.js` (attaches to
`<div id="neuroglancer-container">`); CSS from `.../default_viewer.css`. KEY build facts:
neuroglancer ships compiled ESM in `lib/`, uses conditional subpath imports (`#src/*`,
`#datasource/*`) that resolve to `default`=real impl when no custom condition set (→ full
viewer); workers are `new Worker(new URL("./chunk_worker.bundle.js", import.meta.url),
{type:"module"})` → **must `optimizeDeps.exclude:['neuroglancer']`** so esbuild doesn't mangle
`import.meta.url` (Vite then rewrites to `?worker_file&type=module`, verified 200). Also
`define` the optional `NEUROGLANCER_*` globals = undefined. Camera: set
`viewer.navigationState.pose.position.value = Float32Array([vx,vy,vz])` where vox = nm/res
(absolute, incl. voxel_offset) — same convention as python FlyThrough (`rs/res_nm`). Files:
`web/{package.json,vite.config.ts,tsconfig.json,index.html,src/main.ts}` (HUD shows uptime/
frames/fps-now/**fps-min** for the degradation watch; ping-pongs paths 7-11 forever) +
`web/spike_export.py` (NOT in the loop: dumps `web/public/camera_path.json` from the skeleton
+ `serve_dir(port=0)` CORS-serves the tube cache; needs repo root on sys.path).
Run: `uv run --extra em python web/spike_export.py` + (in `web/`) `npm install` once then
`npm run dev` → open localhost:5173. See `web/README.md`.

**PHASE 0 RESULT — VALIDATED IN-BROWSER (2026-05-29). Architecture confirmed; go build it.**
fps stayed FLAT over the session (no degradation — the python↔ngl 30fps ceiling is gone) AND
the EM glides SHARP during motion. Go browser-native.
Hard-won learnings (KEEP — they cost the whole session):
- **Embedding neuroglancer 2.41 via Vite:** `optimizeDeps.exclude:['neuroglancer']` (protect
  its `import.meta.url` module workers) BUT then its CommonJS deps are served raw and break
  (`does not provide an export named default` / `require is not defined`), so
  `optimizeDeps.include` them: `codemirror` + its `.js` modes/addons (mode/javascript,
  addon/fold/{foldcode,foldgutter,brace-fold}, addon/lint/lint), `core-js/actual/symbol/
  {dispose,async-dispose}.js`, `crc-32`(+`/crc32c.js`), `nifti-reader-js`, `msgpackr`,
  `numcodecs`. ESM deps (gl-matrix, lodash-es, valibot, fzstd) are fine; ikonate `?raw` native.
- **MUST `import "neuroglancer/unstable/main_module.js"`** (side-effect) before
  `setupDefaultViewer()` — it registers layer types + datasources (precomputed) + kvstores
  (http). Without it the viewer shell loads but layers never render (no data backend). Load
  state with `viewer.state.restoreState(obj)` (robust; URL-hash is encoding-fragile). Camera =
  `viewer.navigationState.pose.position.value = Float32Array([vx,vy,vz])` (absolute voxels = nm/res).
- **THE in-motion rendering rule (decisive):** neuroglancer only *finishes loading* chunks
  while navigation is IDLE. A continuous client-side camera keeps it perpetually non-idle, so
  it blanks over any *not-yet-cached* data and only fills in on pause. FIX (what worked):
  **PRE-CACHE** — keep the working set small (ONE per-branch tube, not the whole-cell shared
  one), raise `viewer.dataContext.chunkQueueManager.capacities.{gpuMemory,systemMemory}.
  sizeLimit.value` + `.enablePrefetch.value=true`, and run a buffering pre-pass (step the
  camera along the branch with ~130ms idle dwells to load every frustum) BEFORE the smooth
  glide. Then motion just replays cached chunks → sharp. (This is why the notebook's small
  per-branch tube glided sharp and the 5-branch shared tube blanked.) ALTERNATIVE if the
  working set must be large: multiscale tube (coarse mip renders during motion, sharpens on idle).
- Real-tool implication: review ONE branch at a time; on branch entry show a brief "buffering"
  then glide. NEXT BUILD: FastAPI backend over `proofreading/em` (serve per-branch tube
  precomputed + camera path; record annotations→WAL) + frontend = `web/src/main.ts` kernel
  (embed + buffer + glide) plus the annotation UI / branch checklist / pause→live-layers swap.
  web/ committed as 693d3ca on branch proofreading-package-and-em-design.

**M1 BACKEND BUILT + committed d0ddae2 (2026-06-01) — read-only glide the whole cell.** Plan
`~/.claude/plans/jolly-wondering-feather.md`. New headless modules (NO viewer/widgets; tube.py's
top-level `import neuroglancer` is benign — neuroglancer is a core dep, importing ≠ starting a
server): `proofreading/em/service.py` `CellReviewService` (composes EMClient+SkeletonTree+WAL+
Coverage+CellTube+path; `header()/branches()/camera_path(pid,orient)`; pre-warms tube-mip CVs in
__init__ to avoid threadpool race; **big ids (root_id, seed) returned as STRINGS — exceed JS 2^53**),
`proofreading/em/api.py` `create_app(wal_dir, default_datastack)` (FastAPI; sessions dict keyed by
int root_id; sync `def` handlers → threadpool; **single origin: StaticFiles mount `/tube`→
`<wal_dir>/tube_cache` + CORSMiddleware + no-store middleware** — retired serve_dir for the web
tool; camera endpoint composes absolute `precomputed://<request.base_url>/tube/...` source URLs),
`proofreading/em/serve.py` (uvicorn launcher, binds 127.0.0.1:8000, env PROOFREAD_WAL_DIR/DATASTACK/
HOST/PORT). pyproject new **`serve` extra** = fastapi+uvicorn[standard]; run `uv run --extra em
--extra serve python -m proofreading.em.serve`. __init__ adds guarded `CellReviewService`/
`create_app` exports. Endpoints: POST /api/cells (open/resume→header), GET …/branches (checklist+
summary), GET …/branches/{pid}/camera?orient= (fill_branch synchronously ~10-30s first time,
`.done`-cached instant after; returns points_nm/em_source/tgt_source/build), GET /healthz.
VERIFIED via curl on cell 864691135572530981/minnie65_public: seed 111692652428576376, 189 branches,
cached branch instant, uncached branch built 29.2s→marker→instant re-fetch, /tube info 200 + off-tube
404 + CORS `*` + no-store. Frontend `web/src/main.ts` rewritten to drive the API: open cell → branch
`<select>` picker → per-branch camera; layers are the SAME shared em/tgt for every branch so the
viewer is built ONCE, switching a branch just swaps the camera path + re-buffers (bufferToken cancels
stale sweeps). `?root=/?datastack=/?api=` query overrides. spike_export.py superseded (left as stub).
GOTCHA: `uv sync --extra em --extra serve` PRUNED ad-hoc jupyter pkgs (ipykernel etc.) from .venv →
restored with `uv pip install ipykernel` (not declared deps; a future `uv sync` re-prunes — consider a
dev extra). FRONTEND CRASH FIX (hard-won): in-place branch switch blanked the tab needing reload =
Chrome renderer OOM-kill ("render process gone", NOT a WebGL contextlost — process dies before the
event). Cause = the Phase-0 cache limits (GPU 3e9 / systemMemory 6e9) accumulate across branches.
FIX: bound BOTH caches to ~one branch so a switch EVICTS the previous (LRU) — gpuMemory 1e9 /
systemMemory 1.5e9 / download 16 / **prefetch OFF** (our buffering sweep already loads the branch;
prefetch only inflates the burst). A branch tube is ≤ a few hundred MB so 1.5e9 holds one + margin.
**M2 (annotate+coverage+resume) IN PROGRESS — sliced into M2.1→M2.4, build+test+commit one at a time;
M2.4 (merge-prune+resolve) DEFERRED per user.** Plan `~/.claude/plans/jolly-wondering-feather.md`.
**M2.1 DONE + committed 7fc83ca:** drop a tagged annotation at the cursor. Backend
`service.add_annotation(tag,xyz_nm)` (wal.add_annotation; record only, NO merge-prune yet) +
`list_annotations()`; api `POST/GET /api/cells/{root_id}/annotations` (400 on bad tag); big ids
(supervoxel) as strings. Frontend: m/s/e/q **capture-phase** keydown (beats neuroglancer defaults
s/x/n/e) reads `viewer.mouseState.position`×resolution→nm→POST→draws a colored point; point reuses
the WAL uuid as its neuroglancer annotation id (for M2.2 delete). GOTCHA (cost real time): a
`local://annotations` annotation layer's LocalAnnotationSource takes its **rank from the global
coordinate space AT CREATION**; created in the initial state / via restoreState it captures **rank 0**
(precomputed em/tgt establish the 3-D space only AFTER async load) → a 3-D point overflows on render
(`Float32Array.set offset out of bounds` in visitGeometry; UI shows empty source/output dims). FIX:
create the 4 per-tag annotation layers **lazily, once `viewer.navigationState.pose.position.value.length===3`**,
via the PROGRAMMATIC layer API `makeLayer(viewer.layerSpecification, name, {type:"annotation",
source:"local://annotations", annotationColor})` + `viewer.layerManager.addManagedLayer(...)` — NOT
restoreState (which re-creates image layers and re-races to rank 0). The layer's source is
`layer.layer.localAnnotations` (NOT localAnnotationSource); `.add({type:0,id,point:Float32Array,properties:[]})`.
**M2.2 RESUME DONE + UX, committed c6d6d34:** on reload, after the rank-3 annotation layers are
added, `restoreAnnotations()` GETs the cell's WAL annotations and redraws them (draw uses WAL uuid
as the ngl annotation id; waits for `layer.layer.localAnnotations` to load). Review UX in same
commit: cell-id input (reloads with `?root=`; default now 864691135413357554), **space**=play/pause
(hotkeys ignored while typing in INPUT/SELECT), glide forward-only **stops at branch end** (no
rewind), **progress = draggable scrubber** (drag moves camera, pauses; frame loop updates slider
when !scrubbing). VERIFIED cached coords == live world nm: recorded annotation xyz resolve in the
LIVE de-agglomerated seg to this cell's supervoxels, within tube radius of the skeleton; viewer
coord space is the tube mip1 [16,16,40] but we store nm so it's resolution-independent.
**M2.2 DELETE DONE + committed 63fcb66 (M2.2 now complete):** `d`/Backspace/Delete over a data panel
removes the mark nearest the cursor from viewer+WAL. Backend `service.delete_annotation(uuid)` =
`wal.tombstone(uuid)` + rebuild `Coverage.from_wal_state(WAL.load(path))` (no-op until M2.4's omit
events, but reverses a merge's distal omissions once they exist); idempotent, returns
`{deleted,uuid,summary}`; api `DELETE /api/cells/{root_id}/annotations/{uuid}`. Frontend: module
`annIndex` Map(uuid→{tag,nm}) populated in `drawPoint` (so RESUMED marks are deletable too);
`deleteNearest()` picks the closest mark in nm within a **2500 nm** threshold → DELETE → `removePoint`
(`src.getReference(uuid)`→`src.delete(ref)`→`ref.dispose()` + `annIndex.delete`). Verified backend
add→list(4)→DELETE(deleted:true)→list(3)→re-delete(deleted:false); tsc --noEmit clean. neuroglancer
LocalAnnotationSource delete API (2.41): `getReference(id)` returns an addRef'd AnnotationReference,
`delete(reference)` drops it.
**M2.3 DONE + committed 5b5d22c (mark branch done + advance):** `x` (branch-level, like space;
gated on `currentPid`) marks the loaded branch reviewed, repaints the dropdown + a coverage row,
advances to the next to-review branch. Backend `service.mark_done(pid)` mirrors
`ProofreadSession._on_mark_done`: `l2 = tree.l2_ids_for_vertices(bp.vertices)` (FULL path incl. shared
proximal node — `path_state` classifies on `bp.vertices[1:]` so this is safe) → `wal.mark_visited(l2)`
+ `coverage.mark_visited(l2)`; returns `{path_id, next_path_id (=coverage.to_review[0] or None),
summary, branches}`; api `POST /api/cells/{root_id}/branches/{path_id}/done` (404 out-of-range).
Frontend `markDone()` POSTs → `renderBranches`+`renderSummary` (both factored out, also used on first
load) → `loadBranch(next)` or "cell complete"; `currentPid` set in `loadBranch`; coverage row
`#coverage`. **VISITS HAVE NO UNDO in the API** (WAL tombstone reverses only annotations/omits, not
visits). Verified in temp WAL (no live pollution): 49→48 to_review, branch 0→covered, next=1; HTTP
out-of-range→404 with summary unchanged. M2.1/2.2/2.3 all DONE — **M2 core complete except deferred M2.4**.
**M3 DONE (pause→live full-res EM + graphene seg).** On pause (idle) the browser swaps the sparse
mip1 tube for full-res mip0 EM + the REAL graphene segmentation; on play swaps back. Sliced:
- **M3.1 committed f494fb8 (live EM swap):** backend `service.live_sources()` → `{root_id,
  image_source, segmentation_source (middleauth+ prefixed), viewer_resolution_nm, token}`; api
  `GET /api/cells/{root_id}/live-sources`. KEY: for minnie65_public the **EM is a PUBLIC S3
  precomputed** (`precomputed://https://bossdb-open-data.s3.amazonaws.com/...`, NO auth); mip0 =
  **[4,4,40]nm** (vs the [16,16,40] tube). Frontend fetches live-sources on open; `setLive(on)`
  toggles tube↔live; frame loop reconciles `wantLive = phase==="play" && !running && !scrubbing`
  (live ONLY while truly paused on a branch — graphene paints only when idle, mip0 too heavy in motion).
- **M3.2 committed 7d041d8 (graphene seg + middleauth token injection):** the auth-risky bit. The
  npm neuroglancer middleauth flow (NOT python's `'middleauthapp'` register): graphene kvstore →
  `credentialsManager.getCredentialsProvider("middleauthapp", <seg origin>)` →
  `MiddleAuthAppCredentialsProvider` fetches `/auth_info`→login_url→ `"middleauth"` provider which
  checks `localStorage[auth_token_v2_<login_url>]` else OAuth popup; consumed by
  `fetchOkWithOAuth2CredentialsAdapter` as `Authorization: <tokenType> <accessToken>`. We INJECT
  our CAVE token by OVERRIDING the `"middleauthapp"` provider (skips /auth_info + popup +
  localStorage): `registerDefaultCredentialsProvider("middleauthapp", () => new P())` where
  `P extends CredentialsProvider<any>` with `get = makeCredentialsGetter(async () => ({tokenType:
  "Bearer", accessToken: token}))` — imports from `neuroglancer/unstable/credentials_provider/
  {index,default_manager}.js` (./unstable/* → ./lib/*). MUST register BEFORE setupDefaultViewer()
  (viewer builds its manager from the global registry via `getDefaultCredentialsManager()` at
  creation, viewer.js:325). Seg layer = `makeLayer(... {type:"segmentation", source: seg,
  segments:[root_id_string]})`. Token over HTTP→127.0.0.1 only; sent to graphene over HTTPS.
  CONFIRMED in-browser: seg paints on pause, NO login popup.
- **M3 live-layer lifecycle committed 93931c5:** live layers are ADDED on pause / REMOVED on play
  (not created-once-and-hidden) — a hidden layer keeps chunk sources resident + graphene does
  background work, competing with the tube buffering sweep; `removeManagedLayer` disposes them so
  motion has zero contenders. `liveEmLayer`/`liveSegLayer` refs held only while paused.
PERF NOTE (investigated 2026-06-01): user reported "building tube slower after M3". MEASURED: NOT a
regression — backend camera fetch is 6ms cached / **14-22s for genuine first-visit uncached builds**
(normal, `fill_branch` downloads tube chunks; matches the 10-30s noted at M1). The client buffering
sweep is **time-boxed by fixed sleep(120) steps** (constant wall-time, load-independent). The "slow"
was just hitting fresh branches that build for the first time. Real first-visit latency would need
BACKGROUND PRE-BUILDING of upcoming branches (deferred; annotator.py had a "pre-render to_review" concept).
**M4 IN PROGRESS — 3D skeleton guidance (modeled on guidebook `ceesem/guidebook`).** Plan in
`~/.claude/plans/jolly-wondering-feather.md`; design discussion in `docs/proofreading-workflow.md`
("3D skeleton guidance"). guidebook = static skeleton-guidance proofreading tool (root→skeleton→mark
branch points [merge/split] + end points [extend], order proximal→distal by cover-path/distance-to-root,
emit nglui link). DECISION: reuse guidebook's RECIPE (we already have every primitive in SkeletonTree:
child_count>=2=branch pts, child_count==0=tips, branch_paths=cover paths, parent[]=dist-to-root) — NO
new deps; do NOT import its Flask app / nglui (we're browser-native). **pcg_skel (→ meshparty.Skeleton,
which carries these primitives natively) EARMARKED for Phase C** re-skeletonization of edited cells (the
skeletoncache cached skeleton is stale after an edit). em extra is still just caveclient+cloud-volume;
pcg_skel/meshparty/nglui all ABSENT.
**M4.1 DONE + committed 48b3ac2 (3D skeleton backdrop):** `service.live_sources()` adds `skeleton_source`
= `precomputed://middleauth+{client.skeleton.server_address}/skeletoncache/api/v{api}/{datastack}/precomputed/
skeleton/` (verified resolves to a neuroglancer_skeletons info w/ our token; SAME origin as graphene seg so
the M3.2 token authorizes it). Frontend setupViewer adds a SKELETON-ONLY `segmentation` layer
(segments:[root_id]) + layout `xy`→`xy-3d` + `projectionScale:400000` (frames whole cell; independent of
2D crossSectionScale 0.12). Skeleton-only ⇒ renders ONLY in 3D, nothing in 2D cross-section (EM review view
stays clean); always-on. Falls back to plain xy if skeleton_source null. CAVEAT: 3D perspective re-centers
on nav position → skeleton drifts during glide (mitigated by zoomed-out projectionScale).
**M4.2 DONE + committed 16bfecc (branch + end points):** `service.skeleton_features()` → branch points
(`child_count>=2`) + end points/tips (`child_count==0`) from SkeletonTree, each `{xyz_nm, path_id}` (path
that ENDS at that vertex; soma root→null); api `GET …/skeleton-features`. Verified 23 branch + 26 end = 49
(= n cover paths; each path ends at a branch pt or tip). Frontend: 2 always-on local point layers created at
rank-3 with the tag layers — `skel:branch` (magenta #cc66ff) + `skel:end` (white #fff) — via
`drawSkelPoint`/`drawSkeletonFeatures` (NOT in annIndex, so `d` won't delete them).
**M4.3 + M4.4 DONE + committed c1553f4 (proximal→distal ordering + user-chosen root):**
- M4.3 ordering: `service._dist_to_root()` (geodesic nm root→vertex, BFS, cached) + `_branch_order()`
  (branches sorted by start-node soma-distance, ties→id; guidebook proximal-first sweep). `branches()`/
  `mark_done` next+checklist use it; `branch_metadata` adds `dist_to_root_nm`; frontend dropdown shows `<n>µm`.
- M4.4 choose root: **THIS CELL (864691135413357554) HAS NO SOMA** — soma_pt None, all 1171 verts dendrite,
  skeleton service defaulted root to vertex 0 (a degree-1 tip) → "proximal" was an arbitrary tip. Fix:
  `service.set_root(xyz_nm)`/`_reroot_at` rebuilds the tree rooted at nearest vertex (re-derives branch
  decomposition + ordering + merge-prune direction); coverage (L2-keyed) preserved; `_clear_branch_markers`
  resets per-branch `.done` (spatial chunks reused, re-fill fast); api `POST /api/cells/{root_id}/root`.
  `skeleton_features` switched to **UNDIRECTED degree** (deg>=3 branch, deg==1 tips, like guidebook
  `*_undirected`) so markers are anatomical + ROOT-INVARIANT. Frontend: a **"set root" button** arms a
  ONE-SHOT mode (so a stray click can't re-root — user explicitly requested a button, not bare click); next
  data-panel click snaps to nearest marker (<=2µm) → POST → redraw ordered checklist/summary + restart at new
  most-proximal branch + disarm. Verified: re-root snaps to clicked tip (dist 0), order monotonic, markers
  unchanged.
**PERSIST CHOSEN ROOT DONE (2026-06-02, UNCOMMITTED — awaiting user OK):** the chosen root now survives a
reload. New WAL event `set_root {xyz_nm}` (`wal.WAL.set_root`; replayed into `WalState.root_xyz`, LAST WINS;
no tombstone needed). `service.set_root` persists the SNAPPED vertex xyz (exact → `nearest_vertex` re-snaps
to it). `service.__init__` loads `WalState.root_xyz` (`self._resume_root_xyz`) and, at the END of __init__
(after caches + `_epoch` exist), `_reroot_at(root_xyz, clear_markers=False)` — re-derives the SAME
decomposition the .done markers were written under, so on-disk tube builds stay valid. BACKEND-ONLY: the
frontend draws no root marker (branch/end markers are root-invariant) so the resumed ordering just appears;
no JS change. Verified via curl: re-root at branch-pt vertex 655 → order [4,5,6,31,32] + WAL set_root line
written → RESTART backend (clears in-mem sessions) → re-open → order resumes [4,5,6,31,32] (not default
[0,33,34]). Test-injected set_root stripped from the dev WAL after (cell back at default root).
**BACKGROUND PRE-BUILD DONE + committed e7dccbf:** hides the long-branch first-visit build (scales with
length: branch 0 = 214µm ~99s, branch 6 = 72 nodes ~117s; tiny branches ~14-22s — NOT a regression) behind
review time. `camera_path(pid)` queues `_queue_prebuild(pid)` → builds the next `prebuild_ahead` (=2)
to-review branches in `_branch_order()` (the REVIEW order, proximal→distal — NOT numeric ids) on a single
bg ThreadPoolExecutor worker; `_prebuild(pid, epoch)` shares `_branch_locks[pid]` with on-demand fetch (no
double build). RE-ROOT SAFETY via `self._epoch`: `_reroot_at` bumps epoch + invalidates order cache + clears
.done markers → stale pre-builds abort; the fill + epoch-recheck + stale-marker cleanup all happen INSIDE the
per-branch lock (closed a ~ms window where an on-demand fetch could serve a stale branch). `close()` shuts the
executor. camera response carries `prebuilding:[pids]`; frontend shows "↻ pre-building #N" in the gliding status.
Verified: load branch 0 → queues [33,34] → 33 builds in bg → re-fetch cached/instant.
**M2.4 OMIT-BRANCH DONE + committed 20f55b7 (per-branch, X-crossing safe).** Reframed from the original
"merge error at a vertex prunes ALL distal" (WRONG at an X crossing — would omit the cell's own
continuation, which is also distal): instead `service.omit_branch(pid)` omits THIS branch + its distal
DESCENDANTS only, via `l2_ids_for_vertices(subtree_mask(bp.vertices[1], include_root=True))` — excludes the
shared junction + SIBLINGS, so at a crossing the cell's continuation (a sibling) stays to_review; user omits
each foreign arm individually. Anchored to a 'merge error' mark at the junction (`wal.add_annotation` →
`mark_omitted(l2, because_uuid)` + `coverage.mark_omitted`) so it PERSISTS + records the split location +
auto-advances to next to-review. api `POST /api/cells/{root_id}/branches/{path_id}/omit`. REVERSIBLE:
`delete_annotation` (now also returns `branches`) tombstones the mark → coverage rebuild drops the omit →
branch returns to PRIOR state (covered/to_review). Frontend: "omit branch" button + `o` key (branch-level,
gated on currentPid) draws the red mark + repaints dropdown/summary + advances; `deleteNearest` repaints on
un-omit. Verified: omit branch 33 → 33+45 descendants omitted (1471 L2), siblings 34/35 preserved; delete
mark → exact revert. CAVEAT: "distal" is relative to the chosen root → set root toward soma side first (M4.4).
NEXT (open follow-ups): M4.5 current-branch highlight (line annotations, reuse camera points_nm) · M4.6
clickable marker→loadBranch. (persist chosen root: DONE 2026-06-02, see above.)
Supervoxel resolve (batch points→supervoxels for annotations) still unbuilt; merge-prune now superseded by omit_branch.
M2.4 (merge-prune + supervoxel resolve) still DEFERRED. Other possible: background pre-build of next branches;
CAVE annotation-table sync.

**TUBE-BUILD HANG FIX (2026-06-02, UNCOMMITTED — awaiting user OK):** a long branch (branch 45,
87 nodes/146µm) hung the build for MINUTES with ZERO chunks written, then every page reload/retry
queued behind `_branch_locks[45]` until the FastAPI sync threadpool (~40) saturated → ALL sync
endpoints timed out (even a cached branch); `/healthz` still 200 (async). 165 OS threads. ROOT
CAUSE: the tube EM+seg data is served from ONE host `storage.googleapis.com` whose urllib3 pool is
~10, but `tube.fill_branch` ran EM + target fills as 2 CONCURRENT executors × 16 workers = up to 32
simultaneous CloudVolume reads, and the reads have NO socket timeout — pool thrashes (`Connection
pool is full, discarding connection` ×N), and `_fill_chunks` waited for EVERY box via `ex.map`, so a
wedged read hung the whole build forever. FIX (`tube.py`): (1) fill EM then target SEQUENTIALLY at
`workers=8` (peak 8 ≤ pool 10, no thrash); (2) `_fill_chunks` now submits all boxes + waits with
`as_completed(futs, timeout=budget_s)` (`budget_s=180`/fill), counts still-pending boxes as failed on
timeout, and `ex.shutdown(wait=False, cancel_futures=True)` so a wedged read can't re-hang on shutdown
(leaked worker unwedges on its own; branch not marked `.done` → retried). VERIFIED: branch 45 now
builds CLEAN in 106s (.done written), re-fetch 66ms, API stays responsive (branch 0 4ms). NOTE: pool-
full warnings still log (CloudVolume's own internal connections exceed 10) but are now HARMLESS — the
build completes. Sequential@8 (~106s) is comparable to the old measured ~117s; pre-build hides it.

**Runtime learnings (hard-won, keep):**
- Graphene seg needs browser auth: source must be `graphene://middleauth+https://...`
  AND register a neuroglancer credentials provider for key `'middleauthapp'` returning
  the CAVE token (`client.auth.token`), else `/credentials` 500s with KeyError.
- neuroglancer renders the segmentation **only when the camera is idle** → a LIVE
  fly-through must be smooth glide + a short **rest** at each node (dwell) for the seg to
  paint; there's no continuous-motion trick (manual scroll works only via its idle gaps).
  This is exactly why review pivoted to pre-render (above).
- `viewer.screenshot(size=[w,h])` (sync) returns ScreenshotReply with `.image` (PNG bytes);
  `async_screenshot` takes no size. Run the sync call in a thread w/ join-timeout so it
  can't hang when no browser is attached.
- Drive the camera with `set_state` of a nav-only interpolated state (per-frame `txn`
  is jerky; `neuroglancer.ViewerState.interpolate` crashes on ImageLayer in 2.41.2).
- Default to **axis-aligned XYZ sections** (`orient_to_path=False`); the oriented
  cross-section ⊥ neurite is opt-in (oblique slices stream much slower).

Key locked decisions:
- **v1 = two-phase**: annotate the whole cell first (read-only), edit manually
  second (outside the tool), then re-enter with the new root id to reconcile.
- Input = root id + CAVEclient **materialization version**.
- Durable state anchored to **stable ids**: annotations record click **xyz** (source of
  truth), supervoxel id *derived* via CloudVolume `scattered_points`; coverage→L2 id (via
  the skeleton's `mesh_to_skel_map`, since L2 ids are NOT 1:1 with vertices). Cell identity
  = a **seed supervoxel**. Root id is only a transient session handle.
- API verified live (2026-05-28, minnie65_public): `get_skeleton(output_format='dict')`
  → vertices(nm)/edges(tree)/compartment/radius/lvl2_ids+mesh_to_skel_map/meta(soma_pt);
  needs cloudvolume. Point→supervoxel→root round-trip confirmed via
  `client.info.segmentation_cloudvolume(agglomerate=False)` (mip0 [8,8,40] nm) +
  `chunkedgraph.get_root_id`. EM deps = `em` extra (caveclient + cloud-volume); token at
  ~/.cloudvolume/secrets/cave-secret.json.
- Persistence = **local append-only JSONL WAL** (fsync per event); CAVE-table sync deferred.
- Path (for cross-section mode, opt-in): L2 skeleton → resample → edge tangents →
  rotation-minimizing frame → `crossSectionOrientation` from [N,B,T]. Default is
  axis-aligned (no orientation), just move position.
- Annotations (v1) = **single tagged points**, 4 tags: merge error / split error / extend / question.
- **merge error** → fly-through terminates early + distal subtree pruned (`omitted`).
- Reuses [[flythrough-cleanup]]'s `proofreading.FlyThrough` / `FlyThroughControls`.

**M4.5 HIDDEN (2026-06-11, UNCOMMITTED):** current-branch skeleton highlight suppressed. Root cause: branch-switch re-draw broken — `drawBranchHighlight` adds line annotations to a `local://annotations` layer, but after switching branches (re-`restoreState`?) the annotations do not redraw. Investigation hit a second sub-bug: `LocalAnnotationSource.delete(id)` silently no-ops; the correct call is `src.getReference(id)` → `src.delete(ref)` → `ref.dispose()`. Even after fixing the delete API the switch-time redraw was still broken; root cause unknown. Feature entirely commented out in `web/src/main.ts`: `SKEL_HIGHLIGHT_LAYER`, `SKEL_HIGHLIGHT_COLOR`, `highlightIds`, `drawBranchHighlight`, and all call sites. TypeScript-clean (no unused-var errors). M4.6 (clickable branch/end markers → `loadBranch`) still DEFERRED.

**PHASE B BUILT (2026-06-11, UNCOMMITTED).** `~/.claude/plans/jolly-wondering-feather.md`. Plain-HTML annotation queue + Spelunker link generator — NO neuroglancer in the review page; Spelunker is the editor.

Files added/changed:
- **`web/review.html`** — annotation table (#, type, position µm, supervoxel, spelunker ↗ link, done/todo status); dark theme matching main.ts.
- **`web/src/review.ts`** — fetches WAL annotations (sorted merge→split→extend→question), renders table, event-delegates done/todo toggle clicks to `toggleDone(uuid)` (optimistic update → POST → reconcile with server truth → revert on failure). No neuroglancer imports.
- **`proofreading/em/wal.py`** — `WalState.done_uuids: Set[str]`; `WAL.set_annotation_done(uuid, done)` writes `{"event":"ann_status", "uuid":"...", "done":bool}`; replay handler updates `done_uuids` (last-write-wins).
- **`proofreading/em/service.py`** — `_SPELUNKER_DATASTACK="minnie65_phase3_v1"`; `_spelunker_client()` lazy-caches a second `CAVEclient('minnie65_phase3_v1')` (separate from the session's `minnie65_public` client — avoids WAL session mismatch); `_spelunker_url(xyz_nm)` uses nglui `ViewerState(...).add_layers_from_client(...).to_url(target_site="spelunker")` → seg source derives to `minnie3_v1` automatically from the phase3 client; `list_annotations()` includes `done` + `spelunker_url`; `toggle_annotation_status(uuid)` flips done state + writes WAL event; `resolve()` includes `spelunker_url` in returned annotations.
- **`proofreading/em/api.py`** — `POST /api/cells/{root_id}/annotations/{uuid}/status` → `toggle_annotation_status`; `POST /api/cells/{root_id}/resolve` → `resolve()`.
- **`pyproject.toml`** — `em` extras adds `"nglui>=4.0.0"`.

Key technical decisions:
- **Datastack split**: WAL sessions use `minnie65_public`; Spelunker URL generation uses a lazily-cached `CAVEclient('minnie65_phase3_v1')` so nglui picks up `minnie3_v1` as the segmentation source. Combining them would create a new empty WAL session under the phase3 key.
- **Spelunker state**: nglui 4.7.3 `ViewerState(client, position, ...).add_layers_from_client(client).to_url("spelunker")` → inline JSON URL at `https://spelunker.cave-explorer.org/#!{state}`. Position in mip0 voxels (divide nm by `[4,4,40]`).
- **Done/todo**: fully WAL-persisted; `done` field on every annotation API response; `doneSet` seeded from WAL on page load (survives browser refresh/different machines).

VERIFIED: backend at `http://127.0.0.1:8000` (healthz ok), frontend Vite dev server at `http://localhost:5173`. `review.html?root=864691135413357554` loads 49 annotations with Spelunker links pointing to `minnie3_v1`. Done/todo toggles write `ann_status` to WAL and persist on reload.

NOT YET COMMITTED — M4.5 hide + entire Phase B diff is uncommitted, awaiting user OK.

**Why:** big multi-session feature; resume from the branch + plan doc, don't re-derive.
**How to apply:** Phase A is built and committed on the branch (not master). Next:
Phase B (manual edit step) + Phase C (re-entry reconcile via L2-id coverage), and a
possible CAVE annotation-table sync. Run the demo: `uv run --extra em jupyter lab`
(or the `.venv` kernel in VS Code).
