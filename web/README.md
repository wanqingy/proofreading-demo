# Phase 0 spike — browser-native fly-through

Proves the next-architecture hypothesis before we build on it: **an embedded neuroglancer
whose camera is animated entirely client-side (a `requestAnimationFrame` loop mutating
`navigationState.pose.position`) glides smoothly over a served local tube and does NOT
degrade over a long session** — the ceiling the python↔neuroglancer 30 fps full-state-sync
path hit (see `docs/proofreading-workflow.md`, "Known limitation & next architecture").

**There is no python in the animation loop.** Python only (1) serves the precomputed tube
and (2) dumps the camera path as static JSON. Everything else runs in the browser.

## Run (M1 backend)

Two processes. From the repo root:

```bash
# 1. the FastAPI backend — opens cells, lists branches, builds+serves tubes. Leave running.
uv run --extra em --extra serve python -m proofreading.em.serve
#    -> http://127.0.0.1:8000  (JSON API under /api, tube precomputed under /tube)
 
# 2. the frontend dev server
cd web && npm install   # first time only
npm run dev             # -> http://localhost:5173
```

Open <http://localhost:5173>. It opens cell `864691135572530981` (override with
`?root=<id>` / `?datastack=<ds>` / `?api=<url>` query params), populates a **branch
picker**, and glides the first to-review branch: a buffering pre-pass caches the branch,
then it ping-pongs the camera over the `em` (grayscale) + `tgt` (red) tube. Pick another
branch from the dropdown to jump to it. The HUD shows uptime / frames / **fps (now)** /
**fps (min)** and the buffer/glide status.

The earlier throwaway `spike_export.py` (single hard-coded branch + its own static server)
is superseded by the backend; it's left only as a standalone reference.

## What to watch

- **Smoothness:** the EM should stay sharp *while moving* (the local tube renders during
  motion — same win as the notebook, now without python driving frames).
- **Non-degradation — the real test:** leave it ping-ponging for 10–20 min. `fps (min)`
  should hold near 60 and **not** creep down. In the python path it decayed over a session;
  here there is no per-frame `set_state`, so it should stay flat.
- Pause the camera (HUD button) and drag/scroll — it's the full neuroglancer, free-nav and
  all panels work.

## Notes

- The tube cache must already exist (built by the notebook workflow reviewing paths 7–11):
  `proofread_sessions/tube_cache/minnie65_public/864691135572530981/{em,tgt}`.
- `camera_path.json` is generated (gitignored) — it bakes in the auto-assigned server port.
  Re-run `spike_export.py` if you restart it, then hard-reload the page.
- Stack: Vite + the `neuroglancer` npm package (2.41.2, same as the python side). neuroglancer
  is excluded from Vite's dep-optimizer so its `import.meta.url` web workers survive (see
  `vite.config.ts`).

## If the spike holds

This `web/` becomes the seed of the browser-native tool: the camera/HUD logic in
`src/main.ts` is the kernel of the fly-through UI, and `spike_export.py` is the stub of the
FastAPI backend (`proofreading/em` engine serving tubes + camera paths, recording
annotations to the WAL).
