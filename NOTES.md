# Project notes — proofreading-demo

_Last updated: 2026-05-28_

Working log for the cleanup of the Neuroglancer proofreading fly-through tooling.

> **EM proofreading workflow** is designed in [docs/proofreading-workflow.md](docs/proofreading-workflow.md)
> (vocabulary in [CONTEXT.md](CONTEXT.md), decisions in [docs/adr/](docs/adr/)). v1 = a
> two-phase *annotate-then-edit* tool over `minnie65_phase3_v1`. Not yet implemented.

## What this project does

Drives Neuroglancer fly-throughs along neuron skeletons (ExASPIM / EM) for
proofreading, from Jupyter. Started as exploratory notebooks; the reusable parts
now live in the [`proofreading/`](proofreading/) package.

## Done (2026-05-28)

Cleaned the exploratory notebooks into a reusable package and preserved the
originals.

- **`proofreading/` package**
  - `viewer.py` — Neuroglancer server + image/skeleton layers, incl. the x↔z
    axis-swap transform (`SKELETON_XZ_SWAP`).
  - `skeleton.py` — SWC load via navis, axis swap, `compute_path()` →
    `(positions, orientations)`. Orientation math delegates to `ac_ngl`.
  - `flythrough.py` — tween helpers (`interpolate_to`, `move_to`, `zoom_to`) +
    the threaded **`FlyThrough`** controller.
  - `controls.py` — **`FlyThroughControls`** ipywidgets button panel.
  - `context.py` — `nearby_skeleton_ids()` bounding-box helper.
- **`demo_proofreading.ipynb`** — clean end-to-end walkthrough.
- **`original/`** — every original notebook/artifact, moved via `git mv`
  (history preserved): `Neuroglancer/`, `EM_test/`, `swc/`, `util.py`,
  `test_flythrough*.ipynb`, etc. Nothing was deleted or overwritten.
- **`pyproject.toml`** (uv) — core deps `numpy`, `neuroglancer`, `ipywidgets`;
  `[skeleton]` extra adds `navis`.

### The interaction fix (the thing we were stuck on)

Old notebooks animated with a `for` loop + `time.sleep` on the kernel's main
thread → kernel blocked → Pause/Continue buttons couldn't fire until the loop
ended. `FlyThrough` runs the animation on a **background worker thread**; buttons
only flip thread-safe `threading.Event` state and return in <1 ms. Controls:
Play/Pause, Forward/Reverse, Step ◁/▷ (one node), scrubber (jump + live sync),
sec/step speed slider. Auto-pauses at both ends.

### Verified

- Whole package imports with real `neuroglancer` + `ipywidgets`.
- `FlyThrough` driven against a real headless neuroglancer viewer: non-blocking
  play, autoplay/pause/reverse/step/seek, end-clamping, and the real
  `ViewerState.crossSectionOrientation` / `voxel_coordinates` attributes all work.

## Open to-dos / follow-ups

- [ ] **Auto-update nearby skeletons on each step.** Wire
  `FlyThrough.on_index_change` to recompute `nearby_skeleton_ids()` around the
  current node and call `context.show_segments()`, so neighboring processes
  appear automatically while flying (instead of the current manual cell in the
  demo). _Deferred — picking this up another time._
- [ ] **Confirm the SWC name → segment-id offset.** The old notebooks flip-flopped
  between `-1` and `-2`. It's now the explicit `name_to_id_offset` arg on
  `nearby_skeleton_ids` (default `-1`). Verify the correct value for the data.
- [ ] **Run the full skeleton/orientation path end-to-end** on the Linux
  workstation: needs `ac_ngl` on `sys.path` and `uv sync --extra skeleton`
  (navis). Only the threading + viewer layers have been exercised on this Mac.

## Environment notes

- `uv` project, Python 3.12, env in `.venv`.
- `ac_ngl` (lab helper: precomputed generation + tangent/quaternion math) is not
  on PyPI — add its parent dir to `sys.path`.
- Precomputed volumes are served over `bigkahuna.corp.alleninstitute.org`.
