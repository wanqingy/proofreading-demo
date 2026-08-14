# proofreading-demo

Drive Neuroglancer fly-throughs along neuron skeletons for proofreading ExASPIM
(and EM) data, from Jupyter.

This repo started as a set of exploratory notebooks. The reusable parts have
been cleaned into the [`proofreading/`](proofreading/) package; all the original
exploratory work is preserved untouched under [`original/`](original/)
(`original/Neuroglancer/ExASPIM_Tiff_in_NG-*.ipynb`, etc.).

## Install

```bash
uv sync                    # core deps: numpy, neuroglancer, ipywidgets
uv sync --extra skeleton   # also installs navis (SWC loading / orientations)
```

Two pieces live **outside** PyPI and must be reachable for the full pipeline:

- **`ac_ngl`** — the lab's helper module (precomputed generation, tangent →
  quaternion math). Add its parent dir, e.g.
  `sys.path.append("/home/wanqing.yu/AC_Project/ac_visualization/")`.
- The **precomputed volumes** served over `bigkahuna.corp.alleninstitute.org`.

The package imports fine without these; functions that need them raise a clear
`ImportError` only when called.

## Package layout

| Module | What it does |
| --- | --- |
| [`viewer.py`](proofreading/viewer.py) | Start the Neuroglancer server, add image / skeleton layers (incl. the x↔z axis-swap transform). |
| [`skeleton.py`](proofreading/skeleton.py) | Load SWC with navis, swap axes, build `(positions, orientations)` for the camera path. |
| [`flythrough.py`](proofreading/flythrough.py) | Tween helpers + the threaded **`FlyThrough`** controller. |
| [`controls.py`](proofreading/controls.py) | **`FlyThroughControls`** — the ipywidgets button panel. |
| [`context.py`](proofreading/context.py) | Find skeleton segments near the current camera position. |

## The interactive controls

The old notebooks animated the camera with a `for` loop + `time.sleep` on the
kernel's main thread. That **blocks the kernel**, so the Pause / Continue
buttons couldn't fire until the loop finished — the thing you got stuck on.

`FlyThrough` fixes this by running the animation on a **background worker
thread**. Button callbacks only flip thread-safe state (`threading.Event`s), so
they return instantly and stay responsive mid-flight:

- **Play / Pause** — toggle autoplay; pause takes effect at the end of the
  current node-to-node transition (sub-second).
- **Forward / Reverse** — set the travel direction and play.
- **Step ◁ / ▷** — advance exactly one node (auto-pauses first).
- **Scrubber** — jump to any node; stays in sync while autoplaying.
- **sec/step slider** — change speed live.

Autoplay auto-pauses at either end of the skeleton (no index overrun).

## Quick start (Jupyter)

```python
import sys; sys.path.append("/home/wanqing.yu/AC_Project/ac_visualization/")
import proofreading as pf

viewer = pf.make_viewer(port=9998)
pf.load_image_layer(viewer, "ExASPIM",
    "/ACdata/Users/wanqing/exaSPIM/precomputed/", shader_range=[15, 71])
pf.load_skeleton_layer(viewer, "skeletons",
    "/ACdata/Users/wanqing/Neuroglancer/ExASPIM/skeletons/")
print(viewer)   # open this URL

skel = pf.load_skeleton(".../skeletons_10/0002.swc", swap_xz=True)
positions, orientations = pf.compute_path(skel, k=15)

fly = pf.FlyThrough(viewer, positions, orientations, seconds_per_step=0.3)
pf.FlyThroughControls(fly)        # renders the button panel
```

See [`demo_proofreading.ipynb`](demo_proofreading.ipynb) for the full walkthrough,
including showing nearby skeletons around the current position.

When finished: `fly.stop()` then `neuroglancer.stop()`.

## Proofreading recorder (browser tool)

A lightweight browser tool for recording manual proofreading decisions: which segments
belong to one neurite, notes on individual segments, and the exact view — with one
"Record" button instead of hand-copying IDs and links. Backend: [`proofreading/annotate/`](proofreading/annotate/)
(an append-only JSONL log, replayed to reconstruct state — see [`log.py`](proofreading/annotate/log.py)).
Frontend: [`web/annotate.html`](web/annotate.html) + [`web/src/annotate.ts`](web/src/annotate.ts).

Run it:

```bash
# 1. backend -- leave running (defaults to proofread_sessions/annotate_records.jsonl)
uv run --extra serve python -m proofreading.annotate.serve
#    -> http://127.0.0.1:8001

# 2. frontend dev server
cd web && npm install   # first time only
npm run dev             # -> http://localhost:5173 (strictPort: true -- the OAuth client's
                         #    redirect URI is registered for this exact port, so it must land
                         #    here; kill anything else already bound to 5173 if it doesn't)
```

Open <http://localhost:5173/annotate.html>. Select all segments belonging to one neurite in
the embedded viewer, optionally leave a note by adding an annotation point linked to a
segment, type your name once (remembered per browser), and hit **Record**. The dropdown at
the bottom lists everything recorded so far — **open** (view read-only, new tab), **edit**
(reopen in this tab to update it in place), **del** (delete). Recording a segment already
owned by a different neurite prompts for confirmation (shared segment vs. mistake) instead
of silently overwriting.

**If the data source needs Google auth your own OAuth client can't get** (e.g. a Brainmaps
volume gated to a specific allowlist of client apps): use
[`web/public/ng-recorder.js`](web/public/ng-recorder.js) instead. It injects the same
Record/dropdown control bar into a page you don't control but are already signed into
(e.g. `neuroglancer-demo.appspot.com`) — paste its contents into the DevTools console (or
save as a DevTools Snippet: Sources tab → Snippets → run with Ctrl+Enter) on that page. It
duck-types segmentation/annotation layers instead of relying on fixed names, since layer
names on someone else's page aren't ours to control.

Both write to the same backend, so a shared team log works regardless of which entry point
people use. The `user` field on each entry is self-reported (not cryptographically verified)
for now.

## Reclaiming disk (tube cache)

Flying a cell downloads its EM/mask chunks into `proofread_sessions/tube_cache/<datastack>/<root_id>/`.
This grows fast — tens of GB per heavily-reviewed cell. It's all **derived** data, so deleting it
only costs re-download time; the `*.jsonl` WAL logs alongside it (every annotation you've made)
are never touched by this tool.

```bash
# what's on disk, and which mask volumes are stale
uv run python -m proofreading.em.clear_cache

# drop mask volumes that aren't at the mask mip in effect (keeps all EM -> nothing refetches)
uv run python -m proofreading.em.clear_cache --stale-masks -f

# one cell, or everything
uv run python -m proofreading.em.clear_cache --cell 864691135572530981 -f
uv run python -m proofreading.em.clear_cache --all -f
```

Every mode is a **dry run until you pass `-f`**. Deletion is hard-gated to paths strictly inside
`tube_cache`, and refuses outright if a `.jsonl` is found anywhere in the target.

There is deliberately **no "clear cache" button in the review UIs**: myelin coverage and
error-review coverage are separate dimensions (see `CONTEXT.md`), so a cell that looks "fully
reviewed" to one tool may be untouched by the other — clearing on that signal would force a long
re-cache the moment you start the second pass.
