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
