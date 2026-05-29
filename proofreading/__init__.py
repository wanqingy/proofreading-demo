"""Reusable Neuroglancer proofreading / fly-through toolkit.

This package cleans up the exploratory notebooks (``ExASPIM_Tiff_in_NG-*.ipynb``)
into importable building blocks:

- :mod:`proofreading.viewer`     -- start a Neuroglancer server and load layers.
- :mod:`proofreading.skeleton`   -- load SWC skeletons and compute per-node camera orientations.
- :mod:`proofreading.flythrough` -- low-level state interpolation + the thread-based
  :class:`~proofreading.flythrough.FlyThrough` controller (reliable pause/resume/forward/reverse).
- :mod:`proofreading.controls`   -- ipywidgets UI wired to a ``FlyThrough``.
- :mod:`proofreading.context`    -- find skeletons near the current camera position.

Typical use (in a Jupyter notebook)::

    import proofreading as pf

    viewer = pf.make_viewer(port=9998)
    pf.load_image_layer(viewer, "ExASPIM", "/ACdata/Users/wanqing/exaSPIM/precomputed/")
    pf.load_skeleton_layer(viewer, "skeletons", "/ACdata/Users/wanqing/Neuroglancer/ExASPIM/skeletons/")

    skel = pf.load_skeleton(".../0002.swc", swap_xz=True)
    positions, orientations = pf.compute_path(skel)

    fly = pf.FlyThrough(viewer, positions, orientations)
    pf.FlyThroughControls(fly)        # renders the button panel
"""

from .viewer import make_viewer, load_image_layer, load_skeleton_layer, shareable_url
from .skeleton import load_skeleton, swap_xz, compute_path, nodes_to_positions
from .flythrough import FlyThrough, interpolate_to, move_to, zoom_to, build_target_state
from .context import bounding_box, nearby_skeleton_ids

try:  # ipywidgets is only needed for the interactive UI
    from .controls import FlyThroughControls
except ImportError:  # pragma: no cover - headless environments
    FlyThroughControls = None

__all__ = [
    "make_viewer",
    "load_image_layer",
    "load_skeleton_layer",
    "shareable_url",
    "load_skeleton",
    "swap_xz",
    "compute_path",
    "nodes_to_positions",
    "FlyThrough",
    "interpolate_to",
    "move_to",
    "zoom_to",
    "build_target_state",
    "bounding_box",
    "nearby_skeleton_ids",
    "FlyThroughControls",
]
