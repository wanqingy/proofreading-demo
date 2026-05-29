"""Pre-render a branch fly-through to image frames (the "pregenerate" path).

Live neuroglancer paints the segmentation **only when the camera is idle**, so a smooth
*live* glide can never show the seg mask in motion -- the core bottleneck of the
interactive fly-through. We sidestep it: drive the viewer node-by-node, wait for each
frame to be fully GPU-resident (:func:`viewer.wait_until_loaded`), then capture a
screenshot. The resulting frames play back perfectly smoothly (it's just images) with
the segmentation visible in *every* frame -- because every frame was captured while idle.

Rendering is a slow batch pre-step (seconds per node); playback/review is then instant
and smooth (see :mod:`proofreading.em.review`). Frames + a ``manifest.json`` are written
under ``out_dir``; reload them later with :func:`load_branch_frames`.

This drives whatever browser is currently connected to the viewer (the open tab). A
fully headless batch render would use ``neuroglancer.webdriver`` (needs ``selenium`` +
chromedriver) -- not required here, left as a future option.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import neuroglancer
import numpy as np

from ..flythrough import build_target_state
from .viewer import wait_until_loaded

MANIFEST = "manifest.json"

Nav = Tuple[Sequence[float], Optional[Sequence[float]]]  # (voxel_coordinates, orientation|None)


def _capture(viewer: neuroglancer.Viewer, size, timeout: float) -> Optional[bytes]:
    """Synchronous screenshot at a fixed ``size``, run in a thread with a join-timeout.

    ``viewer.screenshot`` blocks until the connected browser replies; if no browser is
    attached it would hang forever, so we run it on a daemon thread and give up after
    ``timeout`` (returning ``None``). Returns the PNG bytes on success.
    """
    box: dict = {}

    def run():
        try:
            box["reply"] = viewer.screenshot(size=list(size))
        except Exception as exc:  # pragma: no cover - browser/transport errors
            box["err"] = exc

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive() or "reply" not in box:
        return None
    reply = box["reply"]
    img = getattr(reply, "image", None)
    if img is None and hasattr(reply, "screenshot"):
        img = reply.screenshot.image
    if isinstance(img, str):  # some transports return base64
        img = base64.b64decode(img)
    return img


def render_states(
    viewer: neuroglancer.Viewer,
    nav: Sequence[Nav],
    out_dir: str,
    *,
    size: Tuple[int, int] = (900, 900),
    load_timeout: float = 20.0,
    capture_timeout: float = 30.0,
    settle: float = 0.0,
    meta: Optional[Sequence[dict]] = None,
    extra_manifest: Optional[dict] = None,
    on_progress: Optional[Callable[[int, int, bool], None]] = None,
    force: bool = False,
) -> dict:
    """Capture one fully-loaded screenshot per nav state into ``out_dir``.

    ``nav`` is ``[(voxel_coordinates, orientation_or_None), ...]`` -- the camera path
    (typically one resampled branch). For each state we ``set_state`` the camera, block
    on :func:`wait_until_loaded` (EM + seg fully drawn), optionally rest ``settle`` extra
    seconds, then grab a ``size`` screenshot to ``frame_XXXX.png``.

    ``meta`` (len == ``nav``) is merged per-frame into the manifest (e.g. ``node_index``,
    ``xyz_nm``); ``extra_manifest`` is merged at the top level (e.g. ``root_id``,
    ``path_id``). Returns the manifest dict (also written to ``manifest.json``), with a
    non-serialized ``_dir`` key pointing back at ``out_dir``.

    If a complete manifest already exists and ``force`` is False, the render is skipped
    and the existing manifest returned -- so re-running is cheap and only changed branches
    need regeneration after an edit.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(nav)

    if not force:
        existing = _load_if_complete(out_dir, n)
        if existing is not None:
            return existing

    frames: List[dict] = []
    for i, (vox, ori) in enumerate(nav):
        viewer.set_state(build_target_state(viewer, vox, ori))
        loaded = wait_until_loaded(viewer, load_timeout)
        if settle > 0:
            time.sleep(settle)
        img = _capture(viewer, size, capture_timeout)
        fname = f"frame_{i:04d}.png"
        rec = {"file": fname, "loaded": bool(loaded), "captured": img is not None}
        if meta is not None:
            rec.update(meta[i])
        if img is not None:
            with open(os.path.join(out_dir, fname), "wb") as fh:
                fh.write(img)
        frames.append(rec)
        if on_progress is not None:
            try:
                on_progress(i + 1, n, rec["captured"])
            except Exception:
                pass

    manifest = {
        "size": list(size),
        "n_frames": n,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames": frames,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["_dir"] = out_dir
    return manifest


def _load_if_complete(out_dir: str, expected_n: int) -> Optional[dict]:
    """Return a previously-rendered manifest iff it's complete (all frame files present)."""
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            m = json.load(fh)
    except Exception:
        return None
    frames = m.get("frames", [])
    if expected_n is not None and len(frames) != expected_n:
        return None
    for f in frames:
        if not f.get("captured"):
            return None
        if not os.path.exists(os.path.join(out_dir, f["file"])):
            return None
    m["_dir"] = out_dir
    return m


def load_branch_frames(out_dir: str) -> dict:
    """Load a manifest written by :func:`render_states` (sets ``_dir``)."""
    with open(os.path.join(out_dir, MANIFEST)) as fh:
        m = json.load(fh)
    m["_dir"] = out_dir
    return m


def encode_video(manifest: dict, out_path: Optional[str] = None, fps: int = 12) -> Optional[str]:
    """Optionally stitch the frames into an mp4 (needs ``imageio[ffmpeg]``).

    Returns the written path, or ``None`` if imageio/ffmpeg isn't available (the
    interactive :class:`~proofreading.em.review.BranchPlayer` is the primary review path;
    this is just for sharing a clip). The frames themselves are the source of truth.
    """
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    out_dir = manifest["_dir"]
    files = [os.path.join(out_dir, f["file"]) for f in manifest["frames"] if f.get("captured")]
    if not files:
        return None
    if out_path is None:
        out_path = os.path.join(out_dir, "flythrough.mp4")
    frames = [iio.imread(f) for f in files]
    try:
        iio.imwrite(out_path, frames, fps=fps)
    except Exception:
        return None
    return out_path
