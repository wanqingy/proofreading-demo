"""Review pre-rendered branch frames, then jump the live viewer to act.

The "spot-then-act" loop (Phase A): scrub/play the pre-rendered frames (smooth, with the
segmentation visible in every frame) to **find** an error, then hit "-> viewer" to jump
the live neuroglancer viewer to that exact node so you can drop the annotation / make the
edit there. Frame ``i`` maps to a skeleton node via the manifest, so the jump is exact.

Playback is pure client-side image swapping (a ``Play`` timer ``jslink``ed to a slider),
so it never touches neuroglancer and can't jank. Frames are optionally downscaled in
memory (Pillow, pulled in by cloud-volume) to keep the comm light during playback.
"""

from __future__ import annotations

import io
import os
from typing import Callable, List, Optional

import numpy as np


def _maybe_downscale(raw: bytes, max_width: Optional[int]) -> bytes:
    """Downscale a PNG to ``max_width`` and re-encode (JPEG) to keep playback light.

    Returns ``raw`` unchanged if Pillow isn't available or no resize is needed.
    """
    if not max_width:
        return raw
    try:
        from PIL import Image as PILImage
    except Exception:
        return raw
    try:
        im = PILImage.open(io.BytesIO(raw))
        if im.width <= max_width:
            return raw
        h = int(im.height * max_width / im.width)
        im = im.convert("RGB").resize((max_width, h), PILImage.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return raw


class BranchPlayer:
    """ipywidgets player over one branch's pre-rendered frames.

    Parameters
    ----------
    manifest:
        A manifest dict from :func:`proofreading.em.render.render_states` /
        :func:`~proofreading.em.render.load_branch_frames` (needs ``_dir``).
    on_goto:
        Optional ``f(node_index, xyz_nm)`` invoked when "-> viewer" is clicked, to jump
        the live viewer to the current frame's node (wired by the session).
    fps:
        Playback frame rate.
    max_width:
        Downscale frames to this width for display (smoother playback); ``None`` = full res.
    """

    def __init__(
        self,
        manifest: dict,
        on_goto: Optional[Callable[[int, Optional[list]], None]] = None,
        fps: int = 10,
        max_width: Optional[int] = 700,
    ):
        self.manifest = manifest
        self.on_goto = on_goto
        self.fps = int(fps)
        self._dir = manifest.get("_dir")
        # only frames that actually captured an image are playable
        self.frames: List[dict] = [f for f in manifest["frames"] if f.get("captured")]
        if not self.frames:
            raise ValueError("manifest has no captured frames -- render first")
        self._imgs: List[bytes] = []
        for f in self.frames:
            with open(os.path.join(self._dir, f["file"]), "rb") as fh:
                self._imgs.append(_maybe_downscale(fh.read(), max_width))

    # ------------------------------------------------------------------ #
    def _label(self, i: int) -> str:
        f = self.frames[i]
        node = f.get("node_index", "?")
        xyz = f.get("xyz_nm")
        gate = "" if f.get("loaded", True) else "  ⚠ partial-load"
        if xyz is not None:
            xyz = "[" + ", ".join(f"{c:,.0f}" for c in xyz) + "] nm"
        return f"frame {i}/{len(self.frames) - 1}   node {node}   {xyz}{gate}"

    def widget(self):
        """Build and return the player widget (call inside a notebook cell)."""
        from ipywidgets import Button, HBox, Image, IntSlider, Label, Play, VBox, jslink

        fmt = "png" if self._imgs[0][:4] == b"\x89PNG" else "jpeg"
        img = Image(value=self._imgs[0], format=fmt)
        img.layout.width = "auto"
        slider = IntSlider(min=0, max=len(self.frames) - 1, value=0, description="frame",
                           continuous_update=True, readout=False)
        play = Play(min=0, max=len(self.frames) - 1, value=0,
                    interval=int(1000 / max(1, self.fps)))
        jslink((play, "value"), (slider, "value"))  # client-side timer drives the slider
        label = Label(value=self._label(0))
        goto = Button(description="→ viewer (annotate this node)", button_style="success",
                      layout={"width": "260px"})

        def on_change(change):
            i = int(change["new"])
            img.value = self._imgs[i]
            label.value = self._label(i)

        slider.observe(on_change, names="value")

        def on_goto(_b):
            f = self.frames[int(slider.value)]
            if self.on_goto is not None:
                self.on_goto(int(f.get("node_index", slider.value)), f.get("xyz_nm"))

        goto.on_click(on_goto)
        return VBox([img, HBox([play, slider]), HBox([goto, label])])

    def show(self):
        from IPython.display import display

        w = self.widget()
        display(w)
        return w
