"""Find skeletons near the current camera position.

Replaces the repeated "build a bbox around current_pos and collect skeleton ids"
cells. Also fixes the off-by-one churn from the notebooks by making the SWC
name -> segment id mapping an explicit, configurable argument.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def bounding_box(center: Sequence[float], size: Sequence[float]) -> Dict[str, List[float]]:
    """Axis-aligned box of ``size`` centered on ``center`` (both ``[x, y, z]``)."""
    half = [s / 2 for s in size]
    return {
        "min": [center[i] - half[i] for i in range(3)],
        "max": [center[i] + half[i] for i in range(3)],
    }


def nearby_skeleton_ids(
    skeletons,
    center: Sequence[float],
    size: Sequence[float] = (100, 100, 100),
    name_to_id_offset: int = -1,
) -> np.ndarray:
    """Return segment ids of skeletons with at least one node inside the box.

    Parameters
    ----------
    skeletons:
        A navis NeuronList (e.g. from ``navis.read_swc(directory)``).
    center, size:
        Box center and dimensions in ``[x, y, z]`` order.
    name_to_id_offset:
        Added to ``int(skeleton.name)`` to get the Neuroglancer segment id.
        The notebooks flip-flopped between ``-1`` and ``-2``; make it explicit
        here once you know the convention for your data (default ``-1``).
    """
    box = bounding_box(center, size)
    ids: List[int] = []
    for skel in skeletons:
        inside = (
            skel.nodes["x"].between(box["min"][0], box["max"][0])
            & skel.nodes["y"].between(box["min"][1], box["max"][1])
            & skel.nodes["z"].between(box["min"][2], box["max"][2])
        )
        if inside.any():
            ids.append(int(skel.name) + name_to_id_offset)
    return np.asarray(ids, dtype=int)


def show_segments(viewer, layer_name: str, segment_ids: Sequence[int]) -> None:
    """Set the visible segments on a segmentation layer."""
    with viewer.txn() as s:
        s.layers[layer_name].segments = list(segment_ids)
