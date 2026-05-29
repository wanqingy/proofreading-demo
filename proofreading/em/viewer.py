"""Neuroglancer viewer for an EM proofreading session.

Layers: EM imagery + the graphene segmentation (target root highlighted, other
segments off by default), a per-tag annotation layer for the dropped points, and
a line layer showing the current branch path. Sources and the viewer resolution
come straight from CAVE (``client.info``).

Positions are handled in nm by the rest of the engine; this module converts to the
viewer's voxel units (``nm / viewer_resolution``) at the boundary.
"""

from __future__ import annotations

import threading
import time
from typing import Tuple

import neuroglancer
import numpy as np

from .wal import TAGS

# one colored annotation layer per tag
TAG_COLORS = {
    "merge error": "#ff3333",
    "split error": "#33aaff",
    "extend": "#33ff66",
    "question": "#ffcc00",
}
PATH_LAYER = "branch_path"
IMAGE_LAYER = "img"
SEG_LAYER = "seg"


def ann_layer(tag: str) -> str:
    return "ann:" + tag.replace(" ", "_")


def _ensure_middleauth(graphene_source: str) -> str:
    """Graphene segmentation needs browser auth -- ensure the ``middleauth+`` prefix.

    ``client.info.segmentation_source()`` returns the bare ``graphene://https://...``;
    neuroglancer needs ``graphene://middleauth+https://...`` to run the OAuth flow,
    or the segmentation layer fails to load. (Only affects the browser -- the python
    client never fetches the layer.)
    """
    if graphene_source.startswith("graphene://") and "middleauth+" not in graphene_source:
        return graphene_source.replace("graphene://", "graphene://middleauth+", 1)
    return graphene_source


def register_middleauth_token(token: str) -> None:
    """Teach python neuroglancer how to auth to graphene middleauth servers.

    The browser asks the local python server for credentials with key
    ``'middleauthapp'``; python neuroglancer ships google/boss/dvid providers but
    *not* middleauth, so the request 500s with ``KeyError: 'middleauthapp'``. This
    registers a provider that hands back the CAVE token as a Bearer token.
    """
    from neuroglancer import credentials_provider as _cp
    from neuroglancer.default_credentials_manager import default_credentials_manager
    from neuroglancer.futures import run_on_new_thread

    class _MiddleAuthProvider(_cp.CredentialsProvider):
        def get_new(self):
            return run_on_new_thread(
                lambda: dict(tokenType="Bearer", accessToken=token)
            )

    default_credentials_manager.register(
        "middleauthapp", lambda _parameters: _MiddleAuthProvider()
    )


def make_em_viewer(
    emclient,
    root_id: int,
    ip: str = "localhost",
    port: int = 0,
    cross_section_render_scale: float = 1.0,
    gpu_memory_limit: int = 2_000_000_000,
    system_memory_limit: int = 4_000_000_000,
) -> Tuple[neuroglancer.Viewer, np.ndarray, neuroglancer.CoordinateSpace]:
    """Build the viewer with EM + segmentation + annotation + path layers.

    ``cross_section_render_scale`` > 1 renders the EM cross-section at a coarser
    resolution (e.g. 2 ~ "mip1": ~4x less data, loads faster) -- trade detail for
    speed. ``gpu_memory_limit`` / ``system_memory_limit`` size the tile cache so
    loaded data persists (instant reverse / revisit).

    Returns ``(viewer, viewer_resolution_nm, dimensions)``.
    """
    info = emclient.client.info
    img_src = info.image_source()
    seg_src = _ensure_middleauth(info.segmentation_source())
    res = np.asarray(info.viewer_resolution(), dtype=float)

    # let the browser authenticate to the graphene server with the CAVE token
    try:
        register_middleauth_token(emclient.client.auth.token)
    except Exception as exc:  # pragma: no cover
        import warnings

        warnings.warn(f"could not register middleauth token: {exc!r}")

    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()
    dims = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)

    img_layer = neuroglancer.ImageLayer(source=img_src)
    if cross_section_render_scale and cross_section_render_scale != 1.0:
        img_layer.cross_section_render_scale = cross_section_render_scale

    with viewer.txn() as s:
        s.dimensions = dims
        s.layers[IMAGE_LAYER] = img_layer
        s.layers[SEG_LAYER] = neuroglancer.SegmentationLayer(
            source=seg_src, segments=[int(root_id)]
        )
        for tag in TAGS:
            s.layers[ann_layer(tag)] = neuroglancer.LocalAnnotationLayer(
                dimensions=dims, annotation_color=TAG_COLORS[tag]
            )
        s.layers[PATH_LAYER] = neuroglancer.LocalAnnotationLayer(
            dimensions=dims, annotation_color="#888888"
        )
        s.gpu_memory_limit = int(gpu_memory_limit)
        s.system_memory_limit = int(system_memory_limit)
        s.layout = "4panel"
    return viewer, res, dims


def add_point(viewer: neuroglancer.Viewer, tag: str, point_voxel, ann_id: str) -> None:
    """Append a point annotation (in viewer voxel coords) to a tag's layer."""
    with viewer.txn() as s:
        s.layers[ann_layer(tag)].annotations.append(
            neuroglancer.PointAnnotation(id=ann_id, point=[float(c) for c in point_voxel])
        )


def remove_point(viewer: neuroglancer.Viewer, tag: str, ann_id: str) -> None:
    """Remove a point annotation by id (for tombstones)."""
    with viewer.txn() as s:
        anns = s.layers[ann_layer(tag)].annotations
        s.layers[ann_layer(tag)].annotations = [a for a in anns if a.id != ann_id]


def show_branch_path(viewer: neuroglancer.Viewer, points_voxel: np.ndarray) -> None:
    """Draw the current branch path as a polyline (viewer voxel coords)."""
    pts = np.asarray(points_voxel, dtype=float)
    lines = [
        neuroglancer.LineAnnotation(
            id=f"seg{i}", point_a=list(pts[i]), point_b=list(pts[i + 1])
        )
        for i in range(len(pts) - 1)
    ]
    with viewer.txn() as s:
        s.layers[PATH_LAYER].annotations = lines


def set_segments(viewer: neuroglancer.Viewer, segment_ids) -> None:
    with viewer.txn() as s:
        s.layers[SEG_LAYER].segments = [int(x) for x in segment_ids]


def set_prefetch(viewer: neuroglancer.Viewer, nav_list) -> None:
    """Tell neuroglancer to prefetch upcoming frames in the background.

    ``nav_list`` is ``[(voxel_coordinates, orientation_or_None, priority), ...]``.
    Each becomes a ``PrefetchState`` cloned from the current view with only the
    camera moved -- so the browser warms the EM/segmentation tiles for those nodes
    (at low priority, behind the current view) before the camera arrives.
    """
    import copy

    base = viewer.state
    states = []
    for vox, ori, priority in nav_list:
        st = copy.deepcopy(base)
        st.voxel_coordinates = list(vox)
        if ori is not None:
            st.crossSectionOrientation = list(ori)
            st.projectionOrientation = list(ori)
        states.append(neuroglancer.PrefetchState(priority=int(priority), state=st))
    with viewer.config_state.txn() as cs:
        cs.prefetch = states


def wait_until_loaded(
    viewer: neuroglancer.Viewer, timeout: float = 15.0, poll: float = 0.15
) -> bool:
    """Block until **all visible chunks are GPU-resident** (every layer rendered).

    The bare screenshot reply fires once chunks are *downloaded*, but graphene
    segmentation uploads to the GPU a frame later -- so gating on the reply lets the
    camera move before the seg mask has rendered. Instead we poll the screenshot
    statistics and wait until ``visible_chunks_gpu_memory == visible_chunks_total``
    with nothing downloading -- i.e. the image *and* segmentation are actually drawn.

    Returns ``True`` once loaded, ``False`` on timeout (e.g. no browser) -- never hangs,
    so it is safe as a :class:`FlyThrough` ``settle`` callback.
    """
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        stats: dict = {}
        ev = threading.Event()

        def _on_stats(s, _stats=stats):
            try:
                t = s.total
                _stats["total"] = int(t.visible_chunks_total)
                _stats["gpu"] = int(t.visible_chunks_gpu_memory)
                _stats["downloading"] = int(t.visible_chunks_downloading)
            except Exception:
                pass

        try:
            viewer.async_screenshot(lambda _s: ev.set(), statistics_callback=_on_stats)
        except Exception:
            return False
        if not ev.wait(remaining):
            return False
        if (
            stats.get("total", 0) > 0
            and stats.get("gpu", -1) >= stats["total"]
            and stats.get("downloading", 1) == 0
        ):
            return True
        time.sleep(poll)  # not fully GPU-resident yet -- re-poll
