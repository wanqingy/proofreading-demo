"""Neuroglancer server + layer setup.

Wraps the boilerplate from the notebooks' "Generate neuroglancer link" cell.
"""

from __future__ import annotations

from typing import Optional, Sequence

import neuroglancer
import numpy as np

# Host that actually serves the precomputed data over HTTP. The notebooks point
# the layer sources at this machine rather than ``localhost`` so the volumes are
# reachable from a browser on the corp network.
DEFAULT_DATA_HOST = "http://bigkahuna.corp.alleninstitute.org/"

# Rotation that maps skeleton (x, y, z) into the image's (z, y, x) ordering.
# This is the "axis flip" that several notebook commits were fighting with.
SKELETON_XZ_SWAP = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)


def make_viewer(
    ip: str = "localhost",
    port: int = 9998,
) -> neuroglancer.Viewer:
    """Bind the Neuroglancer server and return a fresh viewer."""
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    return neuroglancer.Viewer()


def _precomputed_source(path: str, data_host: str) -> str:
    return "precomputed://" + data_host.rstrip("/") + "/" + path.lstrip("/")


def load_image_layer(
    viewer: neuroglancer.Viewer,
    name: str,
    precomputed_path: str,
    data_host: str = DEFAULT_DATA_HOST,
    shader_range: Optional[Sequence[float]] = None,
) -> None:
    """Add a precomputed image layer to ``viewer``."""
    source = _precomputed_source(precomputed_path, data_host)
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.ImageLayer(source=[source])
        if shader_range is not None:
            lo, hi = shader_range
            s.layers[name].layer.shaderControls = {"normalized": {"range": [lo, hi]}}


def load_skeleton_layer(
    viewer: neuroglancer.Viewer,
    name: str,
    precomputed_path: str,
    data_host: str = DEFAULT_DATA_HOST,
    transform_matrix: Optional[np.ndarray] = SKELETON_XZ_SWAP,
    dimensions: Optional[neuroglancer.CoordinateSpace] = None,
) -> None:
    """Add a precomputed segmentation/skeleton layer to ``viewer``.

    ``transform_matrix`` defaults to the x<->z swap that aligns the skeletons
    with the image volume; pass ``None`` to load them untransformed.
    """
    source = _precomputed_source(precomputed_path, data_host)
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.SegmentationLayer(source=[source])
        if transform_matrix is not None:
            dims = dimensions
            transform = neuroglancer.CoordinateSpaceTransform(
                matrix=transform_matrix,
                input_dimensions=dims,
                output_dimensions=dims,
            )
            s.layers[name].layer.source[0].transform = transform


def shareable_url(viewer: neuroglancer.Viewer) -> str:
    """Return a shareable Neuroglancer URL for the current viewer state.

    Uses the lab's ``ac_ngl`` helper when available, otherwise falls back to
    Neuroglancer's own URL encoder.
    """
    state = viewer.state
    try:
        from ac_ngl import make_neuroglancer_url_vneurodata  # type: ignore

        return make_neuroglancer_url_vneurodata(state.to_json())
    except ImportError:
        return neuroglancer.to_url(state)
