"""Load SWC skeletons and turn them into a camera path.

The orientation math (tangent vectors -> quaternions -> axis flips) lives in the
lab's ``ac_ngl`` module; this file just orchestrates those calls so the notebook
cells become a single :func:`compute_path` call.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import navis
except ImportError:  # pragma: no cover
    navis = None


def load_skeleton(path: str, swap_xz: bool = True):
    """Read a single SWC file (or directory) with navis.

    ``swap_xz=True`` reproduces the notebook's x<->z swap that aligns SWC node
    coordinates with the Neuroglancer image axes.
    """
    if navis is None:
        raise ImportError("navis is required to load skeletons")
    skel = navis.read_swc(path)
    if swap_xz:
        skel = swap_xz_inplace(skel)
    return skel


def swap_xz_inplace(skel):
    """Swap the x and z node columns in place and return the skeleton."""
    tmp_x = skel.nodes.x.copy()
    skel.nodes.x = skel.nodes.z
    skel.nodes.z = tmp_x
    return skel


# Backwards-friendly alias matching the package __all__ export name.
swap_xz = swap_xz_inplace


def nodes_to_positions(skel) -> np.ndarray:
    """Return node coordinates as an ``(N, 3)`` array in ``[x, y, z]`` order.

    This matches the ordering the notebooks fed into ``voxel_coordinates``.
    """
    nodes = skel.nodes
    return np.column_stack([nodes["x"].values, nodes["y"].values, nodes["z"].values])


def compute_orientations(skel, k: int = 15) -> np.ndarray:
    """Per-node camera quaternions tangent to the skeleton.

    Mirrors the notebook sequence:
    tangent vectors -> dotprops -> flip vectors -> quaternions -> flip axis.
    Requires ``ac_ngl``.
    """
    try:
        from ac_ngl import (  # type: ignore
            get_tangent_vector,
            surface_normal_to_quaternion,
            flip_vectors_if_necessary,
            flip_quaternion_axis,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_orientations needs the lab's ac_ngl module on sys.path"
        ) from exc
    if navis is None:
        raise ImportError("navis is required to compute orientations")

    # get_tangent_vector primes navis' dotprops; the dotprops vectors are what
    # we actually convert to quaternions (matches the notebook order).
    get_tangent_vector(skel)
    dp = navis.make_dotprops(skel, k=k)
    vects = flip_vectors_if_necessary(dp.vect)

    quaternions = []
    for v in vects:
        quaternion, _ = surface_normal_to_quaternion(v)
        quaternions.append(quaternion)

    return flip_quaternion_axis(np.asarray(quaternions))


def compute_path(skel, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning ``(positions, orientations)`` for a skeleton.

    ``positions`` is ``(N, 3)`` in ``[x, y, z]`` order; ``orientations`` is the
    matching ``(N, 4)`` array of camera quaternions.
    """
    positions = nodes_to_positions(skel)
    orientations = compute_orientations(skel, k=k)
    return positions, orientations
