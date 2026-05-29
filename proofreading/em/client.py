"""CAVE / CloudVolume wrapper for the EM workflow.

Thin layer over ``caveclient`` + ``cloudvolume`` providing exactly what Phase A
needs. Verified against the live API (``minnie65_public`` @ mat v1718):

- ``get_skeleton`` returns the L2 skeleton dict (vertices nm / edges / lvl2_ids / ...).
- ``points_to_supervoxels`` resolves click ``xyz`` (nm) to supervoxels in batch via
  CloudVolume ``scattered_points`` on the de-agglomerated segmentation.
- ``supervoxel_to_root`` / ``current_root`` re-resolve the (volatile) root from a
  (stable) supervoxel -- the mechanism behind seed-supervoxel identity (ADR 0001).

Requires the ``em`` extra and a CAVE token at ``~/.cloudvolume/secrets/cave-secret.json``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_DATASTACK = "minnie65_phase3_v1"  # live, proofreadable; minnie65_public = sandbox


class EMClient:
    """Convenience wrapper around a CAVEclient pinned to a materialization version."""

    def __init__(self, datastack: str = DEFAULT_DATASTACK, version: Optional[int] = None):
        from caveclient import CAVEclient  # lazy: needs the `em` extra

        self.datastack = datastack
        self.client = (
            CAVEclient(datastack, version=version) if version else CAVEclient(datastack)
        )
        self._seg = None
        self._res = None

    @property
    def mat_version(self) -> int:
        """The CAVEclient materialization version in effect."""
        return self.client.materialize.version

    @property
    def seg(self):
        """De-agglomerated segmentation CloudVolume (returns supervoxels, not roots)."""
        if self._seg is None:
            self._seg = self.client.info.segmentation_cloudvolume(
                agglomerate=False, progress=False
            )
            self._res = np.array(self._seg.mip_resolution(0))
        return self._seg

    @property
    def seg_resolution(self) -> np.ndarray:
        """mip0 voxel size in nm (e.g. [8, 8, 40] for minnie65)."""
        _ = self.seg
        return self._res

    # ----- skeletons ----------------------------------------------------- #
    def get_skeleton(self, root_id: int, skeleton_version: int = 4) -> dict:
        return self.client.skeleton.get_skeleton(
            int(root_id), skeleton_version=skeleton_version, output_format="dict"
        )

    # ----- point -> supervoxel -> root ----------------------------------- #
    def points_to_supervoxels(self, points_nm) -> np.ndarray:
        """Resolve 3D points (nm) to supervoxel ids in batch, order preserved."""
        res = self.seg_resolution
        pts = np.atleast_2d(np.asarray(points_nm, dtype=float))
        vox = np.round(pts / res).astype(np.int64)
        d = self.seg.scattered_points(vox, coord_resolution=res)
        lut = {tuple(int(c) for c in k): int(v) for k, v in d.items()}
        return np.array([lut[tuple(int(c) for c in v)] for v in vox], dtype=np.uint64)

    def supervoxel_to_root(self, supervoxel: int, timestamp=None) -> int:
        return int(self.client.chunkedgraph.get_root_id(int(supervoxel), timestamp=timestamp))

    def supervoxels_to_roots(self, supervoxels, timestamp=None) -> np.ndarray:
        return np.asarray(
            self.client.chunkedgraph.get_roots(
                np.asarray(supervoxels, dtype=np.uint64), timestamp=timestamp
            )
        )

    # ----- seed-supervoxel identity (ADR 0001) --------------------------- #
    def seed_supervoxel(self, skeleton_dict: dict) -> int:
        """The supervoxel at the soma (root) vertex -- the cell's durable identity."""
        root_v = int(skeleton_dict["root"])
        soma_nm = skeleton_dict["vertices"][root_v]
        return int(self.points_to_supervoxels([soma_nm])[0])

    def current_root(self, seed_supervoxel: int) -> int:
        """Resolve the cell's current root id from its durable seed supervoxel."""
        return self.supervoxel_to_root(seed_supervoxel)
