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

# Must run before ANY CAVEclient / CloudVolume / GCS client is constructed (they bind their
# connection pool at creation), hence at import time here rather than inside EMClient.__init__.
from ._http_pool import tune_connection_pool

tune_connection_pool()

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
        self._img_cvs: dict = {}  # mip -> image CloudVolume
        # (mip, timestamp) -> agglomerated segmentation CloudVolume. Keyed on the TIMESTAMP too:
        # the same mip agglomerated at two different times is two different volumes, and caching
        # them under one key would hand back whichever was built first.
        self._agg_cvs: dict = {}

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

    def agg_seg_cv(self, mip: int = 0, timestamp=None):
        """Agglomerated segmentation CloudVolume at ``mip`` (root ids, not supervoxels; cached).

        Used to build the per-branch target mask for the local fly-through preview
        (``mask = cutout == root_id``). Distinct from :attr:`seg` (agglomerate=False).

        ``timestamp`` pins WHEN the agglomeration is evaluated. Without it the chunkedgraph
        agglomerates supervoxels to their **current** roots, which silently breaks the mask for
        any root that is no longer current: the skeleton service happily serves a historical root,
        so the fly-through looks perfect while ``cutout == root_id`` matches zero voxels and the
        overlay is invisible. Observed on 864691135572519149 -- created 2024-10-23, since split
        into 165 roots -- where all 174 mask chunks came out empty. Pass the root's own creation
        time (see :meth:`root_timestamp`) and the comparison lines up again. Measured to cost
        nothing: 5285 ms vs 5222 ms per 64^3 cutout.
        """
        mip = int(mip)
        key = (mip, timestamp)
        cv = self._agg_cvs.get(key)
        if cv is None:
            kw = {"agglomerate": True, "mip": mip, "progress": False}
            if timestamp is not None:
                kw["timestamp"] = timestamp
            cv = self.client.info.segmentation_cloudvolume(**kw)
            self._agg_cvs[key] = cv
        return cv

    def root_timestamp(self, root_id: int):
        """When ``root_id`` came into existence, or None if that can't be determined.

        This is the right moment at which to agglomerate for that root, whether or not it is still
        current -- a root stops being current precisely BECAUSE an edit produced a new one, so
        "still current" means "nothing has changed since it was created". One rule covers both.
        """
        try:
            ts = self.client.chunkedgraph.get_root_timestamps([int(root_id)])
        except Exception:
            return None  # never block a session on this; the caller falls back to live agglomeration
        return ts[0] if len(ts) else None

    def is_current_root(self, root_id: int) -> Optional[bool]:
        """Whether ``root_id`` is still a leaf of the chunkedgraph (None if it can't be checked)."""
        try:
            return bool(self.client.chunkedgraph.is_latest_roots([int(root_id)])[0])
        except Exception:
            return None

    def image_cloudvolume(self, mip: int = 0):
        """EM image CloudVolume at ``mip`` (cached per mip).

        Unlike the segmentation, the imagery was previously only used as a source
        *string* in the browser; this is the first direct (python-side) cutout source.
        """
        mip = int(mip)
        cv = self._img_cvs.get(mip)
        if cv is None:
            cv = self.client.info.image_cloudvolume(mip=mip, progress=False)
            self._img_cvs[mip] = cv
        return cv

    def mip_near(self, cv, target_nm: float) -> int:
        """Pick the CloudVolume mip whose in-plane (x) resolution is closest to ``target_nm``."""
        mips = list(cv.available_mips)
        xy = [float(np.asarray(cv.mip_resolution(m))[0]) for m in mips]
        return int(mips[int(np.argmin([abs(x - float(target_nm)) for x in xy]))])

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
