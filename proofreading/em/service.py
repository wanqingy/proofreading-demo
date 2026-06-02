"""Headless per-cell review service -- the non-UI core of the proofreading workflow.

This is the engine half of :class:`~proofreading.em.annotator.ProofreadSession` with the
neuroglancer viewer, FlyThrough, and ipywidgets stripped out, so it can run inside an HTTP
backend (see :mod:`proofreading.em.api`). It composes the existing primitives:

- :class:`~proofreading.em.client.EMClient`       -- CAVE / CloudVolume.
- :class:`~proofreading.em.skeleton_tree.SkeletonTree` -- branch paths, L2 ids, pruning.
- :class:`~proofreading.em.coverage.Coverage` + :class:`~proofreading.em.wal.WAL` -- state.
- :class:`~proofreading.em.tube.CellTube`         -- the shared per-cell sparse tube.
- :mod:`proofreading.em.path`                     -- camera resample + orientation frame.

IMPORTANT: keep this module **neuroglancer-free** (no import of ``viewer`` / ``annotator`` /
``preview`` / ``render``) so the backend never pulls neuroglancer or ipywidgets.

M1 scope: open/resume a cell, list its branches, and build + describe a branch's camera path
over the served tube. Annotations / coverage writes / live layers are later milestones.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np

from . import path as P
from .coverage import Coverage
from .skeleton_tree import SkeletonTree
from .tube import CellTube
from .wal import WAL

# per-vertex compartment codes from the skeleton service
_COMPARTMENT = {1: "soma", 2: "axon", 3: "dendrite"}


class CellReviewService:
    """Headless review state for one cell, keyed durably by its seed supervoxel."""

    def __init__(
        self,
        emclient,
        root_id: int,
        wal_dir: str,
        *,
        step_nm: float = 500.0,
        tube_mip: int = 1,
        tube_radius_nm: float = 1000.0,
        orient_to_path: bool = False,
    ):
        self.client = emclient
        self.root_id = int(root_id)
        self.datastack = emclient.datastack
        self.step_nm = float(step_nm)
        self.tube_mip = int(tube_mip)
        self.tube_radius_nm = float(tube_radius_nm)
        self.orient_to_path = bool(orient_to_path)

        # skeleton + tree + durable identity (CAVE calls; single-threaded here)
        self.sk = emclient.get_skeleton(self.root_id)
        self.tree = SkeletonTree.from_skeleton_dict(self.sk)
        self.seed = emclient.seed_supervoxel(self.sk)
        self.mat_version = int(emclient.mat_version)

        # pre-warm the tube-mip CloudVolumes so threadpool handlers only READ the caches
        self.res = np.asarray(emclient.image_cloudvolume(self.tube_mip).resolution).tolist()
        emclient.agg_seg_cv(self.tube_mip)

        # durable state (resumes prior coverage if a log already exists)
        self.wal = WAL.for_cell(wal_dir, self.datastack, self.seed)
        self.coverage = Coverage.from_wal_state(WAL.load(self.wal.path))

        self.tube_cache_dir = os.path.join(
            os.path.abspath(wal_dir), "tube_cache", self.datastack, str(self.root_id)
        )
        self._tube: CellTube | None = None
        self._tube_lock = threading.Lock()
        self._branch_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)

        # proximal->distal review order (M4.3), computed once (the tree is fixed per session)
        self._dtr: np.ndarray | None = None  # geodesic distance (nm) root->each vertex
        self._order: list[int] | None = None  # branch ids sorted soma-outward

    # ------------------------------------------------------------------ #
    # tube
    # ------------------------------------------------------------------ #
    def ensure_tube(self) -> CellTube:
        """Open (once) the shared per-cell tube; constructing it writes the ``info`` files."""
        if self._tube is None:
            with self._tube_lock:
                if self._tube is None:
                    self._tube = CellTube(
                        self.client, self.root_id, self.tube_mip,
                        self.tube_radius_nm, self.tube_cache_dir,
                    )
        return self._tube

    def _branch_done(self, path_id: int) -> bool:
        return os.path.exists(
            os.path.join(self.tube_cache_dir, "_branches", f"path_{int(path_id)}.done")
        )

    # ------------------------------------------------------------------ #
    # branches (M4.3: ordered proximal -> distal, the guidebook sweep)
    # ------------------------------------------------------------------ #
    def _dist_to_root(self) -> np.ndarray:
        """Geodesic distance (nm) from the soma root to each vertex, along tree edges (cached)."""
        if self._dtr is None:
            tree = self.tree
            verts = np.asarray(tree.vertices, dtype=float)
            dist = np.zeros(tree.n, dtype=float)
            dq = deque([tree.root])
            while dq:
                u = dq.popleft()
                for c in tree.children[u]:
                    dist[c] = dist[u] + float(np.linalg.norm(verts[c] - verts[u]))
                    dq.append(c)
            self._dtr = dist
        return self._dtr

    def _branch_order(self) -> list[int]:
        """Branch ids sorted proximal -> distal by the soma-distance of each branch's start node.

        Mirrors guidebook's proximal-first review sweep (it groups by cover-path region + min
        distance_to_root; ordering by each branch's start distance is the same intent, simpler).
        Ties (sibling branches sharing a start node) fall back to branch id for determinism.
        """
        if self._order is None:
            dist = self._dist_to_root()
            bps = self.tree.branch_paths
            self._order = sorted(
                range(len(bps)), key=lambda i: (dist[int(bps[i].vertices[0])], i)
            )
        return self._order

    # ------------------------------------------------------------------ #
    # re-rooting (M4.4: somaless cells -> let the user choose the proximal anchor)
    # ------------------------------------------------------------------ #
    def _clear_branch_markers(self) -> None:
        """Drop the per-branch tube ``.done`` markers (branch ids change on re-root).

        The spatial chunks stay on disk and are reused, so re-filling under the new decomposition
        only fetches genuinely-missing chunks -- fast.
        """
        mdir = os.path.join(self.tube_cache_dir, "_branches")
        if os.path.isdir(mdir):
            for f in os.listdir(mdir):
                if f.endswith(".done"):
                    try:
                        os.remove(os.path.join(mdir, f))
                    except OSError:
                        pass

    def _reroot_at(self, xyz_nm, *, clear_markers: bool) -> int:
        """Rebuild the tree rooted at the skeleton vertex nearest ``xyz_nm`` (re-derives branch
        paths, ordering, and the merge-prune distal direction). Coverage (L2-keyed) is untouched."""
        v = int(self.tree.nearest_vertex(xyz_nm))
        t = self.tree
        self.tree = SkeletonTree(
            vertices=t.vertices, edges=t.edges, root=v,
            lvl2_ids=t.lvl2_ids, mesh_to_skel_map=t.mesh_to_skel_map,
            compartment=t.compartment, radius=t.radius, meta=t.meta,
        )
        self._dtr = None  # invalidate the proximal->distal caches
        self._order = None
        if clear_markers:
            self._clear_branch_markers()
        return v

    def set_root(self, xyz_nm) -> dict:
        """Re-root the review at the vertex nearest the clicked point; return the new structure.

        Branch/end-point markers are root-invariant (undirected), so the frontend only repaints the
        ordered checklist + summary (markers stay put). ``skeleton_features`` is returned so click-
        to-jump path ids stay current (M4.5).
        """
        v = self._reroot_at(xyz_nm, clear_markers=True)
        return {
            "root_vertex": int(v),
            "root_xyz_nm": [float(c) for c in self.tree.vertices[v]],
            "branches": [self.branch_metadata(i) for i in self._branch_order()],
            "summary": self.coverage.summary(self.tree),
            "skeleton_features": self.skeleton_features(),
        }

    def branch_metadata(self, path_id: int) -> dict:
        bp = self.tree.branch_paths[int(path_id)]
        verts = self.tree.vertices[bp.vertices]
        length_nm = float(np.sum(np.linalg.norm(np.diff(verts, axis=0), axis=1))) if len(verts) > 1 else 0.0
        comp = "unknown"
        if self.tree.compartment is not None:
            codes = [int(c) for c in self.tree.compartment[bp.vertices]]
            if codes:
                comp = _COMPARTMENT.get(Counter(codes).most_common(1)[0][0], "unknown")
        return {
            "path_id": int(bp.id),
            "state": self.coverage.path_state(self.tree, bp),
            "n_nodes": int(len(bp.vertices)),
            "length_nm": round(length_nm, 1),
            "compartment": comp,
            "parent": None if bp.parent is None else int(bp.parent),
            "children": [int(c) for c in bp.children],
            "built": self._branch_done(bp.id),
            "dist_to_root_nm": round(float(self._dist_to_root()[int(bp.vertices[0])]), 1),
        }

    def branches(self) -> dict:
        # ordered proximal -> distal so the checklist + auto-advance sweep the cell soma-outward
        return {
            "branches": [self.branch_metadata(i) for i in self._branch_order()],
            "summary": self.coverage.summary(self.tree),
        }

    # ------------------------------------------------------------------ #
    # camera path (+ build the branch tube)
    # ------------------------------------------------------------------ #
    def camera_path(self, path_id: int, orient: bool | None = None) -> dict:
        """Resample the branch into a camera path and ensure its tube chunks are cached.

        Returns the camera payload with the tube volume names as *relative* paths
        (``em_rel`` / ``tgt_rel``); the HTTP layer turns those into absolute
        ``precomputed://`` source URLs against the request origin.
        """
        pid = int(path_id)
        bp = self.tree.branch_paths[pid]
        verts_nm = np.asarray(self.tree.vertices[bp.vertices], dtype=float)
        rs = P.resample_path(verts_nm, self.step_nm)  # (M,3) nm camera path

        orientations = None
        if (self.orient_to_path if orient is None else orient) and len(rs) >= 2:
            T, N, B = P.rotation_minimizing_frame(rs)
            orientations = P.frame_to_quaternion(T, N, B).tolist()  # (M,4) xyzw

        tube = self.ensure_tube()
        with self._branch_locks[pid]:
            cached = self._branch_done(pid)
            t0 = time.time()
            tube.fill_branch(pid, verts_nm, verbose=False)  # raw verts; resamples internally
            seconds = round(time.time() - t0, 1)

        rel = f"tube/{self.datastack}/{self.root_id}"
        return {
            "path_id": pid,
            "root_id": str(self.root_id),  # string: exceeds JS 2^53 safe-int range
            "resolution_nm": [int(x) for x in self.res],
            "points_nm": rs.tolist(),
            "orientations": orientations,
            "step_nm": self.step_nm,
            "em_rel": f"{rel}/{tube.em_name}",
            "tgt_rel": f"{rel}/{tube.tgt_name}",
            "build": {"cached": bool(cached), "seconds": seconds},
        }

    # ------------------------------------------------------------------ #
    # annotations (M2.1: record + list; merge-prune is M2.4)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ann(a) -> dict:
        return {
            "uuid": a.uuid,
            "tag": a.tag,
            "xyz": [float(c) for c in a.xyz],  # nm
            "supervoxel": None if a.supervoxel is None else str(int(a.supervoxel)),
        }

    def add_annotation(self, tag: str, xyz_nm) -> dict:
        """Record a tagged click (durably to the WAL). Returns the new annotation + summary.

        Merge-error pruning is deferred to M2.4; for now every tag just records a point.
        """
        ann = self.wal.add_annotation(tag, xyz_nm, self.root_id, self.mat_version, self.seed)
        return {"annotation": self._ann(ann), "summary": self.coverage.summary(self.tree)}

    def list_annotations(self) -> list:
        """Live annotations from the durable log (for redraw on resume)."""
        st = WAL.load(self.wal.path)
        return [self._ann(a) for a in st.annotations.values()]

    def delete_annotation(self, uuid: str) -> dict:
        """Soft-delete an annotation (tombstone in the WAL) and rebuild coverage.

        The coverage rebuild is a no-op until merge-prune (M2.4) writes ``omit`` events, but we
        do it now so that deleting a merge-error mark will later revert the distal omissions it
        caused. Idempotent: tombstoning an unknown/already-dead uuid is harmless.
        """
        existed = uuid in WAL.load(self.wal.path).annotations
        self.wal.tombstone(uuid)
        self.coverage = Coverage.from_wal_state(WAL.load(self.wal.path))
        return {
            "deleted": existed,
            "uuid": uuid,
            "summary": self.coverage.summary(self.tree),
        }

    # ------------------------------------------------------------------ #
    # coverage (M2.3: mark a branch reviewed + advance)
    # ------------------------------------------------------------------ #
    def mark_done(self, path_id: int) -> dict:
        """Mark a branch reviewed: record its L2 ids visited (durably) and pick the next.

        Mirrors :meth:`ProofreadSession._on_mark_done`: uses the FULL path (incl. the shared
        proximal node) deliberately -- ``path_state`` classifies on ``bp.vertices[1:]``, so
        covering the whole path is safe and keeps the parent's shared node counted. Returns
        the next to-review branch id (or ``None`` = cell complete) plus the refreshed summary
        and full branch checklist (so the frontend can repaint the dropdown in one round-trip).
        """
        pid = int(path_id)
        bp = self.tree.branch_paths[pid]
        l2 = self.tree.l2_ids_for_vertices(bp.vertices)
        self.wal.mark_visited(l2)
        self.coverage.mark_visited(l2)
        # next = the first still-to-review branch in proximal -> distal order (M4.3)
        todo = set(self.coverage.to_review(self.tree))
        next_pid = next((i for i in self._branch_order() if i in todo), None)
        return {
            "path_id": pid,
            "next_path_id": next_pid,
            "summary": self.coverage.summary(self.tree),
            "branches": [self.branch_metadata(i) for i in self._branch_order()],
        }

    # ------------------------------------------------------------------ #
    # live sources (M3: pause -> full-res EM + real graphene segmentation)
    # ------------------------------------------------------------------ #
    def live_sources(self) -> dict:
        """Full-res live layer sources for the pause->live swap.

        The browser shows these only when the camera is IDLE (paused): the mip0 EM is too
        heavy to stream during motion and graphene seg only paints when idle, so the sparse
        tube stays the in-motion view. The segmentation source carries the ``middleauth+``
        prefix so the browser authenticates with the CAVE token (returned here for a JS
        credentials provider; the backend binds 127.0.0.1 only, so the token stays on localhost).
        """
        info = self.client.client.info
        seg = info.segmentation_source()
        if seg.startswith("graphene://") and "middleauth+" not in seg:
            seg = seg.replace("graphene://", "graphene://middleauth+", 1)
        res = [float(x) for x in np.asarray(info.viewer_resolution(), dtype=float)]
        token = None
        try:
            token = self.client.client.auth.token
        except Exception:
            pass
        return {
            "root_id": str(self.root_id),  # string: exceeds JS 2^53 safe-int range
            "image_source": info.image_source(),
            "segmentation_source": seg,
            "skeleton_source": self._skeleton_source(),
            "viewer_resolution_nm": res,
            "token": token,
        }

    def skeleton_features(self) -> dict:
        """Branch points and end points for the 3D guidance overlay (M4.2), the guidebook markers.

        Branch points (undirected degree >= 3) are where **merge/split** errors hide; end points /
        tips (degree == 1) are where **extends** (premature terminations) hide. Uses UNDIRECTED
        degree (like guidebook's ``*_undirected``) so the markers are anatomical and **root-invariant**
        -- re-rooting (M4.4) doesn't move them, only the ordering/decomposition changes. Each point is
        tagged with the id of the branch path that ENDS at it (for click-to-jump, M4.5; the chosen
        root ends no path -> ``null``).
        """
        tree = self.tree
        end_to_path = {int(bp.vertices[-1]): int(bp.id) for bp in tree.branch_paths}
        deg = np.bincount(tree.edges.reshape(-1), minlength=tree.n)

        def _pt(v: int) -> dict:
            xyz = tree.vertices[int(v)]
            return {"xyz_nm": [float(c) for c in xyz], "path_id": end_to_path.get(int(v))}

        return {
            "branch_points": [_pt(int(v)) for v in np.where(deg >= 3)[0]],
            "end_points": [_pt(int(v)) for v in np.where(deg == 1)[0]],
        }

    def _skeleton_source(self) -> str | None:
        """skeletoncache precomputed skeleton source (M4): a 3D backdrop of the cell skeleton.

        Same origin as the graphene seg (``minnie.microns-daf.com``), so the browser's injected
        middleauth token (M3.2) authorizes it. Skeleton-only (no volumetric) -> renders in the 3D
        panel, nothing in the 2D cross-section, so the EM review view stays clean.
        """
        try:
            sk = self.client.client.skeleton
            server = sk.server_address.rstrip("/")
            api_v = getattr(sk, "api_version", 1)
            base = f"{server}/skeletoncache/api/v{api_v}/{self.datastack}/precomputed/skeleton/"
            return f"precomputed://middleauth+{base}"
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # header / snapshot
    # ------------------------------------------------------------------ #
    def header(self) -> dict:
        return {
            "root_id": str(self.root_id),       # strings: exceed JS 2^53 safe-int range
            "seed_supervoxel": str(int(self.seed)),
            "mat_version": self.mat_version,
            "datastack": self.datastack,
            "resolution_nm": [int(x) for x in self.res],
            "n_branches": int(len(self.tree.branch_paths)),
            "step_nm": self.step_nm,
            "summary": self.coverage.summary(self.tree),
            "tube_rel": f"tube/{self.datastack}/{self.root_id}",
        }

    def close(self) -> None:
        try:
            self.wal.close()
        except Exception:
            pass
