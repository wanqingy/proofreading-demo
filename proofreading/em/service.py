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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import path as P
from .coverage import Coverage
from .skeleton_tree import SkeletonTree
from .tube import CellTube, branch_marker_state
from .wal import WAL

# per-vertex compartment codes from the skeleton service
_COMPARTMENT = {1: "soma", 2: "axon", 3: "dendrite"}


def _scope(compartment: str | None) -> tuple[str | None, str]:
    """Resolve a ``compartment`` argument into (branch filter, coverage dimension key).

    Three real cases, and the middle one is the reason this exists as a function:

    ======================  ===============  ==================  ==========================
    ``compartment``         branch filter    coverage dimension  caller
    ======================  ===============  ==================  ==========================
    ``"axon"``              axon only        ``myelin_state``    myelin tool (default)
    ``"all"``               none             ``myelin_state``    myelin tool, whole skeleton
    ``None``                none             ``state``           review tool (main.ts)
    ======================  ===============  ==================  ==========================

    This used to be inlined as ``"myelin_state" if compartment == "axon" else "state"``, which
    inferred *which tool is asking* from *which compartment it wants*. That held only while the
    myelin tool was axon-only by construction: the moment it can sweep the whole skeleton, an
    unfiltered myelin request would silently fall through to the ERROR-REVIEW coverage and warm
    or prebuild the wrong branches, with nothing raised. ``"all"`` keeps the two questions apart.
    """
    if compartment == "axon":
        return "axon", "myelin_state"
    if compartment == "all":
        return None, "myelin_state"
    return None, "state"


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
        prebuild_ahead: int = 2,
        tgt_mip: int | None = None,  # None -> CellTube.DEFAULT_TGT_MIP (coarse; see tube.py)
    ):
        self.client = emclient
        self.root_id = int(root_id)
        self.datastack = emclient.datastack
        self.step_nm = float(step_nm)
        self.tube_mip = int(tube_mip)
        self.tgt_mip = None if tgt_mip is None else int(tgt_mip)
        self.tube_radius_nm = float(tube_radius_nm)
        self.orient_to_path = bool(orient_to_path)

        # skeleton + tree + durable identity (CAVE calls; single-threaded here)
        self.sk = emclient.get_skeleton(self.root_id)
        self.tree = SkeletonTree.from_skeleton_dict(self.sk)
        self.seed = emclient.seed_supervoxel(self.sk)
        self.mat_version = int(emclient.mat_version)

        # WHEN to agglomerate the segmentation for the mask. The skeleton service serves historical
        # roots quite happily, so without this the fly-through of an edited-since cell looks perfect
        # while its mask matches zero voxels and renders invisible (observed on 864691135572519149).
        # Resolved once per session; None just means "live agglomeration", i.e. the old behaviour.
        self.agg_timestamp = emclient.root_timestamp(self.root_id)
        self.root_is_current = emclient.is_current_root(self.root_id)

        # pre-warm the tube-mip CloudVolumes so threadpool handlers only READ the caches. Warm the
        # SAME (mip, timestamp) the mask build will ask for, or the warm-up populates a different
        # cache entry than the one that gets used.
        self.res = np.asarray(emclient.image_cloudvolume(self.tube_mip).resolution).tolist()
        emclient.agg_seg_cv(self.tube_mip, timestamp=self.agg_timestamp)

        # durable state (resumes prior coverage if a log already exists)
        self.wal = WAL.for_cell(wal_dir, self.datastack, self.seed, root_id=self.root_id)
        _state = WAL.load(self.wal.path)
        self.coverage = Coverage.from_wal_state(_state)
        # myelin events live in a SEPARATE file from the review tags above (different tool,
        # different vocabulary -- see wal.py's `for_cell` docstring), so myelin review progress
        # is resumed from its own log; no omission concept for this coverage dimension.
        self.myelin_wal = WAL.for_cell(
            wal_dir, self.datastack, self.seed, kind="myelin", root_id=self.root_id
        )
        _myelin_state = WAL.load(self.myelin_wal.path)
        self.myelin_coverage = Coverage(visited_l2=set(_myelin_state.myelin_visited_l2))
        self.myelin_done = _myelin_state.cell_done
        self.myelin_done_ts = _myelin_state.cell_done_ts
        # "axon" | "all" -- which part of the skeleton this cell is annotated over. Defaults to
        # "axon" for any log predating the scope event, i.e. how it was actually reviewed.
        self.myelin_scope = _myelin_state.scope
        self._resume_root_xyz = _state.root_xyz  # re-applied at the end of __init__ (below)

        self.tube_cache_dir = os.path.join(
            os.path.abspath(wal_dir), "tube_cache", self.datastack, str(self.root_id)
        )
        self._tube: CellTube | None = None
        self._tube_lock = threading.Lock()
        self._branch_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)

        # proximal->distal review order (M4.3), computed once (the tree is fixed per session)
        self._dtr: np.ndarray | None = None  # geodesic distance (nm) root->each vertex
        self._order: list[int] | None = None  # branch ids sorted soma-outward

        # background pre-build (M4): build the next to-review branch(es) while the user reviews the
        # current one, so advancing is instant. Single worker -> one branch at a time, full
        # bandwidth during review (the current branch's build already finished + is just gliding).
        self.prebuild_ahead = int(prebuild_ahead)
        self._prebuild_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prebuild")
        self._prebuilding: set[int] = set()
        self._prebuild_lock = threading.Lock()
        # live chunk-level fill progress, path_id -> {phase, done, total, ts}. Lets the UI show a
        # real caching bar and, via `ts`, tell "slowly working a big branch" from "wedged read"
        # -- the .done marker alone can't, since it only flips at the very end of a branch.
        self._fill_progress: dict[int, dict] = {}
        self._fill_progress_lock = threading.Lock()
        # Monotonic count of chunks fetched this session. The per-fill numbers above are NOT
        # monotonic -- they reset when a branch switches em->tgt phase, and the set of active
        # fills shrinks as branches finish -- so a total that only ever increases is what
        # actually answers "is it still making progress or is it wedged?".
        self._chunks_cached = 0
        self._epoch = 0  # bumped on re-root so stale in-flight pre-builds abort

        # resume a previously chosen review root (persisted set_root). xyz is resolution-independent
        # -> nearest_vertex snaps back to the same skeleton vertex. Keep the on-disk tube builds:
        # this re-derives the SAME decomposition the .done markers were written under, so the cached
        # branches stay valid (clear_markers=False). Done last: needs the caches + _epoch set above.
        if self._resume_root_xyz is not None:
            self._reroot_at(self._resume_root_xyz, clear_markers=False)

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
                        self.tube_radius_nm, self.tube_cache_dir, tgt_mip=self.tgt_mip,
                        agg_timestamp=self.agg_timestamp,
                    )
        return self._tube

    def _branch_done(self, path_id: int) -> bool:
        """Fully built for the CURRENT config -- em AND the mask at the mask mip in effect.

        Deliberately not just "a marker exists": the mask resolution is configurable, so a branch
        built under a different mask mip is NOT done for this session (its mask volume has no
        chunks). Reads markers off disk rather than via ``ensure_tube()`` so the branch checklist
        doesn't force CloudVolume construction. See :func:`tube.branch_marker_state`.
        """
        tgt_mip = CellTube.DEFAULT_TGT_MIP if self.tgt_mip is None else self.tgt_mip
        return all(branch_marker_state(
            os.path.join(self.tube_cache_dir, "_branches"), path_id, self.tube_mip, tgt_mip,
        ))

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
        self._epoch += 1  # abort any in-flight pre-builds for the old decomposition
        if clear_markers:
            self._clear_branch_markers()
        return v

    def set_root(self, xyz_nm) -> dict:
        """Re-root the review at the vertex nearest the clicked point; return the new structure.

        Persisted to the WAL (``set_root`` event) so the choice survives a reload -- ``__init__``
        re-applies it on open (somaless cells otherwise reset to the skeleton's arbitrary default
        tip). Branch/end-point markers are root-invariant (undirected), so the frontend only repaints
        the ordered checklist + summary (markers stay put). ``skeleton_features`` is returned so
        click-to-jump path ids stay current (M4.5).
        """
        v = self._reroot_at(xyz_nm, clear_markers=True)
        # persist the SNAPPED vertex position (exact -> nearest_vertex re-snaps to it on reload)
        self.wal.set_root([float(c) for c in self.tree.vertices[v]])
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
            "myelin_state": self.myelin_coverage.path_state(self.tree, bp),
            "n_nodes": int(len(bp.vertices)),
            "length_nm": round(length_nm, 1),
            "compartment": comp,
            "parent": None if bp.parent is None else int(bp.parent),
            "children": [int(c) for c in bp.children],
            "built": self._branch_done(bp.id),
            "dist_to_root_nm": round(float(self._dist_to_root()[int(bp.vertices[0])]), 1),
        }

    @staticmethod
    def _tally_myelin_state(metas: list[dict]) -> dict:
        counts = {"to_review": 0, "covered": 0, "omitted": 0}
        for m in metas:
            counts[m["myelin_state"]] += 1
        return counts

    def branches(self, compartment: str | None = None) -> dict:
        # ordered proximal -> distal so the checklist + auto-advance sweep the cell soma-outward.
        # `compartment` restricts the checklist -- "axon" to that dominant type, "all" to the whole
        # skeleton with the myelin coverage dimension, absent for main.ts's unfiltered sweep.
        comp_filter, _ = _scope(compartment)
        all_metas = [self.branch_metadata(i) for i in self._branch_order()]
        metas = (
            [m for m in all_metas if m["compartment"] == comp_filter] if comp_filter else all_metas
        )
        # myelin_summary counts the set actually being myelin-reviewed, which is why it follows the
        # scope rather than the raw filter: pinned to axon it could never reach 0 in whole-skeleton
        # scope, and pinned to `metas` it would read 0/0 for main.ts, which doesn't review myelin.
        summary_metas = self._myelin_scope_metas(all_metas, compartment)
        return {
            "branches": metas,
            "summary": self.coverage.summary(self.tree),
            "myelin_summary": self._tally_myelin_state(summary_metas),
            "myelin_done": self.myelin_done,
            "myelin_done_ts": self.myelin_done_ts,
            "myelin_scope": self.myelin_scope,
        }

    def _myelin_scope_metas(self, all_metas: list[dict], compartment: str | None = None) -> list[dict]:
        """The branches the myelin dimension covers: everything in "all" scope, axon otherwise.

        ``compartment`` lets a caller ask about a scope other than the session's persisted one
        (the branches endpoint takes it straight from the URL); everything else passes None and
        gets the cell's own recorded scope.
        """
        scope = compartment if compartment in ("axon", "all") else self.myelin_scope
        if scope == "all":
            return all_metas
        return [m for m in all_metas if m["compartment"] == "axon"]

    # ------------------------------------------------------------------ #
    # camera path (+ build the branch tube)
    # ------------------------------------------------------------------ #
    def camera_path(self, path_id: int, orient: bool | None = None, compartment: str | None = None) -> dict:
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
            try:
                # raw verts; resamples internally
                tube.fill_branch(pid, verts_nm, verbose=False, on_progress=self._progress_cb(pid))
            finally:
                self._clear_progress(pid)
            seconds = round(time.time() - t0, 1)

        rel = f"tube/{self.datastack}/{self.root_id}"
        # build the next to-review branch(es) in the bg, in the SAME sequence the caller is
        # actually reviewing (see _queue_prebuild's docstring)
        prebuilding = self._queue_prebuild(pid, compartment=compartment)
        return {
            "path_id": pid,
            "root_id": str(self.root_id),  # string: exceeds JS 2^53 safe-int range
            "resolution_nm": [int(x) for x in self.res],
            "points_nm": rs.tolist(),
            "nodes_nm": verts_nm.tolist(),  # TRUE skeleton vertices (sparse) vs. the resampled points_nm
            "orientations": orientations,
            "step_nm": self.step_nm,
            "em_rel": f"{rel}/{tube.em_name}",
            # mask lives in a mip-keyed directory (see CellTube.__init__), NOT always "tgt"
            "tgt_rel": f"{rel}/{tube.tgt_name_dir}",
            "build": {"cached": bool(cached), "seconds": seconds},
            "prebuilding": prebuilding,
            # Set only when this branch's mask was just built and matched NOTHING -- an overlay
            # that renders perfectly invisibly, which the user can only detect as an absence.
            "mask_warning": (
                f"mask is empty for this branch -- no voxel belongs to root {self.root_id}"
                if getattr(tube, "empty_mask", False) else None
            ),
        }

    # ------------------------------------------------------------------ #
    # background pre-build (M4: hide the next branch's build behind review time)
    # ------------------------------------------------------------------ #
    def _queue_prebuild(self, after_path_id: int, compartment: str | None = None) -> list[int]:
        """Queue background builds of the next to-review branch(es) (proximal->distal) after the
        current one. Returns the path ids queued (for an optional HUD hint).

        ``compartment`` scopes this to the SAME sequence the caller is actually navigating.
        Without it, this walks the full, unfiltered branch order using the error-review
        coverage (``self.coverage``) -- correct for main.ts's unfiltered sweep, but WRONG for
        the myelin fly-through, which visits its own branch set in myelin-coverage order:
        prebuilding from the unfiltered order would almost always guess a branch the myelin
        tool was never going to load next (one it already marked myelin-covered, or -- in axon
        scope -- a dendrite branch), wasting the one background worker while the branch actually
        coming up next builds on-demand instead. See :func:`_scope` for how ``"axon"``/``"all"``
        select both the candidate list and the coverage dimension.
        """
        if self.prebuild_ahead <= 0:
            return []
        comp_filter, state_key = _scope(compartment)
        all_metas = {pid: self.branch_metadata(pid) for pid in self._branch_order()}
        order = [
            pid for pid in self._branch_order()
            if comp_filter is None or all_metas[pid]["compartment"] == comp_filter
        ]
        try:
            start = order.index(int(after_path_id)) + 1
        except ValueError:
            start = 0
        queued: list[int] = []
        for pid in order[start:]:
            if len(queued) >= self.prebuild_ahead:
                break
            if all_metas[pid][state_key] != "to_review" or self._branch_done(pid):
                continue
            with self._prebuild_lock:
                if pid in self._prebuilding:
                    continue
                self._prebuilding.add(pid)
            self._prebuild_ex.submit(self._prebuild, pid, self._epoch)
            queued.append(int(pid))
        return queued

    def _progress_cb(self, path_id: int):
        """Build an ``on_progress`` callback that records this branch's live fill progress."""
        pid = int(path_id)

        def cb(phase: str, done: int, total: int) -> None:
            with self._fill_progress_lock:
                prev = self._fill_progress.get(pid)
                # accumulate the DELTA so the running total stays monotonic across the em->tgt
                # phase switch (which restarts `done` at 0 against a different total)
                prev_done = prev["done"] if prev and prev["phase"] == phase else 0
                self._chunks_cached += max(0, int(done) - prev_done)
                self._fill_progress[pid] = {
                    "phase": phase, "done": int(done), "total": int(total), "ts": time.time(),
                }

        return cb

    def _clear_progress(self, path_id: int) -> None:
        with self._fill_progress_lock:
            self._fill_progress.pop(int(path_id), None)

    def warm_status(self) -> dict:
        """Live caching progress: what's filling right now, how far along, and how long since it
        last advanced (``stalled_s``) so the UI can flag a wedged read rather than just spinning.
        """
        now = time.time()
        with self._fill_progress_lock:
            active = [
                {"path_id": pid, **p, "stalled_s": round(now - p["ts"], 1)}
                for pid, p in self._fill_progress.items()
            ]
        for a in active:
            a.pop("ts", None)
        active.sort(key=lambda a: a["path_id"])
        with self._prebuild_lock:
            queued = len(self._prebuilding)
        return {
            "active": active,
            "in_flight": queued,
            "chunks_cached": self._chunks_cached,  # monotonic; the reliable "still alive" signal
        }

    def warm_targets(self, compartment: str | None = None) -> dict:
        """Branches that still need a tube build, in review order (proximal->distal).

        Split out of :meth:`warm_cell` so an external driver -- the multi-cell warm queue in
        :mod:`proofreading.em.warm_queue` -- can walk the same list ONE branch at a time via
        :meth:`warm_branch` instead of dumping it all on this session's background executor.
        That is what makes a queued cell interruptible between branches, and what lets the
        queue know how much work a cell actually represents before it starts.
        """
        comp_filter, state_key = _scope(compartment)
        all_metas = {pid: self.branch_metadata(pid) for pid in self._branch_order()}
        order = [
            pid for pid in self._branch_order()
            if comp_filter is None or all_metas[pid]["compartment"] == comp_filter
        ]
        pending: list[int] = []
        already_built = 0
        for pid in order:
            if self._branch_done(pid):
                already_built += 1
                continue
            if all_metas[pid][state_key] != "to_review":
                continue
            pending.append(int(pid))
        return {"pending": pending, "already_built": already_built, "total": len(order)}

    def branch_built(self, path_id: int) -> bool:
        """True once this branch's tube is cached at the mips in effect (public: the warm queue
        checks it to tell a real build from one that hit :meth:`fill_branch`'s time budget)."""
        return self._branch_done(int(path_id))

    def warm_cell(self, compartment: str | None = None) -> dict:
        """Queue background tube builds for EVERY remaining to-review branch, not just the next
        ``prebuild_ahead``.

        Same machinery as :meth:`_queue_prebuild` with the lookahead cap lifted -- it already
        skips branches that are built / not to-review / in flight, and :meth:`_prebuild` already
        aborts stale work after a re-root (``_epoch``). The single-worker ``_prebuild_ex`` keeps
        these serial so a long warm-up can't starve the on-demand fetch the user is waiting on;
        the point is that it runs AHEAD of time, not that it runs wider.
        """
        t = self.warm_targets(compartment)
        queued: list[int] = []
        for pid in t["pending"]:
            with self._prebuild_lock:
                if pid in self._prebuilding:
                    continue
                self._prebuilding.add(pid)
            self._prebuild_ex.submit(self._prebuild, pid, self._epoch)
            queued.append(int(pid))
        return {"queued": queued, "n_queued": len(queued),
                "already_built": t["already_built"], "total": t["total"]}

    def warm_branch(self, path_id: int) -> bool:
        """Build ONE branch's tube synchronously, in the CALLER's thread.

        For the multi-cell warm queue, which serializes the work itself and therefore must not
        hand it to this session's executor. Returns False if this session's own pre-builder
        already has the branch in flight -- the queue skips it rather than blocking, since
        ``_prebuild``'s per-branch lock means waiting would only buy a duplicate early-out.
        """
        pid = int(path_id)
        with self._prebuild_lock:
            if pid in self._prebuilding:
                return False
            self._prebuilding.add(pid)  # _prebuild's `finally` discards it
        self._prebuild(pid, self._epoch)
        return True

    def _prebuild(self, path_id: int, epoch: int) -> None:
        """Background worker: build one branch's tube unless a re-root (epoch bump) invalidated it.

        Shares ``_branch_locks[pid]`` with on-demand ``camera_path``, so a fetch of a branch that
        is mid-pre-build just blocks until it's done, then sees it cached -- never a double build.
        """
        pid = int(path_id)
        try:
            if epoch != self._epoch or self._branch_done(pid):
                return  # early-out: re-rooted or already built before we got scheduled
            tube = self.ensure_tube()
            with self._branch_locks[pid]:
                # everything under the lock + the verified epoch, so a concurrent re-root can't
                # make us fill stale verts or leave a stale .done that an on-demand fetch observes
                if epoch != self._epoch or self._branch_done(pid):
                    return
                bp = self.tree.branch_paths[pid]
                verts = np.asarray(self.tree.vertices[bp.vertices], dtype=float)
                tube.fill_branch(pid, verts, verbose=False, on_progress=self._progress_cb(pid))
                if epoch != self._epoch:
                    # re-rooted DURING the fill: these markers are for the OLD decomposition ->
                    # drop them inside the lock, before any on-demand fetch can see them. All
                    # per-phase variants, not just the legacy name (see tube.branch_marker_state).
                    mdir = os.path.join(self.tube_cache_dir, "_branches")
                    for name in (f"path_{pid}.done", f"path_{pid}.em.done",
                                 f"path_{pid}.tgt{tube.tgt_mip}.done"):
                        try:
                            os.remove(os.path.join(mdir, name))
                        except OSError:
                            pass
        except Exception:
            pass  # best-effort; an on-demand fetch will build it if needed
        finally:
            self._clear_progress(pid)
            with self._prebuild_lock:
                self._prebuilding.discard(pid)

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

    # ------------------------------------------------------------------ #
    # myelin tags -- a discrete, per-NODE tag ("myelinated"), same shape as the review Tags
    # but a separate vocabulary/file (see CONTEXT.md, wal.py's module docstring). Presence of
    # a tag on a node means myelinated; absence means the unmyelinated default.
    # ------------------------------------------------------------------ #
    def tag_myelinated_node(self, xyz_nm, path_id: int | None = None) -> dict:
        """Tag the single skeleton vertex nearest ``xyz_nm`` as myelinated. In axon scope, warns
        (does not block) if that vertex isn't classified as axon -- compartment labels can be
        imperfect, so an off-axon tag is worth flagging but not refusing.
        """
        vertex = int(self.tree.nearest_vertex(xyz_nm))
        warning = None
        # In "all" scope, tagging a dendrite is the POINT, so the warning would fire on nearly
        # every tag -- and a warning that always fires is one you stop reading.
        if self.myelin_scope != "all" and self.tree.compartment is not None:
            code = int(self.tree.compartment[vertex])
            comp = _COMPARTMENT.get(code, "unknown")
            if comp != "axon":
                warning = f"nearest skeleton vertex is classified '{comp}', not axon"
        snapped_xyz = self.tree.vertices[vertex].tolist()
        tag = self.myelin_wal.tag_myelinated(
            snapped_xyz, self.root_id, self.mat_version, self.seed, path_id,
        )
        return {"uuid": tag.uuid, "xyz_nm": snapped_xyz, "warning": warning}

    def delete_myelin_tag(self, uuid: str) -> dict:
        """Remove a myelin tag, reverting that node to the unmyelinated default."""
        self.myelin_wal.delete_myelin_tag(uuid)
        return {"uuid": uuid}

    def set_myelin_cell_done(self, done: bool) -> dict:
        """Declare (or un-declare) the whole cell finished for myelin tagging. Independent of any
        branch's own `built`/coverage state -- it's the signal the reopen-last-cell flow checks
        so a finished cell isn't handed back to you again on the next launch."""
        self.myelin_wal.set_cell_done(done)
        # Re-read rather than stamp our own clock: the WAL's `_now()` (inside `_write`) is the one
        # source of truth for `ts`, and every other read of this state already goes through
        # `WAL.load` (see `myelin_tags` above) rather than duplicating timestamp formatting here.
        _state = WAL.load(self.myelin_wal.path)
        self.myelin_done = _state.cell_done
        self.myelin_done_ts = _state.cell_done_ts
        return {
            "myelin_done": self.myelin_done,
            "myelin_done_ts": self.myelin_done_ts,
            "myelin_summary": self._tally_myelin_state(
                self._myelin_scope_metas(
                    [self.branch_metadata(i) for i in self._branch_order()]
                )
            ),
        }

    def set_myelin_scope(self, scope: str) -> dict:
        """Set which part of the skeleton this cell is annotated over ("axon" | "all").

        Durable, because it changes what "reviewed" and "cell done" mean for this cell, and
        because reopening should resume the scope you were working in rather than snapping back
        to the axon default. Deliberately does NOT touch the done mark: switching scope on a
        finished cell leaves it marked (the badge and `undo cell done` are right there) -- quietly
        clearing a durable mark because a dropdown moved is exactly what makes a log untrustworthy.
        """
        if scope not in ("axon", "all"):
            raise ValueError(f"unknown scope {scope!r}; expected 'axon' or 'all'")
        self.myelin_wal.set_scope(scope)
        self.myelin_scope = scope
        all_metas = [self.branch_metadata(i) for i in self._branch_order()]
        comp_filter, _ = _scope(scope)
        return {
            "myelin_scope": self.myelin_scope,
            "branches": (
                [m for m in all_metas if m["compartment"] == comp_filter]
                if comp_filter else all_metas
            ),
            "myelin_summary": self._tally_myelin_state(
                self._myelin_scope_metas(all_metas, scope)
            ),
        }

    def myelin_tags(self, path_id: int | None = None) -> dict:
        """Live myelin tags, optionally restricted to one branch (for refreshing just that
        branch's overlay when it loads)."""
        state = WAL.load(self.myelin_wal.path)
        tags = [
            {"uuid": t.uuid, "xyz_nm": [float(c) for c in t.xyz], "path_id": t.path_id}
            for t in state.myelin_tags.values()
            if path_id is None or t.path_id == int(path_id)
        ]
        return {"tags": tags}

    # datastack with the editable production segmentation (minnie3_v1); used for Spelunker links
    # regardless of the session datastack so links always open the live proofreading table
    _SPELUNKER_DATASTACK = "minnie65_phase3_v1"

    def _spelunker_client(self):
        """Lazy-cached CAVEclient for the production datastack (for Spelunker URL generation)."""
        if not hasattr(self, "_spelunker_client_cache"):
            from caveclient import CAVEclient
            self._spelunker_client_cache = CAVEclient(self._SPELUNKER_DATASTACK)
        return self._spelunker_client_cache

    def _spelunker_url(self, xyz_nm: list) -> str:
        """Spelunker URL at the given nm position with this cell's root selected.

        Always uses the minnie65_phase3_v1 CAVEclient so nglui derives the production
        minnie3_v1 segmentation source, even when the session is against minnie65_public.
        """
        from nglui.statebuilder import ViewerState
        client = self._spelunker_client()
        res = np.array(client.info.viewer_resolution(), dtype=float)
        pos = (np.array(xyz_nm, dtype=float) / res).tolist()
        return (
            ViewerState(client=client, position=pos, show_slices=True, layout="xy-3d")
            .add_layers_from_client(
                client=client,
                segmentation_kws={"segments": [str(self.root_id)]},
            )
            .to_url(target_site="spelunker")
        )

    def list_annotations(self) -> list:
        """Live annotations from the durable log (for redraw on resume)."""
        st = WAL.load(self.wal.path)
        return [
            {**self._ann(a), "done": a.uuid in st.done_uuids, "spelunker_url": self._spelunker_url(a.xyz)}
            for a in st.annotations.values()
        ]

    def toggle_annotation_status(self, uuid: str) -> dict:
        """Toggle the done/todo status of an annotation and persist to the WAL."""
        state = WAL.load(self.wal.path)
        if uuid not in state.annotations:
            raise KeyError(f"annotation {uuid} not found")
        done = uuid not in state.done_uuids  # flip current state
        self.wal.set_annotation_done(uuid, done)
        return {"uuid": uuid, "done": done}

    def resolve(self) -> dict:
        """Batch-resolve annotation click positions → supervoxel IDs, then re-resolve the cell root.

        Called once per Phase B session (after edits are done in Spelunker). Fills ``supervoxel``
        on every live annotation in the WAL via ``points_to_supervoxels`` + a ``resolve_supervoxel``
        write per annotation, then calls ``current_root(self.seed)`` to find the cell's new root.
        The seed supervoxel (soma) is stable across split/merge edits; the root ID is not.
        """
        state = WAL.load(self.wal.path)
        anns = list(state.annotations.values())
        if not anns:
            return {"annotations": [], "new_root_id": str(self.root_id), "resolved_count": 0}
        xyzs = np.array([a.xyz for a in anns])           # (N, 3) nm
        svs = self.client.points_to_supervoxels(xyzs)    # (N,) uint64
        for ann, sv in zip(anns, svs):
            self.wal.resolve_supervoxel(ann.uuid, int(sv))
        new_root = self.client.current_root(self.seed)
        return {
            "annotations": [{**self._ann(a), "spelunker_url": self._spelunker_url(a.xyz)} for a in anns],
            "new_root_id": str(new_root),
            "resolved_count": len(anns),
        }

    def delete_annotation(self, uuid: str) -> dict:
        """Soft-delete an annotation (tombstone in the WAL) and rebuild coverage.

        Deleting an ``omit_branch`` mark reverts its omission: the rebuilt coverage drops the
        ``omit`` event keyed to this uuid, so the branch + descendants return to ``to_review``. We
        return the refreshed checklist so the frontend repaints in one round-trip. Idempotent.
        """
        existed = uuid in WAL.load(self.wal.path).annotations
        self.wal.tombstone(uuid)
        self.coverage = Coverage.from_wal_state(WAL.load(self.wal.path))
        return {
            "deleted": existed,
            "uuid": uuid,
            "summary": self.coverage.summary(self.tree),
            "branches": [self.branch_metadata(i) for i in self._branch_order()],
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

    def myelin_mark_done(self, path_id: int) -> dict:
        """Mark a branch reviewed FOR MYELINATION (a separate coverage dimension from
        :meth:`mark_done` -- see wal.py's ``myelin_visit`` docstring), and advance to the next
        to-review branch within this cell's myelin scope (axon only, or the whole skeleton).

        Reads ``self.myelin_scope`` rather than taking a compartment argument: it's a myelin-only
        method, and the scope is a durable property of the cell, not of the request.
        """
        pid = int(path_id)
        bp = self.tree.branch_paths[pid]
        l2 = self.tree.l2_ids_for_vertices(bp.vertices)
        self.myelin_wal.mark_myelin_visited(l2, root_id=self.root_id)
        self.myelin_coverage.mark_visited(l2)
        todo = set(self.myelin_coverage.to_review(self.tree))
        scope_metas = self._myelin_scope_metas(
            [self.branch_metadata(i) for i in self._branch_order()]
        )
        next_pid = next((m["path_id"] for m in scope_metas if m["path_id"] in todo), None)
        return {
            "path_id": pid,
            "next_path_id": next_pid,
            "myelin_summary": self._tally_myelin_state(scope_metas),
            "branches": scope_metas,
        }

    def omit_branch(self, path_id: int) -> dict:
        """Omit a branch + its DISTAL DESCENDANTS from review (a foreign segment to be split off).

        Per-branch on purpose (NOT prune-everything-distal-from-a-vertex): at an X crossing the
        cell's true continuation is a *sibling* branch, so omitting only this branch's subtree
        (``subtree_mask(bp.vertices[1])`` -- excludes the shared junction + siblings) leaves the
        continuation to-review. Anchored to a 'merge error' mark at the junction so it persists and
        is reversed by deleting that mark (see :meth:`delete_annotation`); the mark also records
        WHERE the split is needed for the manual edit. 'Distal' is relative to the current root.
        """
        pid = int(path_id)
        bp = self.tree.branch_paths[pid]
        distal_root = int(bp.vertices[1]) if len(bp.vertices) >= 2 else int(bp.vertices[0])
        mask = self.tree.subtree_mask(distal_root, include_root=True)
        l2 = self.tree.l2_ids_for_vertices(np.where(mask)[0])
        junction_xyz = [float(c) for c in self.tree.vertices[int(bp.vertices[0])]]
        ann = self.wal.add_annotation(
            "merge error", junction_xyz, self.root_id, self.mat_version, self.seed
        )
        self.wal.mark_omitted(l2, because_uuid=ann.uuid)
        self.coverage.mark_omitted(l2)
        todo = set(self.coverage.to_review(self.tree))
        next_pid = next((i for i in self._branch_order() if i in todo), None)
        return {
            "path_id": pid,
            "omitted_l2_count": int(len(l2)),
            "annotation": self._ann(ann),
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
            # False means this root has been edited since and is no longer a leaf of the
            # chunkedgraph. Everything still works -- the skeleton and the mask are both served as
            # of `agg_timestamp` -- but you are reviewing the cell AS IT WAS, which is worth
            # knowing before you tag it. None = couldn't check (offline); don't claim either way.
            "root_is_current": self.root_is_current,
            "root_timestamp": None if self.agg_timestamp is None else str(self.agg_timestamp),
        }

    def close(self) -> None:
        try:
            self._prebuild_ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.wal.close()
        except Exception:
            pass
        try:
            self.myelin_wal.close()
        except Exception:
            pass
