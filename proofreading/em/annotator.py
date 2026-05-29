"""Interactive Phase A session: fly each branch path, drop tagged annotations.

Ties the engine together: skeleton -> branch paths -> ``FlyThrough`` per path, with
neuroglancer key bindings that record the cursor ``xyz`` (nm) to the WAL. A
``merge error`` additionally prunes the strict-distal subtree from coverage.

Key map (neuroglancer viewer keys):
  m = merge error   s = split error   e = extend   q = question
  n = toggle the segment under the cursor (reveal/hide a neighbor)
  x = mark the current branch path reviewed (and advance)

Annotations record only the click ``xyz``; supervoxels are resolved in batch via
:meth:`ProofreadSession.resolve_supervoxels` (CloudVolume). Everything is durable
in the WAL immediately; the neuroglancer layers and coverage are views of it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import path as P
from . import viewer as V
from .coverage import Coverage, PathState
from .skeleton_tree import SkeletonTree
from .wal import WAL, TAGS
from ..flythrough import FlyThrough

# neuroglancer key name -> tag
TAG_KEYS = {
    "keym": "merge error",
    "keys": "split error",
    "keye": "extend",
    "keyq": "question",
}


class ProofreadSession:
    """One annotate session over a single cell (root id)."""

    def __init__(
        self,
        emclient,
        root_id: int,
        wal_dir: str = "./proofread_sessions",
        step_nm: float = 1000.0,
        ip: str = "localhost",
        port: int = 0,
        seconds_per_step: float = 0.5,
        frames_per_second: float = 30,
        animate: bool = True,
        dwell_seconds: float = 1.5,
        orient_to_path: bool = False,
        load_gated: bool = False,
        load_timeout: float = 15.0,
        prefetch_window: int = 3,
        cross_section_render_scale: float = 1.0,
        gpu_memory_limit: int = 2_000_000_000,
        system_memory_limit: int = 4_000_000_000,
    ):
        self.client = emclient
        self.root_id = int(root_id)
        self.step_nm = float(step_nm)
        self.seconds_per_step = seconds_per_step  # tween glide between nodes (animate mode)
        self.frames_per_second = frames_per_second
        # Default playback: smooth glide between nodes + a rest (dwell_seconds) at each
        # node so the segmentation mask loads and is seen without pausing (~2 s/node total
        # lets caching keep up). animate=False jump-cuts instead. load_gated is an opt-in
        # adaptive gate (can over-wait on 3D meshes/prefetch), off by default.
        self.animate = animate
        # Default: smooth glide between nodes + a brief rest *at* each node. The rest is
        # required, not cosmetic: neuroglancer only renders the (expensive) segmentation
        # when the camera is idle, so it paints during the rest. (Manual scrolling shows
        # the seg because it has natural idle gaps; continuous motion has none.)
        # dwell_seconds=0 -> pure continuous glide, but then the seg only appears on pause.
        self.dwell_seconds = dwell_seconds
        # orient_to_path=False keeps the native axis-aligned XYZ sections (fast, loads like
        # manual scrolling); True tilts panel 1 to a cross-section ⊥ the neurite (nicer for
        # judging merges, but oblique slices are slower to stream).
        self.orient_to_path = orient_to_path
        self.load_gated = load_gated
        self.load_timeout = load_timeout
        # prefetch: warm the next N nodes (both directions) so frames are ready on arrival.
        self.prefetch_window = int(prefetch_window)
        self._fly_pos_vox = None
        self._fly_quats = None

        # cell identity + structure
        self.sk = emclient.get_skeleton(root_id)
        self.tree = SkeletonTree.from_skeleton_dict(self.sk)
        self.seed = emclient.seed_supervoxel(self.sk)
        self.mat_version = emclient.mat_version

        # durable state (resumes a prior log for this cell, keyed by seed supervoxel)
        self.wal = WAL.for_cell(wal_dir, emclient.datastack, self.seed)
        self.coverage = Coverage.from_wal_state(WAL.load(self.wal.path))

        # viewer
        self.viewer, self.res, self.dims = V.make_em_viewer(
            emclient, root_id, ip=ip, port=port,
            cross_section_render_scale=cross_section_render_scale,
            gpu_memory_limit=gpu_memory_limit,
            system_memory_limit=system_memory_limit,
        )
        self._restore_annotations()

        self.fly: Optional[FlyThrough] = None
        self.current_path_id: Optional[int] = None
        self._positions_nm: Optional[np.ndarray] = None

        self._bind_keys()

    # ------------------------------------------------------------------ #
    # navigation
    # ------------------------------------------------------------------ #
    def review_path(self, path_id: int) -> FlyThrough:
        """Load a branch path's camera path and start a (paused) fly-through."""
        bp = self.tree.branch_paths[int(path_id)]
        self.current_path_id = int(path_id)
        verts_nm = self.tree.vertices[bp.vertices]
        rs = P.resample_path(verts_nm, self.step_nm)
        if self.orient_to_path:
            T, N, B = P.rotation_minimizing_frame(rs)
            quats = P.frame_to_quaternion(T, N, B)
        else:
            quats = None  # keep the native axis-aligned sections (just move position)
        self._positions_nm = rs
        self._fly_pos_vox = rs / self.res  # for prefetch
        self._fly_quats = quats

        V.show_branch_path(self.viewer, verts_nm / self.res)
        if self.fly is not None:
            self.fly.stop()
        self.fly = FlyThrough(
            self.viewer,
            rs / self.res,  # nm -> viewer voxels
            quats,
            seconds_per_step=self.seconds_per_step,
            frames_per_second=self.frames_per_second,
            settle=self._make_settle(),
            animate=self.animate,
            dwell_seconds=self.dwell_seconds,
        )
        self.fly.start()
        return self.fly

    def _make_settle(self):
        """Per-node worker callback: prefetch upcoming nodes, load-gate, then dwell.

        Reads the knobs dynamically, so the live setters take effect on a running
        fly-through without rebuilding it.
        """

        def settle():
            if self.prefetch_window > 0:
                self._update_prefetch()
            if self.load_gated:
                V.wait_until_loaded(self.viewer, self.load_timeout)

        return settle

    def _update_prefetch(self) -> None:
        """Prefetch the nodes just ahead/behind the camera (nearest = highest priority)."""
        if self._fly_pos_vox is None or self.fly is None:
            return
        i = self.fly.index
        n = len(self._fly_pos_vox)
        k = self.prefetch_window
        order = []
        for d in range(1, k + 1):  # interleave ahead/behind, nearest first
            order += [i + d, i - d]
        nav = []
        for rank, j in enumerate(order):
            if 0 <= j < n:
                ori = None if self._fly_quats is None else self._fly_quats[j]
                nav.append((self._fly_pos_vox[j], ori, 2 * k - rank))
        V.set_prefetch(self.viewer, nav)

    def set_load_gated(self, enabled: bool) -> None:
        """Turn load-gated playback on/off live (also updates a running fly-through)."""
        self.load_gated = bool(enabled)
        if self.fly is not None:
            self.fly.settle = self._make_settle()

    def set_animate(self, animate: bool) -> None:
        """Toggle smooth tween (True) vs jump-cut (False) playback, live."""
        self.animate = bool(animate)
        if self.fly is not None:
            self.fly.animate = self.animate

    def set_dwell(self, seconds: float) -> None:
        """Set the rest at each node during autoplay (live) -- lets the seg mask load."""
        self.dwell_seconds = float(seconds)
        if self.fly is not None:
            self.fly.dwell_seconds = self.dwell_seconds

    def set_orient_to_path(self, enabled: bool) -> None:
        """Toggle cross-section ⊥ neurite (True) vs native axis-aligned sections (False).

        Rebuilds the current path's camera so it takes effect immediately. Axis-aligned
        (False) streams much faster (native chunk layout, like manual scrolling).
        """
        self.orient_to_path = bool(enabled)
        if self.current_path_id is not None:
            self.review_path(self.current_path_id)

    def set_render_scale(self, scale: float) -> None:
        """Change the EM cross-section render scale live.

        ``1.0`` = full resolution (mip0, sharpest/slowest); higher = coarser & faster
        (``2.0`` ~ mip1). Toggle mip0<->mip1 without recreating the session.
        """
        with self.viewer.txn() as s:
            s.layers[V.IMAGE_LAYER].cross_section_render_scale = float(scale)

    def review_next(self) -> Optional[FlyThrough]:
        """Jump to the next branch path still ``to_review``."""
        todo = self.coverage.to_review(self.tree)
        return self.review_path(todo[0]) if todo else None

    # ------------------------------------------------------------------ #
    # key bindings
    # ------------------------------------------------------------------ #
    def _bind_keys(self) -> None:
        for tag in TAGS:
            self.viewer.actions.add(
                f"annotate:{tag}", lambda s, t=tag: self._on_annotate(t, s)
            )
        self.viewer.actions.add("toggle-neighbors", lambda s: self._on_toggle_neighbors(s))
        self.viewer.actions.add("mark-path-done", lambda s: self._on_mark_done())
        with self.viewer.config_state.txn() as cs:
            for key, tag in TAG_KEYS.items():
                cs.input_event_bindings.viewer[key] = f"annotate:{tag}"
            cs.input_event_bindings.viewer["keyn"] = "toggle-neighbors"
            cs.input_event_bindings.viewer["keyx"] = "mark-path-done"

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def _mouse_nm(self, action_state) -> Optional[np.ndarray]:
        pos = getattr(action_state, "mouse_voxel_coordinates", None)
        if pos is None:
            return None
        return np.asarray(pos, dtype=float) * self.res

    def _on_annotate(self, tag: str, action_state):
        xyz_nm = self._mouse_nm(action_state)
        if xyz_nm is None:
            return None
        ann = self.wal.add_annotation(
            tag, xyz_nm, self.root_id, self.mat_version, self.seed
        )
        V.add_point(self.viewer, tag, xyz_nm / self.res, ann.uuid)
        if tag == "merge error":
            self._prune_merge_error(xyz_nm, ann.uuid)
        return ann

    def _prune_merge_error(self, xyz_nm: np.ndarray, ann_uuid: str):
        v = self.tree.nearest_vertex(xyz_nm)
        pr = self.tree.prune_merge_error(v)
        if len(pr.omitted_l2_ids):
            self.wal.mark_omitted(pr.omitted_l2_ids, because_uuid=ann_uuid)
            self.coverage.mark_omitted(pr.omitted_l2_ids)
        # park the camera at the merge point if it's on the current path
        if self.fly is not None and self._positions_nm is not None:
            d = self._positions_nm - self.tree.vertices[v]
            self.fly.seek(int(np.argmin(np.einsum("ij,ij->i", d, d))))
        return pr

    def _on_mark_done(self):
        if self.current_path_id is None:
            return None
        bp = self.tree.branch_paths[self.current_path_id]
        l2 = self.tree.l2_ids_for_vertices(bp.vertices)
        self.wal.mark_visited(l2)
        self.coverage.mark_visited(l2)
        return self.review_next()

    def _segment_under_cursor(self, action_state) -> Optional[int]:
        sel = getattr(action_state, "selected_values", None)
        if not sel:
            return None
        try:
            val = sel[V.SEG_LAYER]
            seg = getattr(val, "value", val)
            return None if seg is None else int(seg)
        except (KeyError, TypeError, ValueError):
            return None

    def _on_toggle_neighbors(self, action_state) -> None:
        seg = self._segment_under_cursor(action_state)
        if seg is None:
            return
        with self.viewer.txn() as s:
            segs = {int(x) for x in s.layers[V.SEG_LAYER].segments}
            if seg in segs and seg != self.root_id:
                segs.discard(seg)
            else:
                segs.add(seg)
            s.layers[V.SEG_LAYER].segments = list(segs)

    # ------------------------------------------------------------------ #
    # supervoxel resolution (batch, off the hot path) + restore
    # ------------------------------------------------------------------ #
    def resolve_supervoxels(self) -> int:
        """Batch-resolve any annotations missing a supervoxel; returns how many."""
        state = WAL.load(self.wal.path)
        pending = [a for a in state.annotations.values() if a.supervoxel is None]
        if not pending:
            return 0
        svs = self.client.points_to_supervoxels(np.array([a.xyz for a in pending]))
        for a, sv in zip(pending, svs):
            self.wal.resolve_supervoxel(a.uuid, int(sv))
        return len(pending)

    def _restore_annotations(self) -> None:
        """Redraw annotations from a prior session's log onto the viewer."""
        for a in WAL.load(self.wal.path).annotations.values():
            V.add_point(self.viewer, a.tag, np.asarray(a.xyz) / self.res, a.uuid)

    # ------------------------------------------------------------------ #
    # status / lifecycle / UI
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        return self.coverage.summary(self.tree)

    def close(self) -> None:
        if self.fly is not None:
            self.fly.stop()
        self.wal.close()

    def panel(self):
        """An ipywidgets control panel (checklist + review/resolve buttons)."""
        from ipywidgets import Button, Dropdown, HBox, Label, Output, VBox
        from IPython.display import display

        out = Output()
        status = Label(value=self._status_text())
        todo = self.coverage.to_review(self.tree)
        dd = Dropdown(options=todo, description="path", value=todo[0] if todo else None)
        b_review = Button(description="Review", button_style="primary")
        b_done = Button(description="Mark done (x)")
        b_resolve = Button(description="Resolve supervoxels")

        def refresh():
            t = self.coverage.to_review(self.tree)
            dd.options = t
            status.value = self._status_text()

        def on_review(_b):
            with out:
                if dd.value is not None:
                    self.review_path(dd.value)
                    from ..controls import FlyThroughControls
                    FlyThroughControls(self.fly)

        def on_done(_b):
            with out:
                self._on_mark_done()
                refresh()

        def on_resolve(_b):
            with out:
                print("resolved", self.resolve_supervoxels(), "supervoxels")

        b_review.on_click(on_review)
        b_done.on_click(on_done)
        b_resolve.on_click(on_resolve)
        panel = VBox([HBox([dd, b_review, b_done, b_resolve]), status, out])
        display(panel)
        return panel

    def _status_text(self) -> str:
        s = self.summary()
        return (f"root {self.root_id} | seed {self.seed} | "
                f"to_review {s['to_review']}  covered {s['covered']}  omitted {s['omitted']}")
