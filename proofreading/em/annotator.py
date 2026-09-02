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

import copy
import os
from typing import Optional

import numpy as np

from . import path as P
from . import preview as PV
from . import render as R
from . import tube as T
from . import viewer as V
from .coverage import Coverage, PathState
from .review import BranchPlayer
from .skeleton_tree import SkeletonTree
from .wal import WAL, TAGS
from ..flythrough import FlyThrough, build_target_state

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
        tube_mip: int = 1,
        tube_radius_nm: float = 1000.0,
        preview_target_nm: float = 256.0,
        preview_pad_nm: float = 1500.0,
        preview_max_voxels: int = 25_000_000,
        load_gated: bool = False,
        load_timeout: float = 15.0,
        prefetch_window: int = 0,
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
        # tube glide (DEFAULT review): a sparse local precomputed EM+target along the branch
        # (only the chunks within tube_radius_nm of the skeleton), served to neuroglancer and
        # rendered sharp at mip1 *during motion*; pause flips to the live img+seg. See tube.py.
        self.tube_mip = int(tube_mip)
        self.tube_radius_nm = float(tube_radius_nm)
        self.tube_cache_dir = os.path.join(wal_dir, "tube_cache", emclient.datastack, str(self.root_id))
        self._tube = None              # CellTube: ONE shared per-cell precomputed, filled lazily
        self._tube_layers_added = False  # the single persistent tube layer pair (never swapped)
        # in-memory coarse preview glide (mode='preview' fallback): a precomputed local layer at ~target_nm
        # over the branch bbox (+pad), so the camera can glide smoothly with the target
        # visible; pausing reveals the live img+seg for annotation. See preview.py.
        self.preview_target_nm = float(preview_target_nm)
        self.preview_pad_nm = float(preview_pad_nm)
        self.preview_max_voxels = int(preview_max_voxels)
        self._preview = None
        self.load_gated = load_gated
        self.load_timeout = load_timeout
        # prefetch (opt-in, default off): warm the next N nodes by injecting extra full
        # ViewerStates for the browser to render -- can add jank, so off unless you want it.
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
        # pre-rendered fly-through frames, keyed by the (geometry-specific) root id
        self.render_dir = os.path.join(wal_dir, "renders", emclient.datastack, str(self.root_id))

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
        self._controls = None  # FlyThroughControls shown by panel() (rebuilt per branch)
        self._on_fly_change = None  # panel() sets this; review_path fires it when self.fly changes

        self._bind_keys()

    # ------------------------------------------------------------------ #
    # navigation
    # ------------------------------------------------------------------ #
    def _compute_path(self, path_id: int):
        """Resample a branch path -> ``(verts_nm, positions_nm, positions_vox, quats)``.

        ``quats`` is ``None`` unless ``orient_to_path`` (then a per-node cross-section ⊥
        the neurite). Shared by the live fly-through, the pre-render, and ``goto_node``.
        """
        bp = self.tree.branch_paths[int(path_id)]
        verts_nm = self.tree.vertices[bp.vertices]
        rs = P.resample_path(verts_nm, self.step_nm)
        if self.orient_to_path:
            T, N, B = P.rotation_minimizing_frame(rs)
            quats = P.frame_to_quaternion(T, N, B)
        else:
            quats = None  # keep the native axis-aligned sections (just move position)
        return verts_nm, rs, rs / self.res, quats

    def review_path(self, path_id: int, mode: str = "tube") -> FlyThrough:
        """Load a branch path's camera path and start a (paused) fly-through.

        ``mode``:
          - ``"tube"`` (default): a **sparse local precomputed** EM+target tube at mip1 along
            the branch (only the chunks within ``tube_radius_nm`` of the skeleton), served to
            neuroglancer. Sharp continuous glide (``dwell=0``); the target is baked in; pausing
            flips to the live ``img``+``seg`` to annotate. Per-branch result is cached, so
            revisiting is instant.
          - ``"preview"``: an in-memory coarse local preview (cheaper on disk, blockier).
          - ``"live"``: no precompute -- the old live glide with a rest at each node so the
            live seg paints (can't show seg while moving).
        """
        self.current_path_id = int(path_id)
        verts_nm, rs, pos_vox, quats = self._compute_path(path_id)
        self._positions_nm = rs
        self._fly_pos_vox = pos_vox  # for prefetch
        self._fly_quats = quats

        V.show_branch_path(self.viewer, verts_nm / self.res)
        if self.fly is not None:
            self.fly.stop()

        on_play = on_pause = settle = None
        dwell = self.dwell_seconds
        self._preview = None

        if mode == "tube":
            try:
                self._ensure_tube_layers()              # ONE shared per-cell volume + persistent layers
                self._tube.fill_branch(path_id, verts_nm)  # write this branch's chunks (skips if cached)
                dwell = 0.0
                on_play = lambda: V.set_preview_mode(self.viewer, True)
                on_pause = lambda: V.set_preview_mode(self.viewer, False)
            except Exception as exc:  # no token / network / etc. -> live-glide fallback
                import warnings

                warnings.warn(f"tube build failed, falling back to live glide: {exc!r}")
                mode = "live"

        if mode in ("preview", "live"):  # not tube -> clear any preview/tube layers
            PV.remove_preview_layers(self.viewer)
            self._tube_layers_added = False

        if mode == "preview":
            try:
                self._preview = PV.build_preview(
                    self.client, rs, self.root_id,
                    target_nm=self.preview_target_nm, pad_nm=self.preview_pad_nm,
                    max_voxels=self.preview_max_voxels,
                )
                PV.add_preview_layers(self.viewer, self._preview, visible=False)
                dwell = 0.0
                on_play = lambda: V.set_preview_mode(self.viewer, True)
                on_pause = lambda: V.set_preview_mode(self.viewer, False)
            except Exception as exc:
                import warnings

                warnings.warn(f"preview build failed, falling back to live glide: {exc!r}")
                self._preview = None
                mode = "live"

        if mode == "live":  # rest at each node so the live seg paints
            settle = self._make_settle()
        V.set_preview_mode(self.viewer, False)  # start paused -> live layers shown

        self.fly = FlyThrough(
            self.viewer,
            rs / self.res,  # nm -> viewer voxels
            quats,
            seconds_per_step=self.seconds_per_step,
            frames_per_second=self.frames_per_second,
            settle=settle,
            animate=self.animate,
            dwell_seconds=dwell,
            on_play=on_play,
            on_pause=on_pause,
        )
        self.fly.start()
        if self._on_fly_change is not None:  # let the panel rebuild its controls for the new fly
            try:
                self._on_fly_change()
            except Exception:
                pass
        return self.fly

    def _ensure_tube_layers(self) -> None:
        """Create the per-cell :class:`CellTube` and add its single (persistent) layer pair once.

        The layers point at one shared local volume and are never swapped per branch -- that's
        what keeps neuroglancer's chunk cache bounded so the glide doesn't slow over a session.
        """
        if self._tube is None:
            self._tube = T.CellTube(
                self.client, self.root_id, self.tube_mip, self.tube_radius_nm, self.tube_cache_dir,
                # agglomerate as of this root's own creation, or the mask silently comes out empty
                # for any root that has been edited since -- see EMClient.agg_seg_cv
                agg_timestamp=self.client.root_timestamp(self.root_id),
            )
        if not self._tube_layers_added:
            base = self._tube.serve()
            T.add_tube_layers(self.viewer, base, self._tube.em_name, self._tube.tgt_name)
            self._tube_layers_added = True

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
    # pre-render -> review -> act (the default Phase A loop)
    # ------------------------------------------------------------------ #
    def _branch_render_dir(self, path_id: int) -> str:
        return os.path.join(self.render_dir, f"path_{int(path_id)}")

    def _progress_printer(self, path_id: int):
        def cb(done: int, total: int, captured: bool) -> None:
            miss = "" if captured else "  (no capture!)"
            end = "\n" if done == total else "\r"
            print(f"  path {path_id}: rendered {done}/{total}{miss}", end=end, flush=True)

        return cb

    def prerender_branch(
        self,
        path_id: int,
        *,
        size=(900, 900),
        render_layout: Optional[str] = "xy",
        load_timeout: Optional[float] = None,
        settle: float = 0.0,
        show_path: bool = True,
        force: bool = False,
        on_progress="print",
    ) -> dict:
        """Capture one fully-loaded frame per node along a branch (the slow batch step).

        Drives the live viewer node-by-node, waiting for each frame to fully render (so
        the segmentation is painted) before screenshotting. ``render_layout="xy"`` renders
        a single 2D cross-section so the load-gate only waits on the fast EM+seg tiles, not
        the slow 3D mesh; pass ``None`` to capture the current layout as-is. Re-running is a
        no-op unless ``force`` (already-rendered branches are skipped).

        Requires the viewer open in a browser. Returns the manifest dict.
        """
        verts_nm, pos_nm, pos_vox, quats = self._compute_path(path_id)
        nav = [(pos_vox[i], None if quats is None else quats[i]) for i in range(len(pos_vox))]
        meta = [
            {"node_index": int(i), "xyz_nm": [float(c) for c in pos_nm[i]]}
            for i in range(len(pos_nm))
        ]
        if show_path:
            V.show_branch_path(self.viewer, verts_nm / self.res)

        prev_layout = None
        if render_layout is not None:
            prev_layout = copy.deepcopy(self.viewer.state.layout)
            with self.viewer.txn() as s:
                s.layout = render_layout

        cb = self._progress_printer(path_id) if on_progress == "print" else on_progress
        try:
            return R.render_states(
                self.viewer,
                nav,
                self._branch_render_dir(path_id),
                size=size,
                load_timeout=self.load_timeout if load_timeout is None else load_timeout,
                settle=settle,
                meta=meta,
                extra_manifest={
                    "root_id": self.root_id,
                    "path_id": int(path_id),
                    "orient_to_path": bool(self.orient_to_path),
                    "step_nm": self.step_nm,
                },
                on_progress=cb,
                force=force,
            )
        finally:
            if prev_layout is not None:
                with self.viewer.txn() as s:
                    s.layout = prev_layout

    def prerender_all(self, **kw) -> dict:
        """Pre-render every branch still ``to_review`` (resumable: skips done ones)."""
        todo = self.coverage.to_review(self.tree)
        out = {}
        for k, pid in enumerate(todo):
            print(f"=== prerender path {pid} ({k + 1}/{len(todo)}) ===")
            out[pid] = self.prerender_branch(pid, **kw)
        return out

    def review_branch(
        self,
        path_id: int,
        *,
        fps: int = 10,
        max_width: Optional[int] = 700,
        render_if_missing: bool = True,
        **render_kw,
    ) -> BranchPlayer:
        """Show the smooth frame player for a branch; "→ viewer" jumps the live viewer.

        Renders the branch first if needed (``render_if_missing``). The player scrubs the
        pre-rendered frames (seg visible in every frame); clicking "→ viewer" calls
        :meth:`goto_node` so you can drop the annotation at that exact node.
        """
        out_dir = self._branch_render_dir(path_id)
        try:
            manifest = R.load_branch_frames(out_dir)
        except FileNotFoundError:
            if not render_if_missing:
                raise
            manifest = self.prerender_branch(path_id, **render_kw)
        self.current_path_id = int(path_id)
        player = BranchPlayer(
            manifest,
            on_goto=lambda ni, xyz, pid=int(path_id): self.goto_node(pid, ni, xyz),
            fps=fps,
            max_width=max_width,
        )
        player.show()
        return player

    def goto_node(self, path_id: int, node_index: int, xyz_nm=None) -> int:
        """Jump the **live** viewer to a node on a branch (so you can annotate there)."""
        verts_nm, pos_nm, pos_vox, quats = self._compute_path(path_id)
        self.current_path_id = int(path_id)
        self._positions_nm = pos_nm
        self._fly_pos_vox = pos_vox
        self._fly_quats = quats
        i = max(0, min(len(pos_vox) - 1, int(node_index)))
        V.show_branch_path(self.viewer, verts_nm / self.res)
        ori = None if quats is None else quats[i]
        self.viewer.set_state(build_target_state(self.viewer, pos_vox[i], ori))
        return i

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

    def diagnostics(self) -> dict:
        """Print signals for debugging the glide perf. Run it after it starts to slow.

        ``tube_wired`` must be True -- if False, this ``sess`` predates the current code
        (autoreload can't add new __init__ attributes): **restart the kernel and recreate
        sess**. ``flythrough_threads`` should be 0-1; more means workers are leaking.
        """
        import threading

        def _dir_mb(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return round(total / 1e6, 1)

        d = {
            "tube_wired": hasattr(self, "_tube"),
            "tube_built": getattr(self, "_tube", None) is not None,
            "tube_layers_added": getattr(self, "_tube_layers_added", None),
            "flythrough_threads": sum(t.name == "flythrough" for t in threading.enumerate()),
            "total_threads": threading.active_count(),
            "fly_playing": self.fly.is_playing if self.fly else None,
            "current_path": self.current_path_id,
            "tube_cache_MB": _dir_mb(self.tube_cache_dir),
        }
        for k, v in d.items():
            print(f"  {k}: {v}")
        return d

    def close(self) -> None:
        if self.fly is not None:
            self.fly.stop()
        self.wal.close()

    def panel(self):
        """An ipywidgets control panel (checklist + review/resolve buttons).

        Has a single controls area that rebuilds whenever ``self.fly`` changes -- driven by
        the ``_on_fly_change`` hook -- so Review, **Mark done**, and the ``x`` key all keep
        the FlyThrough controls + checklist in sync with the current branch (no stale or
        stacked panels).
        """
        from ipywidgets import Button, Dropdown, HBox, Label, Output, VBox
        from ..controls import FlyThroughControls

        status = Label(value=self._status_text())
        todo = self.coverage.to_review(self.tree)
        dd = Dropdown(options=todo, description="path", value=todo[0] if todo else None)
        b_review = Button(description="Review", button_style="primary")
        b_done = Button(description="Mark done (x)")
        b_resolve = Button(description="Resolve supervoxels")
        controls_box = VBox([])  # holds the current branch's controls; children swapped (never stacks)
        log_out = Output()       # build logs / messages

        def refresh_status():
            status.value = self._status_text()
            t = self.coverage.to_review(self.tree)
            dd.options = t
            if self.current_path_id in t:
                dd.value = self.current_path_id
            elif t:
                dd.value = t[0]

        def rebuild_controls():  # fired by review_path via _on_fly_change
            if self.fly is not None:
                self._controls = FlyThroughControls(self.fly, auto_display=False)
                controls_box.children = (self._controls.panel,)  # replace, not append
            else:
                controls_box.children = ()
            refresh_status()

        self._on_fly_change = rebuild_controls

        def on_review(_b):
            with log_out:
                log_out.clear_output()
                if dd.value is not None:
                    self.review_path(dd.value)  # fires _on_fly_change -> rebuild_controls

        def on_done(_b):
            with log_out:
                log_out.clear_output()
                self._on_mark_done()  # -> review_next -> review_path -> rebuild_controls

        def on_resolve(_b):
            with log_out:
                print("resolved", self.resolve_supervoxels(), "supervoxels")

        b_review.on_click(on_review)
        b_done.on_click(on_done)
        b_resolve.on_click(on_resolve)
        # return (don't display()) so Jupyter renders it exactly once as the cell result
        return VBox([HBox([dd, b_review, b_done, b_resolve]), status, controls_box, log_out])

    def _status_text(self) -> str:
        s = self.summary()
        return (f"root {self.root_id} | seed {self.seed} | "
                f"to_review {s['to_review']}  covered {s['covered']}  omitted {s['omitted']}")
