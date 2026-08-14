"""Sparse-tube mip1 preview: a local precomputed EM cutout that follows the neurite.

A neuron is a thin tree in a huge box: a *dense* mip1 cutout of even one branch is hundreds
of MB to TBs (median 833 MB / up to 5 TB for this cell), but the **tube** around the
skeleton is ~10s of MB (median 14 MB). So we fetch only the chunks the tube passes through,
write them to a **local precomputed dataset** (CloudVolume ``file://``), and serve it to
neuroglancer as a ``precomputed://`` image layer. The browser streams just those chunks
(sharp, low local latency; blank off the neurite) and renders them *during* camera motion,
so the fly-through is smooth at full mip1 detail. Pausing flips to the live ``img``+``seg``
(:func:`viewer.set_preview_mode`) for annotation.

This is the one-branch prototype: ``tube_prototype`` builds + serves a branch's tube and
returns ``(viewer, fly)`` to glide.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FuturesTimeout
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Tuple

import neuroglancer
import numpy as np

from . import path as P
from . import viewer as V
from .skeleton_tree import SkeletonTree
from ..flythrough import FlyThrough

Box = Tuple[np.ndarray, np.ndarray]
_servers: dict = {}  # abs cache dir -> base url (reuse one server per dir)

# red overlay for the target mask precomputed layer (uint8 0/255)
_TINT = (
    "void main() {\n"
    "  float v = toNormalized(getDataValue());\n"
    "  emitRGBA(vec4(1.0, 0.2, 0.2, v > 0.5 ? 0.6 : 0.0));\n"
    "}\n"
)


# --------------------------------------------------------------------------- #
# tube chunk selection
# --------------------------------------------------------------------------- #
def tube_chunks(points_nm, voxel_offset, resolution, chunk_size, radius_nm) -> List[Box]:
    """Chunk-aligned global-voxel boxes covering a ``radius_nm`` tube along the path.

    For each (densely resampled) point, take the ``[p ± radius]`` box in voxels, snap to the
    ``chunk_size`` grid (anchored at ``voxel_offset``), and collect the **unique** chunks --
    so overlapping points along the path don't re-download the same chunk.
    """
    res = np.asarray(resolution, dtype=float)
    cs = np.asarray(chunk_size, dtype=np.int64)
    off = np.asarray(voxel_offset, dtype=np.int64)
    r_vox = np.maximum(np.asarray(radius_nm, dtype=float) / res, 0.0)
    pts_vox = np.asarray(points_nm, dtype=float) / res  # global voxel coords

    seen = set()
    for p in pts_vox:
        lo = np.floor(p - r_vox).astype(np.int64)
        hi = np.ceil(p + r_vox).astype(np.int64)
        c_lo = np.floor((lo - off) / cs).astype(np.int64)
        c_hi = np.floor((hi - off) / cs).astype(np.int64)
        for cx in range(int(c_lo[0]), int(c_hi[0]) + 1):
            for cy in range(int(c_lo[1]), int(c_hi[1]) + 1):
                for cz in range(int(c_lo[2]), int(c_hi[2]) + 1):
                    seen.add((cx, cy, cz))

    boxes: List[Box] = []
    for cx, cy, cz in seen:
        bmin = off + np.array([cx, cy, cz], dtype=np.int64) * cs
        boxes.append((bmin, bmin + cs))
    return boxes


# --------------------------------------------------------------------------- #
# build the local sparse precomputed EM
# --------------------------------------------------------------------------- #
def build_local_volume(
    src,
    points_nm,
    radius_nm: float,
    cache_dir: str,
    name: str,
    *,
    transform=None,
    data_type: str = "uint8",
    resample_nm: float = 256.0,
    workers: int = 16,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Download only the tube's chunks from CloudVolume ``src`` into a local precomputed vol.

    Mirrors ``src``'s scale (so it registers in world nm with the live layers), writes a
    single scale and only the tube chunks, **in parallel** (chunk fetches are latency-bound).
    ``transform(data) -> data`` lets a segmentation source become a uint8 target mask
    (``== root_id``). Returns metadata incl. ``cloudpath`` / ``mb`` / ``seconds``.
    """
    from cloudvolume import CloudVolume

    vol_dir = os.path.join(os.path.abspath(cache_dir), name)
    cloudpath = "file://" + vol_dir
    marker = os.path.join(vol_dir, ".tube_done")
    if not force and os.path.exists(marker):  # already built -> skip (no source access)
        if verbose:
            print(f"tube[{name}]: cached", flush=True)
        return {"cloudpath": cloudpath, "name": name, "cached": True,
                "n_chunks": 0, "mb": 0.0, "seconds": 0.0}

    res = np.asarray(src.resolution, dtype=np.int64)
    bounds = src.bounds
    off = np.asarray(bounds.minpt, dtype=np.int64)
    vol_size = np.asarray(bounds.size3(), dtype=np.int64)
    chunk = [64, 64, 64]

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type=data_type,
        encoding="raw",
        resolution=res.tolist(),
        voxel_offset=off.tolist(),
        volume_size=vol_size.tolist(),
        chunk_size=chunk,
    )
    local = CloudVolume(cloudpath, info=info, compress=False, fill_missing=True, progress=False)
    local.commit_info()

    rs = P.resample_path(np.asarray(points_nm, dtype=float), resample_nm)
    boxes = tube_chunks(rs, off, res, chunk, radius_nm)
    bmax_all = off + vol_size

    def _work(box) -> int:
        lo = np.maximum(box[0], off)
        hi = np.minimum(box[1], bmax_all)
        if np.any(hi <= lo):
            return 0
        try:
            data = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            if transform is not None:
                data = transform(data)
            local[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = data
            return int(data.size)
        except Exception:
            return -1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(_work, boxes))
    dt = time.time() - t0
    nbytes = sum(s for s in sizes if s > 0)
    fails = sum(1 for s in sizes if s < 0)
    if fails == 0:  # mark complete so revisiting this branch skips the rebuild
        try:
            with open(marker, "w") as fh:
                fh.write(str(time.time()))
        except Exception:
            pass
    if verbose:
        msg = f"tube[{name}]: {len(boxes) - fails}/{len(boxes)} chunks  {nbytes / 1e6:.0f} MB  {dt:.1f}s"
        print(msg + (f"  ({fails} failed)" if fails else ""), flush=True)
    return {
        "cloudpath": cloudpath, "name": name, "resolution": res, "voxel_offset": off,
        "n_chunks": len(boxes) - fails, "mb": nbytes / 1e6, "seconds": dt, "cached": False,
    }


def build_local_tube(emclient, points_nm, mip, radius_nm, cache_dir, name, **kw) -> dict:
    """EM convenience wrapper around :func:`build_local_volume` (back-compat)."""
    return build_local_volume(
        emclient.image_cloudvolume(int(mip)), points_nm, radius_nm, cache_dir, name, **kw
    )


def build_branch_tube(
    emclient, points_nm, root_id, mip, radius_nm, cache_dir, name,
    *, workers: int = 16, force: bool = False, verbose: bool = True,
) -> dict:
    """Build the EM tube and the target-mask tube (``agg_seg == root_id``) **concurrently**.

    Returns ``{em_name, tgt_name, em, tgt}``. Each sub-build skips if already complete
    (``.tube_done`` marker), so revisiting a branch is instant.
    """
    em_name, tgt_name = name + "_em", name + "_tgt"
    root = int(root_id)

    def _em():
        return build_local_volume(emclient.image_cloudvolume(int(mip)), points_nm, radius_nm,
                                  cache_dir, em_name, workers=workers, force=force, verbose=verbose)

    def _tgt():
        return build_local_volume(
            emclient.agg_seg_cv(int(mip)), points_nm, radius_nm, cache_dir, tgt_name,
            transform=lambda d: ((np.asarray(d) == np.uint64(root)) * 255).astype(np.uint8),
            workers=workers, force=force, verbose=verbose,
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_em, f_tgt = ex.submit(_em), ex.submit(_tgt)
        em_meta, tgt_meta = f_em.result(), f_tgt.result()
    return {"em_name": em_name, "tgt_name": tgt_name, "em": em_meta, "tgt": tgt_meta}


def add_tube_layers(viewer, base_url: str, em_name: str, tgt_name: str, visible: bool = False) -> None:
    """Add the served EM + red-target precomputed layers (as PREVIEW_EM / PREVIEW_TGT)."""
    with viewer.txn() as s:
        s.layers[V.PREVIEW_EM] = neuroglancer.ImageLayer(source=f"precomputed://{base_url}/{em_name}")
        s.layers[V.PREVIEW_TGT] = neuroglancer.ImageLayer(
            source=f"precomputed://{base_url}/{tgt_name}", shader=_TINT
        )
        s.layers[V.PREVIEW_EM].visible = visible
        s.layers[V.PREVIEW_TGT].visible = visible


# --------------------------------------------------------------------------- #
# shared per-cell tube (ONE local precomputed, filled lazily per branch)
# --------------------------------------------------------------------------- #
def _open_shared(src, cache_dir: str, name: str, data_type: str = "uint8"):
    """Open (or create) a local precomputed volume mirroring ``src``'s scale, covering the
    full source bounds. Only the tube chunks ever get written; the rest stays absent."""
    from cloudvolume import CloudVolume

    vol_dir = os.path.join(os.path.abspath(cache_dir), name)
    cloudpath = "file://" + vol_dir
    if os.path.exists(os.path.join(vol_dir, "info")):
        local = CloudVolume(cloudpath, compress=False, fill_missing=True, progress=False)
    else:
        res = np.asarray(src.resolution, dtype=np.int64)
        b = src.bounds
        info = CloudVolume.create_new_info(
            num_channels=1, layer_type="image", data_type=data_type, encoding="raw",
            resolution=res.tolist(), voxel_offset=np.asarray(b.minpt, dtype=np.int64).tolist(),
            volume_size=np.asarray(b.size3(), dtype=np.int64).tolist(), chunk_size=[64, 64, 64],
        )
        local = CloudVolume(cloudpath, info=info, compress=False, fill_missing=True, progress=False)
        local.commit_info()
    return local, src


def branch_marker_state(markers_dir, path_id, em_mip, tgt_mip) -> tuple:
    """``(em_done, tgt_done)`` for one branch, honoring the legacy single marker.

    Markers are per PHASE and the mask's is keyed by its mip, because the mask resolution is
    configurable (see :attr:`CellTube.DEFAULT_TGT_MIP`): a single ``path_N.done`` cannot express
    "em is built, and the mask is built *at this resolution*". With one shared marker, lowering
    the mask mip made every already-built branch skip its fill entirely, leaving the new mask
    volume empty (all chunk requests 404 -> no overlay at all).

    The legacy ``path_N.done`` predates configurable mask mips, when the mask was always built at
    the EM mip -- so it counts as both phases done only when ``tgt_mip == em_mip``, and otherwise
    only as the em phase (so switching mips refetches just the mask, not the imagery).
    """
    pid = int(path_id)
    legacy = os.path.exists(os.path.join(markers_dir, f"path_{pid}.done"))
    em_done = legacy or os.path.exists(os.path.join(markers_dir, f"path_{pid}.em.done"))
    tgt_done = os.path.exists(os.path.join(markers_dir, f"path_{pid}.tgt{int(tgt_mip)}.done")) or (
        legacy and int(tgt_mip) == int(em_mip)
    )
    return em_done, tgt_done


def _fill_chunks(local, src, points_nm, radius_nm, transform, workers, budget_s,
                 on_progress=None) -> tuple:
    """Copy the tube's chunks from ``src`` into ``local`` (parallel). Returns (bytes, fails, n).

    ``budget_s`` caps the WHOLE copy: the tube data is served from one host (GCS), whose HTTP
    connection pool is small, and CloudVolume reads have no socket timeout -- so a wedged read
    can hang ``ex.map`` forever, holding the branch lock and starving the API (observed on long
    branches). We wait on the reads with an overall deadline and, on timeout, count the still-
    pending boxes as failed and ``shutdown(wait=False)`` so the request RETURNS (the leaked
    worker(s) unwedge on their own; the branch isn't marked ``.done`` -> retried next visit).

    ``on_progress(done, total)`` fires as each chunk lands, so callers can distinguish "slowly
    working through a big branch" from "wedged" -- a distinction the per-branch ``.done`` marker
    alone can't make (it only flips at the very end).
    """
    res = np.asarray(src.resolution, dtype=np.int64)
    b = src.bounds
    off = np.asarray(b.minpt, dtype=np.int64)
    bmax = np.asarray(b.maxpt, dtype=np.int64)
    boxes = tube_chunks(points_nm, off, res, [64, 64, 64], radius_nm)

    def work(box) -> int:
        lo = np.maximum(box[0], off)
        hi = np.minimum(box[1], bmax)
        if np.any(hi <= lo):
            return 0
        try:
            d = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            if transform is not None:
                d = transform(d)
            local[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = d
            return int(d.size)
        except Exception:
            return -1

    ex = ThreadPoolExecutor(max_workers=workers)
    futs = [ex.submit(work, box) for box in boxes]
    nbytes = fails = done = 0
    if on_progress:
        on_progress(0, len(boxes))
    try:
        for f in as_completed(futs, timeout=budget_s):
            done += 1
            s = f.result()  # work() swallows read errors -> -1; never raises
            if s > 0:
                nbytes += s
            elif s < 0:
                fails += 1
            if on_progress:
                on_progress(done, len(boxes))
    except _FuturesTimeout:
        fails += len(futs) - done  # the reads still pending at the deadline (likely wedged)
    finally:
        # wait=False so a wedged GCS read can't re-hang us on shutdown; cancel any not-yet-started
        ex.shutdown(wait=False, cancel_futures=True)
    return nbytes, fails, len(boxes)


class CellTube:
    """One shared local precomputed EM + target volume per cell, filled lazily per branch.

    The fix for the glide slowing down over a session: instead of a *new* precomputed source
    per branch (which churns neuroglancer's chunk cache / data-source registry), there is one
    EM and one target volume covering the whole cell, and a **single, never-swapped** layer.
    Reviewing a branch just writes its tube chunks (if not already present) and moves the
    camera; neuroglancer streams from the one source with a bounded LRU.
    """

    em_name = "em"
    tgt_name = "tgt"

    # The target mask is a translucent "does this voxel belong to the cell" tint, so it does NOT
    # need the EM's resolution -- and fetching it at the EM mip dominates build time. The seg's
    # native chunk is 256x256x32 uint64 (16.8 MB) at mips 1-3 and only shrinks to 128x128x16
    # (2.1 MB) at mip4, which is why mip4 (not mip3) is where the big win is. Neuroglancer
    # registers layers by their declared resolution + voxel_offset, so a coarser mask lines up
    # with the 16nm EM in world space with no resampling on our side.
    #
    # Measured, minnie65_public branch 15 (an AXON -- the thinnest structure, worst case for
    # downsampling). "coverage" = fraction of skeleton centerline vertices the mask marks:
    #   mip1 (16x16x40):   tgt phase 144s, 376 MB   coverage --
    #   mip2 (32x32x40):   tgt phase 101s, 142 MB   coverage 100%
    #   mip3 (64x64x40):   tgt phase  88s,  69 MB   coverage  84%
    #   mip4 (128x128x80): tgt phase  42s,  26 MB   coverage  68%   <- default
    # mip4 is the deliberate speed-first choice: whole-cell caching ~33 min instead of ~86, at
    # the cost of the tint missing roughly a third of a thin axon's centerline. Treat it as an
    # orientation cue, not a reliable cell boundary; raise PROOFREAD_TGT_MIP (2 = full fidelity)
    # for a cell where that matters.
    DEFAULT_TGT_MIP = int(os.environ.get("PROOFREAD_TGT_MIP", "4"))

    def __init__(self, emclient, root_id, mip, radius_nm, cache_dir, tgt_mip=None):
        self.root = int(root_id)
        self.mip = int(mip)
        self.tgt_mip = int(self.DEFAULT_TGT_MIP if tgt_mip is None else tgt_mip)
        self.radius_nm = float(radius_nm)
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.em_local, self.em_src = _open_shared(
            emclient.image_cloudvolume(self.mip), self.cache_dir, self.em_name
        )
        # NOTE: the mask volume is keyed by its mip in the directory name -- changing the mip
        # must not append coarse chunks into a volume whose `info` declares a finer resolution.
        tgt_dir = self.tgt_name if self.tgt_mip == self.mip else f"{self.tgt_name}_mip{self.tgt_mip}"
        self.tgt_name_dir = tgt_dir
        self.tgt_local, self.tgt_src = _open_shared(
            emclient.agg_seg_cv(self.tgt_mip), self.cache_dir, tgt_dir
        )
        self._markers = os.path.join(self.cache_dir, "_branches")
        os.makedirs(self._markers, exist_ok=True)

    def marker_state(self, path_id) -> tuple:
        """``(em_done, tgt_done)`` for this branch at THIS instance's mask mip."""
        return branch_marker_state(self._markers, path_id, self.mip, self.tgt_mip)

    def branch_done(self, path_id) -> bool:
        return all(self.marker_state(path_id))

    def fill_branch(self, path_id, points_nm, *, workers: int = 16, budget_s: float = 360.0,
                    force: bool = False, verbose: bool = True, on_progress=None) -> None:
        """Write this branch's tube chunks into the shared volumes (skips if already done).

        EM and target are filled SEQUENTIALLY (not 2 concurrent executors) to keep the peak
        request count in hand. Concurrent GCS requests per fill is
        ``workers x native-chunks-per-64^3-box``, measured on minnie65_public mip1 as:
        EM native chunk is exactly 64^3 -> **1** per box; agg-seg native chunk is 256x256x32,
        and our boxes are aligned to the same origin, so 64 in z -> **2** per box. So a fill
        uses ``workers`` (em) to ``2 x workers`` (tgt) connections.

        At the old ``workers=2`` that was just 2-4 connections -- far under the 64-connection
        ceiling (:mod:`proofreading.em._http_pool`) and slow enough that a ~1400-chunk branch
        could not finish inside ``budget_s`` at all. 16 gives 16-32, still inside the ceiling
        even with a second fill running concurrently.

        ``budget_s`` bounds each fill (see :func:`_fill_chunks`) so a wedged read can't hang the
        build indefinitely. ``on_progress(phase, done, total)`` reports chunk-level progress
        ("em" then "tgt") for a live caching indicator.
        """
        pid = int(path_id)
        em_done, tgt_done = self.marker_state(pid)
        if not force and em_done and tgt_done:
            if verbose:
                print(f"tube path_{pid}: cached", flush=True)
            return
        rs = P.resample_path(np.asarray(points_nm, dtype=float), 256.0)
        root = self.root
        tint = lambda d: ((np.asarray(d) == np.uint64(root)) * 255).astype(np.uint8)
        t0 = time.time()
        cb = (lambda phase: (lambda d, t: on_progress(phase, d, t))) if on_progress else (lambda _p: None)

        def _mark(name: str) -> None:
            try:
                with open(os.path.join(self._markers, name), "w") as fh:
                    fh.write(str(time.time()))
            except Exception:
                pass

        # Each phase is skipped and marked INDEPENDENTLY, so changing the mask mip refetches only
        # the mask, and a phase that failed last time isn't re-done alongside one that succeeded.
        em_b = em_f = em_n = 0
        if force or not em_done:
            em_b, em_f, em_n = _fill_chunks(self.em_local, self.em_src, rs, self.radius_nm, None,
                                            workers, budget_s, on_progress=cb("em"))
            if em_f == 0:
                _mark(f"path_{pid}.em.done")
        tg_b = tg_f = tg_n = 0
        if force or not tgt_done:
            tg_b, tg_f, tg_n = _fill_chunks(self.tgt_local, self.tgt_src, rs, self.radius_nm, tint,
                                            workers, budget_s, on_progress=cb("tgt"))
            if tg_f == 0:
                _mark(f"path_{pid}.tgt{self.tgt_mip}.done")
        dt = time.time() - t0
        if verbose:
            print(f"tube path_{int(path_id)}: {em_n + tg_n} chunks  "
                  f"{(em_b + tg_b) / 1e6:.0f} MB  {dt:.1f}s"
                  + ("  (some failed)" if (em_f or tg_f) else ""), flush=True)

    def serve(self) -> str:
        return serve_dir(self.cache_dir)


# --------------------------------------------------------------------------- #
# serve the local precomputed dir to the browser (CORS)
# --------------------------------------------------------------------------- #
def serve_dir(cache_dir: str, port: int = 0) -> str:
    """Start (once) a CORS static server rooted at ``cache_dir``; return its base URL.

    neuroglancer fetches ``<base>/<name>/info`` and ``<base>/<name>/<key>/<chunk>`` -- whole
    files, no range requests -- so a plain static server with an ``Access-Control-Allow-Origin``
    header suffices. Reused across tubes in the same dir.
    """
    key = os.path.abspath(cache_dir)
    if key in _servers:
        return _servers[key]

    class _CORS(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):  # quiet
            pass

    handler = partial(_CORS, directory=key)
    httpd = ThreadingHTTPServer(("localhost", port), handler)
    base = f"http://localhost:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True, name="tube-serve").start()
    _servers[key] = base
    return base


# --------------------------------------------------------------------------- #
# one-branch prototype
# --------------------------------------------------------------------------- #
def tube_prototype(
    emclient,
    root_id: int,
    path_id: int,
    *,
    mip: int = 1,
    radius_nm: float = 1000.0,
    wal_dir: str = "./proofread_sessions",
    step_nm: float = 1000.0,
    seconds_per_step: float = 0.4,
    ip: str = "localhost",
    port: int = 0,
):
    """Build + serve one branch's sparse mip1 tube and return ``(viewer, fly)`` to glide.

    Play -> sharp mip1 EM glides along the neurite (local precomputed, renders in motion).
    Pause -> live ``img``+``seg`` paint at that spot (registered in world nm) for annotation.
    """
    sk = emclient.get_skeleton(int(root_id))
    tree = SkeletonTree.from_skeleton_dict(sk)
    bp = tree.branch_paths[int(path_id)]
    verts_nm = tree.vertices[bp.vertices]
    rs = P.resample_path(verts_nm, step_nm)  # camera path (nm)

    viewer, res_nm, _dims = V.make_em_viewer(emclient, int(root_id), ip=ip, port=port)

    cache_dir = os.path.join(wal_dir, "tube_cache", emclient.datastack, str(int(root_id)))
    os.makedirs(cache_dir, exist_ok=True)
    name = f"path_{int(path_id)}_mip{int(mip)}"

    m = build_branch_tube(emclient, verts_nm, root_id, mip, radius_nm, cache_dir, name)
    base = serve_dir(cache_dir)
    add_tube_layers(viewer, base, m["em_name"], m["tgt_name"], visible=False)
    V.set_preview_mode(viewer, False)  # start paused -> live layers shown

    fly = FlyThrough(
        viewer,
        rs / res_nm,
        None,
        seconds_per_step=seconds_per_step,
        dwell_seconds=0.0,
        on_play=lambda: V.set_preview_mode(viewer, True),
        on_pause=lambda: V.set_preview_mode(viewer, False),
    )
    fly.start()
    print(f"tube served at {base}/{name}")
    return viewer, fly
