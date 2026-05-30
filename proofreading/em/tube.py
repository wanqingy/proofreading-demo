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
from concurrent.futures import ThreadPoolExecutor
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


def _fill_chunks(local, src, points_nm, radius_nm, transform, workers) -> tuple:
    """Copy the tube's chunks from ``src`` into ``local`` (parallel). Returns (bytes, fails, n)."""
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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(work, boxes))
    return sum(s for s in sizes if s > 0), sum(1 for s in sizes if s < 0), len(boxes)


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

    def __init__(self, emclient, root_id, mip, radius_nm, cache_dir):
        self.root = int(root_id)
        self.mip = int(mip)
        self.radius_nm = float(radius_nm)
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.em_local, self.em_src = _open_shared(
            emclient.image_cloudvolume(self.mip), self.cache_dir, self.em_name
        )
        self.tgt_local, self.tgt_src = _open_shared(
            emclient.agg_seg_cv(self.mip), self.cache_dir, self.tgt_name
        )
        self._markers = os.path.join(self.cache_dir, "_branches")
        os.makedirs(self._markers, exist_ok=True)

    def fill_branch(self, path_id, points_nm, *, workers: int = 16, force: bool = False,
                    verbose: bool = True) -> None:
        """Write this branch's tube chunks into the shared volumes (skips if already done)."""
        marker = os.path.join(self._markers, f"path_{int(path_id)}.done")
        if not force and os.path.exists(marker):
            if verbose:
                print(f"tube path_{int(path_id)}: cached", flush=True)
            return
        rs = P.resample_path(np.asarray(points_nm, dtype=float), 256.0)
        root = self.root
        tint = lambda d: ((np.asarray(d) == np.uint64(root)) * 255).astype(np.uint8)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:  # EM + target concurrently
            f_em = ex.submit(_fill_chunks, self.em_local, self.em_src, rs, self.radius_nm, None, workers)
            f_tg = ex.submit(_fill_chunks, self.tgt_local, self.tgt_src, rs, self.radius_nm, tint, workers)
            (em_b, em_f, em_n), (tg_b, tg_f, tg_n) = f_em.result(), f_tg.result()
        dt = time.time() - t0
        if em_f == 0 and tg_f == 0:  # mark complete so revisiting this branch skips
            try:
                with open(marker, "w") as fh:
                    fh.write(str(time.time()))
            except Exception:
                pass
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
