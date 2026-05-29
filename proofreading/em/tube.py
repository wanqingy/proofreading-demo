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
    verbose: bool = True,
) -> dict:
    """Download only the tube's chunks from CloudVolume ``src`` into a local precomputed vol.

    Mirrors ``src``'s scale (so it registers in world nm with the live layers), writes a
    single scale and only the tube chunks, **in parallel** (chunk fetches are latency-bound).
    ``transform(data) -> data`` lets a segmentation source become a uint8 target mask
    (``== root_id``). Returns metadata incl. ``cloudpath`` / ``mb`` / ``seconds``.
    """
    from cloudvolume import CloudVolume

    res = np.asarray(src.resolution, dtype=np.int64)
    bounds = src.bounds
    off = np.asarray(bounds.minpt, dtype=np.int64)
    vol_size = np.asarray(bounds.size3(), dtype=np.int64)
    chunk = [64, 64, 64]

    cloudpath = "file://" + os.path.join(os.path.abspath(cache_dir), name)
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
    if verbose:
        msg = f"tube[{name}]: {len(boxes) - fails}/{len(boxes)} chunks  {nbytes / 1e6:.0f} MB  {dt:.1f}s"
        print(msg + (f"  ({fails} failed)" if fails else ""), flush=True)
    return {
        "cloudpath": cloudpath, "name": name, "resolution": res, "voxel_offset": off,
        "n_chunks": len(boxes) - fails, "mb": nbytes / 1e6, "seconds": dt,
    }


def build_local_tube(emclient, points_nm, mip, radius_nm, cache_dir, name, **kw) -> dict:
    """EM convenience wrapper around :func:`build_local_volume` (back-compat)."""
    return build_local_volume(
        emclient.image_cloudvolume(int(mip)), points_nm, radius_nm, cache_dir, name, **kw
    )


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
    root = int(root_id)

    # EM tube + the target (agglomerated seg == root) tube, over the same chunks
    em_meta = build_local_volume(emclient.image_cloudvolume(int(mip)), verts_nm, radius_nm,
                                 cache_dir, name + "_em")
    tgt_meta = build_local_volume(
        emclient.agg_seg_cv(int(mip)), verts_nm, radius_nm, cache_dir, name + "_tgt",
        transform=lambda d: ((np.asarray(d) == np.uint64(root)) * 255).astype(np.uint8),
    )
    base = serve_dir(cache_dir)

    with viewer.txn() as s:
        s.layers[V.PREVIEW_EM] = neuroglancer.ImageLayer(
            source=f"precomputed://{base}/{name}_em"
        )
        s.layers[V.PREVIEW_TGT] = neuroglancer.ImageLayer(
            source=f"precomputed://{base}/{name}_tgt", shader=_TINT
        )
        s.layers[V.PREVIEW_EM].visible = False
        s.layers[V.PREVIEW_TGT].visible = False
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
    print(f"tube served at {base}: EM {em_meta['mb']:.0f} MB + target {tgt_meta['mb']:.0f} MB")
    return viewer, fly
