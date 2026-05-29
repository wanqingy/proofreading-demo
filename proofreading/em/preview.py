"""Per-branch precomputed fly-through preview, served as **local** neuroglancer layers.

The bottleneck: neuroglancer paints the live graphene segmentation only when the camera is
idle, and the live EM blurs during motion (streaming latency). The fix: precompute a small
EM (+ target mask) volume for the branch and serve it via :class:`neuroglancer.LocalVolume`
-- a *local* layer that is already in memory, so it renders **sharply during motion**. It
sits at the same world (nm) coordinates as the live ``img``/``seg`` layers, so pausing and
flipping to the live layers (see :func:`viewer.set_preview_mode`) lands at the exact spot.

The preview is a coarse mip (default ~96 nm) over the branch's bounding box -- a thin tube
for a single ``BranchPath``, so a handful of MB. Built lazily per branch, discarded on
advance. Use :func:`localvolume_spike` to validate the core assumption (local layer renders
in motion) before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import neuroglancer
import numpy as np

from .viewer import PREVIEW_EM, PREVIEW_TGT


@dataclass
class Preview:
    """A built per-branch preview: two local image volumes + metadata."""

    em_lv: "neuroglancer.LocalVolume"
    mask_lv: "neuroglancer.LocalVolume"
    mip: int
    resolution: np.ndarray  # nm/voxel of the preview grid
    shape: Tuple[int, int, int]
    voxel_offset: np.ndarray
    nbytes: int
    root_id: int


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def branch_bbox_nm(points_nm: np.ndarray, pad_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """World-space (nm) min/max around a path, padded by ``pad_nm`` on every side."""
    pts = np.atleast_2d(np.asarray(points_nm, dtype=float))
    return pts.min(axis=0) - pad_nm, pts.max(axis=0) + pad_nm


def _choose_image_mip(emclient, bbox_min, bbox_max, target_nm, max_voxels):
    """Image mip nearest ``target_nm``, coarsened further if the bbox exceeds ``max_voxels``."""
    cv0 = emclient.image_cloudvolume(0)
    res_of = lambda m: np.asarray(cv0.mip_resolution(m), dtype=float)
    start = emclient.mip_near(cv0, target_nm)
    start_x = res_of(start)[0]
    # candidate mips from the target resolution and coarser, sorted fine -> coarse
    order = sorted(
        [m for m in cv0.available_mips if res_of(m)[0] >= start_x],
        key=lambda m: res_of(m)[0],
    )
    for m in order:
        res = res_of(m)
        n = float(np.prod(np.ceil((np.asarray(bbox_max) - np.asarray(bbox_min)) / res) + 1))
        if n <= max_voxels:
            return m, res
    m = order[-1] if order else start
    return m, res_of(m)


def _cutout(cv, bbox_min_nm, bbox_max_nm, res):
    """Download the world-bbox at ``cv``'s current mip; returns ``(array_xyz, voxel_offset)``.

    ``array`` is ``None`` if the bbox falls entirely outside the volume bounds.
    """
    res = np.asarray(res, dtype=float)
    v0 = np.floor(np.asarray(bbox_min_nm) / res).astype(np.int64)
    v1 = np.ceil(np.asarray(bbox_max_nm) / res).astype(np.int64) + 1
    b = cv.bounds
    bmin = np.asarray(b.minpt, dtype=np.int64)
    bmax = np.asarray(b.maxpt, dtype=np.int64)
    v0 = np.clip(v0, bmin, bmax)
    v1 = np.clip(v1, bmin, bmax)
    if np.any(v1 <= v0):
        return None, v0
    cube = cv[v0[0]:v1[0], v0[1]:v1[1], v0[2]:v1[2]]
    arr = np.asarray(cube)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr, v0


def _rasterize_path_mask(points_nm, em_v0, em_res, em_shape, radius_nm):
    """Paint a tube (ellipsoids of radius ``radius_nm``) along the path into the EM grid.

    A network-free target highlight: marks voxels within ``radius_nm`` of the skeleton
    centerline. Cheap (a few hundred points x a small ball each). Returns a uint8 mask
    (255 inside the tube), anisotropy handled via per-axis voxel radii.
    """
    res = np.asarray(em_res, dtype=float)
    mask = np.zeros(em_shape, dtype=np.uint8)
    pv = np.asarray(points_nm, dtype=float) / res - np.asarray(em_v0, dtype=float)  # -> em voxels
    r = np.maximum(radius_nm / res, 1e-6)  # radius in voxels, per axis
    rr = np.ceil(r).astype(int)
    X, Y, Z = em_shape
    for cx, cy, cz in pv:
        x0, x1 = max(0, int(cx - rr[0])), min(X, int(cx + rr[0]) + 1)
        y0, y1 = max(0, int(cy - rr[1])), min(Y, int(cy + rr[1]) + 1)
        z0, z1 = max(0, int(cz - rr[2])), min(Z, int(cz + rr[2]) + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        xs = (np.arange(x0, x1) - cx) / r[0]
        ys = (np.arange(y0, y1) - cy) / r[1]
        zs = (np.arange(z0, z1) - cz) / r[2]
        d2 = xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2
        sub = mask[x0:x1, y0:y1, z0:z1]
        sub[d2 <= 1.0] = 255
    return mask


def _gather_mask(seg_arr, seg_v0, seg_res, em_v0, em_res, em_shape, root_id):
    """Resample ``seg_arr == root_id`` onto the EM voxel grid (nearest, via world-nm centers)."""
    if seg_arr is None:
        return np.zeros(em_shape, dtype=np.uint8)
    seg_mask = seg_arr == np.uint64(int(root_id))
    seg_res = np.asarray(seg_res, dtype=float)
    em_res = np.asarray(em_res, dtype=float)
    idx = []
    for k in range(3):
        em_world = (em_v0[k] + np.arange(em_shape[k]) + 0.5) * em_res[k]
        sk = np.round(em_world / seg_res[k] - 0.5 - seg_v0[k]).astype(np.int64)
        idx.append(np.clip(sk, 0, seg_mask.shape[k] - 1))
    m = seg_mask[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    return (m.astype(np.uint8) * 255)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_preview(
    emclient,
    points_nm: np.ndarray,
    root_id: int,
    target_nm: float = 256.0,
    pad_nm: float = 1500.0,
    max_voxels: int = 25_000_000,
    mask_from: str = "skeleton",
    radius_nm: float = 1000.0,
    verbose: bool = True,
) -> Preview:
    """Build the local EM + target-mask volumes for a branch path.

    ``points_nm`` is the (resampled) branch path in nm. Pulls a coarse EM cutout over the
    padded bbox and builds a target overlay, wrapping both as
    :class:`neuroglancer.LocalVolume`s placed at world coordinates (so they register with
    the live layers).

    ``mask_from``:
      - ``"skeleton"`` (default): rasterize a ``radius_nm`` tube along the path -- **no
        network**, robust, fast. The centerline highlight, not the exact seg boundary.
      - ``"segmentation"``: download the agglomerated seg and use ``== root_id`` (the true
        boundary, but a second, often slow, cutout that can dwarf the EM on long branches).

    Cost scales with the bbox volume, so the EM mip is chosen coarse enough to stay under
    ``max_voxels`` (long branch -> coarser preview). The glide only needs to follow the
    neurite; pause to the live layers for boundary detail.
    """
    import time as _time

    bbox_min, bbox_max = branch_bbox_nm(points_nm, pad_nm)

    em_mip, em_res = _choose_image_mip(emclient, bbox_min, bbox_max, target_nm, max_voxels)
    img_cv = emclient.image_cloudvolume(em_mip)
    t0 = _time.time()
    em, em_v0 = _cutout(img_cv, bbox_min, bbox_max, em_res)
    t_em = _time.time() - t0
    if em is None:
        raise ValueError("branch bbox is outside the image volume bounds")
    em = np.ascontiguousarray(em.astype(np.uint8))
    em_shape = em.shape

    t0 = _time.time()
    if mask_from == "segmentation":
        seg_mip = emclient.mip_near(emclient.agg_seg_cv(0), float(em_res[0]))
        seg_cv = emclient.agg_seg_cv(seg_mip)
        seg_res = np.asarray(seg_cv.mip_resolution(seg_mip), dtype=float)
        seg, seg_v0 = _cutout(seg_cv, bbox_min, bbox_max, seg_res)
        mask = _gather_mask(seg, seg_v0, seg_res, em_v0, em_res, em_shape, root_id)
    else:  # "skeleton": network-free tube along the centerline
        mask = _rasterize_path_mask(points_nm, em_v0, em_res, em_shape, radius_nm)
    mask = np.ascontiguousarray(mask)
    t_mask = _time.time() - t0
    if verbose:
        mb = (em.nbytes + mask.nbytes) / 1e6
        print(
            f"preview: mip{em_mip} {em_res.astype(int).tolist()}nm  shape {em_shape}  "
            f"{mb:.0f} MB   EM {t_em:.1f}s   mask[{mask_from}] {t_mask:.1f}s",
            flush=True,
        )

    dims = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[float(x) for x in em_res]
    )
    offset = [int(x) for x in em_v0]
    em_lv = neuroglancer.LocalVolume(data=em, dimensions=dims, voxel_offset=offset,
                                     volume_type="image")
    mask_lv = neuroglancer.LocalVolume(data=mask, dimensions=dims, voxel_offset=offset,
                                       volume_type="image")
    return Preview(
        em_lv=em_lv, mask_lv=mask_lv, mip=int(em_mip), resolution=em_res,
        shape=tuple(int(x) for x in em_shape), voxel_offset=np.asarray(em_v0),
        nbytes=int(em.nbytes + mask.nbytes), root_id=int(root_id),
    )


# --------------------------------------------------------------------------- #
# attach / detach layers
# --------------------------------------------------------------------------- #
def _tint_shader(color=(1.0, 0.25, 0.25), alpha: float = 0.55) -> str:
    r, g, b = color
    return (
        "void main() {\n"
        "  float v = toNormalized(getDataValue());\n"
        f"  emitRGBA(vec4({r:.3f}, {g:.3f}, {b:.3f}, v > 0.5 ? {alpha:.3f} : 0.0));\n"
        "}\n"
    )


def add_preview_layers(
    viewer: neuroglancer.Viewer,
    preview: Preview,
    color=(1.0, 0.25, 0.25),
    alpha: float = 0.55,
    visible: bool = False,
) -> None:
    """Add the EM preview + tinted target overlay layers (hidden by default)."""
    with viewer.txn() as s:
        s.layers[PREVIEW_EM] = neuroglancer.ImageLayer(source=preview.em_lv)
        s.layers[PREVIEW_TGT] = neuroglancer.ImageLayer(
            source=preview.mask_lv, shader=_tint_shader(color, alpha)
        )
        s.layers[PREVIEW_EM].visible = visible
        s.layers[PREVIEW_TGT].visible = visible


def remove_preview_layers(viewer: neuroglancer.Viewer) -> None:
    """Remove the preview layers (e.g. before building the next branch's)."""
    with viewer.txn() as s:
        for name in (PREVIEW_EM, PREVIEW_TGT):
            if name in s.layers:
                del s.layers[name]


# --------------------------------------------------------------------------- #
# validation spike (no CloudVolume / auth needed)
# --------------------------------------------------------------------------- #
def localvolume_spike(ip: str = "localhost", port: int = 0, n: int = 256):
    """Confirm the core assumption: a ``LocalVolume`` image layer renders sharp in motion.

    Builds a synthetic high-frequency checker volume as a single local image layer and a
    :class:`~proofreading.flythrough.FlyThrough` straight through it. Open the returned
    viewer, ``fly.play()``, and watch: if the checker stays **crisp while moving** (doesn't
    blur or blank), the precomputed-preview plan holds. No network / token required.
    """
    from ..flythrough import FlyThrough

    ax = np.arange(n)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    data = np.ascontiguousarray((((gx // 4 + gy // 4 + gz // 4) % 2) * 255).astype(np.uint8))

    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()
    dims = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[8, 8, 40])
    lv = neuroglancer.LocalVolume(data=data, dimensions=dims, volume_type="image")
    with viewer.txn() as s:
        s.dimensions = dims
        s.layers["spike"] = neuroglancer.ImageLayer(source=lv)
        s.layout = "xy"
    path = np.column_stack(
        [np.full(n, n // 2), np.full(n, n // 2), np.arange(n)]
    ).astype(float)
    fly = FlyThrough(viewer, path, None, seconds_per_step=0.05, dwell_seconds=0.0)
    fly.start()
    return viewer, fly
