"""Phase 0 spike — data side (NOT in the animation loop).

Does two things, then blocks:

1. Dumps the branch camera path(s) for a cell to ``web/public/camera_path.json`` — the
   resampled skeleton points (nm) plus the served ``precomputed://`` source URLs.
2. Serves the on-disk tube cache (the shared ``em`` + ``tgt`` precomputed volumes built by
   the notebook workflow) over CORS so the browser can stream chunks.

The browser frontend (web/src/main.ts) then embeds neuroglancer and animates the camera
entirely client-side over this static path + served data. Python never touches the camera.

Run from the repo root:

    uv run --extra em python web/spike_export.py

Leave it running while you use the Vite dev server (``cd web && npm run dev``).
"""

from __future__ import annotations

import json
import os
import sys
import threading

import numpy as np

# repo root (parent of web/) so `proofreading` imports regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proofreading.em as em
from proofreading.em import path as P
from proofreading.em.skeleton_tree import SkeletonTree
from proofreading.em.tube import serve_dir

# --- what to export ---
# ONE branch over its own small per-branch tube (path_<id>_mip<mip>_{em,tgt}). A small
# working set is the whole point of pre-caching: the browser can load the entire branch
# into neuroglancer's cache up front, so the smooth glide replays cached chunks and stays
# sharp during motion (neuroglancer only finishes loading chunks while navigation is idle).
DATASTACK = "minnie65_public"
ROOT_ID = 864691135572530981
PATH_ID = 5                    # per-branch tube present at proofread_sessions/.../path_5_mip1_{em,tgt}
MIP = 1
STEP_NM = 500.0                # camera node spacing (denser = smoother interpolation)
PORT = 0                       # 0 = auto-assign a free port (URL is written into the JSON)
NAME = f"path_{PATH_ID}_mip{MIP}"  # per-branch volume prefix -> NAME_em / NAME_tgt

HERE = os.path.dirname(os.path.abspath(__file__))
WAL_DIR = os.path.join(HERE, "..", "proofread_sessions")
PUBLIC = os.path.join(HERE, "public")


def main() -> None:
    client = em.EMClient(DATASTACK)
    print(f"datastack {client.datastack} | materialization {client.mat_version}")

    sk = client.get_skeleton(int(ROOT_ID))
    tree = SkeletonTree.from_skeleton_dict(sk)

    if PATH_ID >= len(tree.branch_paths):
        raise SystemExit(f"path {PATH_ID} out of range (have {len(tree.branch_paths)})")
    bp = tree.branch_paths[PATH_ID]
    verts_nm = np.asarray(tree.vertices[bp.vertices], dtype=float)
    rs = P.resample_path(verts_nm, STEP_NM)
    pts = rs.tolist()
    print(f"  path {PATH_ID}: {len(bp.vertices)} verts -> {len(rs)} camera nodes")

    if len(pts) < 2:
        raise SystemExit("no camera points produced; check PATH_ID")

    cache_dir = os.path.join(WAL_DIR, "tube_cache", client.datastack, str(int(ROOT_ID)))
    if not os.path.isdir(os.path.join(cache_dir, NAME + "_em")):
        raise SystemExit(
            f"per-branch tube not found at {cache_dir}/{NAME}_em\n"
            f"build it first (e.g. em.tube_prototype(client, {ROOT_ID}, {PATH_ID}, mip={MIP}))."
        )

    base = serve_dir(cache_dir, port=PORT)  # CORS static server, daemon thread
    payload = {
        "datastack": client.datastack,
        "root_id": int(ROOT_ID),
        "paths": [PATH_ID],
        "resolution_nm": [int(v) for v in np.asarray(client.image_cloudvolume(MIP).resolution)],
        "base_url": base,
        "em_source": f"precomputed://{base}/{NAME}_em",
        "tgt_source": f"precomputed://{base}/{NAME}_tgt",
        "step_nm": STEP_NM,
        "points_nm": pts,
    }
    os.makedirs(PUBLIC, exist_ok=True)
    out = os.path.join(PUBLIC, "camera_path.json")
    with open(out, "w") as fh:
        json.dump(payload, fh)

    arc = float(np.sum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1)))
    print(f"\nwrote {out}")
    print(f"  {len(pts)} camera nodes, {arc / 1000:.1f} µm total (single branch {PATH_ID})")
    print(f"serving tube cache at {base}/  ({NAME}_em, {NAME}_tgt)")
    print("\nnow in another shell:  cd web && npm run dev   then open http://localhost:5173")
    print("Ctrl-C to stop serving.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
