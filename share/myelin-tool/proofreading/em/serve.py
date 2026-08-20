"""Launch the EM proofreading backend.

    uv run --extra em --extra serve python -m proofreading.em.serve

Binds to 127.0.0.1 only (single-user, localhost). Override via env:
    PROOFREAD_WAL_DIR   default <repo>/proofread_sessions
    PROOFREAD_DATASTACK default minnie65_public  (read-only dev sandbox)
    PROOFREAD_HOST      default 127.0.0.1
    PROOFREAD_PORT      default 8000
    PROOFREAD_WEB_DIST  default <repo>/web/dist -- if a prebuilt UI exists there, it's served
                        from this same origin (no Vite dev server needed); if not, this repo's
                        normal `cd web && npm run dev` loop is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .api import create_app

# repo root = .../proofreading/em/serve.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
WAL_DIR = os.environ.get("PROOFREAD_WAL_DIR", str(_REPO_ROOT / "proofread_sessions"))
DATASTACK = os.environ.get("PROOFREAD_DATASTACK", "minnie65_public")
HOST = os.environ.get("PROOFREAD_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROOFREAD_PORT", "8000"))
WEB_DIST = os.environ.get("PROOFREAD_WEB_DIST", str(_REPO_ROOT / "web" / "dist"))

app = create_app(WAL_DIR, default_datastack=DATASTACK, web_dist=WEB_DIST)


def main() -> None:
    print(f"proofreading-em backend  wal_dir={WAL_DIR}  datastack={DATASTACK}")
    if os.path.isdir(WEB_DIST):
        print(f"serving UI from {WEB_DIST}  ->  http://{HOST}:{PORT}/")
    else:
        print(f"no prebuilt UI at {WEB_DIST} -- run the Vite dev server separately (see README)")
    print(f"listening on http://{HOST}:{PORT}  (tube cache served at /tube)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
