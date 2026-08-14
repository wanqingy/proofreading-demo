"""Launch the simplified proofreading-recorder backend.

    uv run --extra serve python -m proofreading.annotate.serve

Binds to 127.0.0.1 only (single-user, localhost). Override via env:
    PROOFREAD_ANNOTATE_LOG  default <repo>/proofread_sessions/annotate_records.jsonl
    PROOFREAD_ANNOTATE_HOST default 127.0.0.1
    PROOFREAD_ANNOTATE_PORT default 8001
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .api import create_app

# repo root = .../proofreading/annotate/serve.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = os.environ.get(
    "PROOFREAD_ANNOTATE_LOG", str(_REPO_ROOT / "proofread_sessions" / "annotate_records.jsonl")
)
HOST = os.environ.get("PROOFREAD_ANNOTATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROOFREAD_ANNOTATE_PORT", "8001"))

app = create_app(LOG_PATH)


def main() -> None:
    print(f"proofreading-annotate backend  log={LOG_PATH}")
    print(f"listening on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
