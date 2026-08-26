#!/usr/bin/env python3
"""Regenerate this package from the parent repo.

    uv run python share/myelin-tool/sync.py            # copy + regenerate everything
    uv run python share/myelin-tool/sync.py --check     # report drift, change nothing

This directory is a trimmed, committed COPY of the myelin-tagging tool -- the backend files
under ``proofreading/em/``, the myelin-only frontend source, and two files that are generated
rather than copied verbatim (see GENERATED below). A committed copy drifts the moment you keep
editing the originals, so the manifest lives here in code instead of in a person's memory:
running this script IS "what's in the minimal package", and ``--check`` is what to run before
handing someone the tarball / pointing them at this dir, to confirm it isn't stale.

Deliberately NOT copied, ever -- checked by :func:`_dest_for`, not just by omission from the
manifest below, so a careless future edit to this file can't reintroduce them:
``proofread_sessions/`` (the WAL logs are the irreplaceable annotation record), any ``*.jsonl``,
``.venv/``, ``node_modules/``, ``.DS_Store``, and anything under ``~/.cloudvolume`` (your CAVE
token). Same posture as ``proofreading/em/clear_cache.py``'s refusal to touch WAL files.

This script has no dependencies beyond the standard library -- it must run before ``uv sync``,
not after.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent       # share/myelin-tool
REPO_ROOT = PKG_ROOT.parent.parent                # the parent repo

_FORBIDDEN = re.compile(r"(^|/)(proofread_sessions|node_modules|\.venv|\.DS_Store)(/|$)|\.jsonl$")

# ------------------------------------------------------------------------- #
# manifest: verbatim copies
# ------------------------------------------------------------------------- #
# The backend as measured against what web/src/myelin.ts + flykernel.ts actually call (see
# share/myelin-tool/README.md's endpoint list): every /api/cells/... route, the warm queue, and
# clear_cache. NOT included: annotator.py/preview.py/render.py/review.py/viewer.py (notebook +
# Spelunker-annotate tooling the myelin tool never imports -- proofreading/em/__init__.py's own
# try/except already treats their absence as optional) and nglui (only reached through
# service.py's _spelunker_url, a lazy import that the myelin UI's endpoints never trigger).
BACKEND_FILES = [
    "proofreading/em/__init__.py",
    "proofreading/em/_http_pool.py",
    "proofreading/em/client.py",
    "proofreading/em/coverage.py",
    "proofreading/em/path.py",
    "proofreading/em/skeleton_tree.py",
    "proofreading/em/wal.py",
    "proofreading/em/tube.py",
    "proofreading/em/service.py",
    "proofreading/em/api.py",
    "proofreading/em/serve.py",
    "proofreading/em/warm_queue.py",
    "proofreading/em/warm_cells.py",
    "proofreading/em/clear_cache.py",
]

# myelin.ts imports nothing from main.ts/review.ts/annotate.ts, so the other three tools (and
# their .html entry points) drop out entirely.
FRONTEND_FILES = [
    "web/myelin.html",
    "web/src/myelin.ts",
    "web/src/flykernel.ts",
    "web/src/crashwatch.ts",
    "web/package.json",
    "web/package-lock.json",
    "web/tsconfig.json",
]

COPY_FILES = BACKEND_FILES + FRONTEND_FILES

# ------------------------------------------------------------------------- #
# generated: content derived from, but not identical to, a repo original
# ------------------------------------------------------------------------- #
_INIT_STUB = '''"""Myelin-tagging tool package (trimmed copy -- see share/myelin-tool/sync.py).

The full repo's ``proofreading/__init__.py`` imports the notebook fly-through toolkit
(``.viewer`` / ``.skeleton`` / ``.flythrough`` / ``.context``), which needs ``neuroglancer`` +
``ipywidgets`` + (optionally) ``navis``. This tool never uses that toolkit -- it drives
neuroglancer entirely from the browser (``web/src/myelin.ts``) -- so this stub replaces it
rather than being a copy, keeping this package's Python dependency list to exactly what
``proofreading.em.serve`` needs: numpy, caveclient, cloud-volume, fastapi, uvicorn.
"""
'''

_PYPROJECT = '''[project]
name = "myelin-tool"
version = "0.1.0"
description = "Standalone axon myelination tagging tool (trimmed from proofreading-demo)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.4.4",
    "caveclient>=8.1.0",
    "cloud-volume>=12.13.1",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    # transitive (google-auth -> caveclient/cloud-volume), pinned directly: cryptography 50.x's
    # compiled _rust extension fails to dlopen on at least one verified machine (arm64 macOS) --
    # "symbol not found in flat namespace '_ASN1_GENERALIZEDTIME_free'", an OpenSSL ABI mismatch
    # in that wheel, not anything specific to this tool. 48.0.0 is what the parent repo's own
    # uv.lock happens to pin and is confirmed working; commit this package's own uv.lock (`uv
    # lock`) so a recipient's `uv sync` reproduces that resolution instead of free-resolving to
    # whatever is newest the day they install.
    "cryptography<49",
]
'''


def _make_vite_config(original: str) -> str:
    """Trim the repo's web/vite.config.ts to a myelin-only build.

    Two changes from the original, beyond dropping the other three HTML entry points:
    - Drop ``NEUROGLANCER_BRAINMAPS_CLIENT_ID``: that's the maintainer's personal GCP OAuth
      client, registered against their own dev-server origin. This tool never uses Brainmaps
      OAuth (it authenticates via the backend's CAVE token -- see myelin.ts's
      ``registerMiddleAuthToken``), so shipping that client id would be both useless to a
      recipient (their origin isn't registered against it) and needlessly identify the
      maintainer's GCP project in a package handed to other people.
    - The ``strictPort`` comment explaining that OAuth redirect-URI requirement no longer
      applies once the client id is gone, so it's rewritten rather than copied verbatim.
    """
    out = original
    # rollupOptions.input -> myelin only (indentation matches the surrounding 6-space nesting
    # the original 4-entry block used -- the leading whitespace before "input:" itself is
    # untouched since it's outside this match)
    out = re.sub(
        r"input:\s*\{[^}]*\},",
        'input: {\n        myelin: resolve(__dirname, "myelin.html"),\n      },',
        out, count=1, flags=re.S,
    )
    # drop the maintainer's personal OAuth client id, back to the same `"undefined"` STRING
    # (not bare `undefined`) every sibling entry above it uses -- esbuild's `define` treats
    # string values as raw replacement source text, so a bare `undefined` here would be a type
    # mismatch against the Record<string,string> the other 8 entries all follow.
    out = re.sub(
        r"\s*// Our own OAuth client.*?NEUROGLANCER_BRAINMAPS_CLIENT_ID:\s*JSON\.stringify\([^)]*\),",
        '\n    NEUROGLANCER_BRAINMAPS_CLIENT_ID: "undefined",',
        out, count=1, flags=re.S,
    )
    # the strictPort comment was about registering that OAuth client's redirect URI -- no
    # longer applicable once the client id above is gone
    out = re.sub(
        r"\s*// strictPort: true --.*?redirect_uri_mismatch\.",
        "\n  // strictPort: true -- keep the dev-server port stable across restarts.",
        out, count=1, flags=re.S,
    )
    return out


def _dest_for(rel: str) -> Path:
    if _FORBIDDEN.search(rel):
        raise SystemExit(f"refusing to sync forbidden path: {rel}")
    dest = (PKG_ROOT / rel).resolve()
    if PKG_ROOT.resolve() not in dest.parents:
        raise SystemExit(f"refusing to write outside package root: {dest}")
    return dest


def _plan() -> list[tuple[str, str]]:
    """Return [(dest_rel_path, new_content), ...] for every copied + generated file."""
    plan = []
    for rel in COPY_FILES:
        src = REPO_ROOT / rel
        if not src.is_file():
            raise SystemExit(f"missing source file (moved or renamed upstream?): {rel}")
        plan.append((rel, src.read_text()))

    plan.append(("proofreading/__init__.py", _INIT_STUB))
    plan.append(("pyproject.toml", _PYPROJECT))

    vite_src = REPO_ROOT / "web" / "vite.config.ts"
    plan.append(("web/vite.config.ts", _make_vite_config(vite_src.read_text())))
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report drift; write nothing")
    args = ap.parse_args(argv)

    plan = _plan()
    drift = []
    for rel, content in plan:
        dest = _dest_for(rel)
        current = dest.read_text() if dest.is_file() else None
        if current == content:
            continue
        drift.append(rel)
        status = "new" if current is None else "changed"
        if args.check:
            print(f"  {status:8} {rel}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            print(f"  wrote    {rel}")

    if args.check:
        if drift:
            print(f"\n{len(drift)} file(s) out of date -- run without --check to regenerate")
            return 1
        print("up to date")
        return 0

    if not drift:
        print("already up to date")
    else:
        print(f"\n{len(drift)} file(s) synced")
    print(
        "\nNot handled by this script (do these by hand when needed):\n"
        "  - web/dist/       rebuild with: cd share/myelin-tool/web && npm install && npm run build\n"
        "  - README.md       hand-maintained, no repo equivalent to sync from"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
