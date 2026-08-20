"""Queue cells for background tube caching, then watch them build.

    uv run python -m proofreading.em.warm_cells 864691135572530981 864691136335553971
    uv run python -m proofreading.em.warm_cells --status          # what's in the queue now
    uv run python -m proofreading.em.warm_cells --cancel 8646911355725309811
    uv run python -m proofreading.em.warm_cells --no-watch 8646911...   # queue and exit

Cells are cached one at a time in the order given (see :mod:`proofreading.em.warm_queue` for
why serial beats parallel here), so the first cell in the list is reviewable long before the
last one starts. Queue the cell you plan to review first, first.

This talks to the **running backend** rather than fetching anything itself: the server process
holds the per-branch build locks and the tuned connection pool, so a second fetcher would
re-download branches the server was already building. Start the backend first:

    uv run --extra em --extra serve python -m proofreading.em.serve

Ctrl-C only stops watching -- the queue keeps building. Use ``--cancel`` to actually stop a
cell (it finishes the branch in flight, so nothing is left half-cached).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = "http://localhost:8000"
DONE = ("done", "cancelled", "error")


def _req(api: str, path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(
        f"{api.rstrip('/')}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read() or "{}")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"no backend at {api} ({e.reason}).\n"
            "start it with: uv run --extra em --extra serve python -m proofreading.em.serve"
        )


def _line(c: dict) -> str:
    """One cell's progress. `already_built` is counted as progress -- resuming a half-cached
    cell should read as 40/57, not restart at 0/34."""
    total = c["total"]
    if not total:
        # `total` is only known once the cell has been opened, so "no branches" is a claim we
        # can make solely for a cell that finished. Before that it just means "not measured yet".
        if c["status"] == "done":
            # several minnie65 cells are genuinely axon-less; say so rather than drawing a 0/1
            # bar, which reads as "one branch failed"
            what = c["compartment"] or "reviewable"
            return f"  {c['root_id']}  {c['status']:<10} no {what} branches -- nothing to cache"
        note = {"queued": "waiting its turn",
                "opening": "fetching skeleton"}.get(c["status"], c.get("error", ""))
        return f"  {c['root_id']}  {c['status']:<10} {note}"
    have = c["already_built"] + c["built"]
    bar_n = int(20 * have / total)
    bar = "#" * bar_n + "-" * (20 - bar_n)
    extra = ""
    fill = c.get("fill") or {}
    active = fill.get("active") or []
    if active:
        a = active[0]
        extra = f"  branch {a['path_id']} {a['phase']} {a['done']}/{a['total']}"
        if a.get("stalled_s", 0) > 120:
            extra += f"  STALLED {a['stalled_s']:.0f}s"
    if c.get("failed"):
        extra += f"  ({c['failed']} incomplete)"
    if c.get("error"):
        extra += f"  {c['error']}"
    return f"  {c['root_id']}  {c['status']:<10} [{bar}] {have}/{total} branches{extra}"


def _render(status: dict) -> None:
    for c in status["cells"]:
        print(_line(c))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root_ids", nargs="*", help="cells to cache, in the order you'll review them")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--compartment", default="axon",
                    help="'axon' (default) matches the myelin tool; '' = every branch")
    ap.add_argument("--datastack", default=None)
    ap.add_argument("--tgt-mip", type=int, default=None, help="mask mip (default: server's)")
    ap.add_argument("--status", action="store_true", help="show the queue and exit")
    ap.add_argument("--cancel", metavar="ROOT_ID", help="drop a cell from the queue")
    ap.add_argument("--no-watch", action="store_true", help="queue and exit instead of following")
    ap.add_argument("--interval", type=float, default=10.0, help="poll seconds while watching")
    args = ap.parse_args(argv)

    if args.cancel:
        print(json.dumps(_req(args.api, f"/api/warm-queue/{int(args.cancel)}", "DELETE")))
        return 0

    if args.status or not args.root_ids:
        st = _req(args.api, "/api/warm-queue")
        if not st["cells"]:
            print("queue is empty")
            return 0
        _render(st)
        return 0

    body = {
        "root_ids": [int(r) for r in args.root_ids],
        "compartment": args.compartment or None,
        "datastack": args.datastack,
        "tgt_mip": args.tgt_mip,
    }
    st = _req(args.api, "/api/warm-queue", "POST", body)
    for s in st["submitted"]:
        if not s["accepted"]:
            print(f"  {s['root_id']}  already {s['status']} -- left as is")
    print(f"queued {sum(1 for s in st['submitted'] if s['accepted'])} cell(s)")
    if args.no_watch:
        return 0

    print("watching (Ctrl-C stops watching, not the queue)\n")
    mine = {str(int(r)) for r in args.root_ids}
    try:
        while True:
            st = _req(args.api, "/api/warm-queue")
            cells = [c for c in st["cells"] if c["root_id"] in mine]
            print(time.strftime("%H:%M:%S"))
            for c in cells:
                print(_line(c))
            if all(c["status"] in DONE for c in cells):
                incomplete = sum(c.get("failed", 0) for c in cells)
                if incomplete:
                    print(f"\n{incomplete} branch(es) did not finish -- re-run to resume "
                          "(cached branches are skipped)")
                return 0
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching; the queue is still building. "
              "`--status` to check, `--cancel ROOT_ID` to stop a cell.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
