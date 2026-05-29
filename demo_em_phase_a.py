#!/usr/bin/env python
"""EM proofreading -- Phase A (annotate) session, as a script (no Jupyter).

Read-only: review a cell along its L2 skeleton and drop tagged annotations; edits
happen manually later (Phase B), then re-enter on the new root id (Phase C).
Design: docs/proofreading-workflow.md. Needs the `em` extra and a CAVE token at
~/.cloudvolume/secrets/cave-secret.json.

    uv run --extra em python demo_em_phase_a.py --root 864691135572530981
    uv run --extra em python demo_em_phase_a.py --datastack minnie65_phase3_v1 --root <id> --version <mat>

Annotate with keys *in the neuroglancer browser window*; drive paths from this
prompt. The ipywidgets panel only renders in Jupyter, so here we use a small REPL.
"""

from __future__ import annotations

import argparse

import proofreading.em as em
from proofreading.em.wal import WAL

KEYMAP = (
    "keys (in the neuroglancer window):\n"
    "  m = merge error   s = split error   e = extend   q = question\n"
    "  n = toggle the segment under the cursor   x = mark current path reviewed\n"
    "  (a merge error ends the path early and prunes its distal subtree)"
)
COMMANDS = (
    "commands: [n]ext  [r <id>] review path  [play] [pause] [step]  "
    "[resolve]  [summary]  [quit]"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A annotate session (script)")
    ap.add_argument("--datastack", default="minnie65_public",
                    help="CAVE datastack (default: minnie65_public sandbox)")
    ap.add_argument("--root", type=int, default=864691135572530981, help="root id of the cell")
    ap.add_argument("--version", type=int, default=None, help="materialization version (live datastacks)")
    ap.add_argument("--wal-dir", default="./proofread_sessions", help="where the write-ahead logs live")
    ap.add_argument("--step-nm", type=float, default=1000.0, help="resample spacing along paths")
    args = ap.parse_args()

    client = em.EMClient(args.datastack, version=args.version)
    print(f"datastack {client.datastack} | materialization {client.mat_version}")
    sess = em.ProofreadSession(client, args.root, wal_dir=args.wal_dir, step_nm=args.step_nm)
    print(f"seed supervoxel: {sess.seed}")
    print(f"branch paths: {len(sess.tree.branch_paths)} | {sess.summary()}")
    print(f"\nOpen the viewer:\n   {sess.viewer}\n")
    print(KEYMAP)
    print("\n" + COMMANDS)

    sess.review_next()  # start on the first to-review path
    if sess.current_path_id is not None:
        print(f"reviewing path {sess.current_path_id} (press 'play' or use the browser)")

    try:
        while True:
            parts = input("phaseA> ").strip().split()
            if not parts:
                continue
            c = parts[0].lower()
            if c in ("q", "quit", "exit"):
                break
            elif c in ("n", "next"):
                fly = sess.review_next()
                print("nothing left to review" if fly is None
                      else f"reviewing path {sess.current_path_id}")
            elif c in ("r", "review") and len(parts) > 1:
                sess.review_path(int(parts[1]))
                print(f"reviewing path {sess.current_path_id}")
            elif c == "play" and sess.fly:
                sess.fly.play()
            elif c == "pause" and sess.fly:
                sess.fly.pause()
            elif c == "step" and sess.fly:
                sess.fly.step(1)
            elif c in ("resolve", "v"):
                print("resolved", sess.resolve_supervoxels(), "supervoxels")
            elif c in ("s", "summary"):
                print(sess.summary())
            else:
                print("?  " + COMMANDS)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        print("resolving supervoxels:", sess.resolve_supervoxels())
        sess.close()
        print("WAL:", sess.wal.path)


if __name__ == "__main__":
    main()
