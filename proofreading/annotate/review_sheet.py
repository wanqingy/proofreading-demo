"""Spreadsheet-based colleague review workflow for the proofreading recorder.

    uv run python -m proofreading.annotate.review_sheet --export proofread_sessions/review_sheet.csv
    uv run python -m proofreading.annotate.review_sheet --apply proofread_sessions/review_sheet_filled.csv

--export writes one row per currently-recorded neurite -- the same "open" link the app's own
dropdown uses, plus seg/note counts and who/when recorded it -- with three blank columns for
colleagues to fill in while reviewing: ``reviewer``, ``updated_link``, ``comment``. Upload the
CSV to Google Sheets/Excel yourself and share it from there; there's no direct spreadsheet
integration here.

Every exported ``link`` also gets a second local annotation layer, ``review`` (see
``add_review_layer`` below), distinct from the original ``notes`` layer so a reviewer's marks
never commingle with the original author's. It carries a declared enum property (``int8``:
0=add, 1=remove, 2=merge error, 3=other, default 3) plus a ``Q``-key tool binding
(``annotateEnumProperty``) so a reviewer can tag a point without typing -- this only works on
a neuroglancer build that implements that tool (confirmed absent from this repo's pinned
``neuroglancer`` npm package; present on the live neuroglancer-demo.appspot.com host at time of
writing). This only changes the CSV's copy of the link -- the authoritative JSONL log is
untouched by --export.

--apply reads a filled-in copy back and reconciles it into the JSONL log, treating a non-empty
link in ``--column`` (default ``updated_link``; pass ``--column verification`` for a sheet
using that schema instead) exactly like the app's own "edit -> Record" flow would: recomputes
that neurite's segment set and notes straight from the pasted link (see ``recompute_from_link``
below -- the same logic used to fix the note-concatenation bug, and which deliberately ignores
a ``review`` layer so a reviewer's raw proposals are never auto-accepted as official notes),
then appends a corrected ``record`` event. That's a new event, not a rewrite of history, same
as every other update in this log. A non-empty value in that column that isn't a real link
(no ``#!`` in it -- e.g. "no changes", "calling for more votes") is printed in the summary, not
persisted -- there's no schema field in the JSONL log for freeform team commentary. Any row
whose segments collide with a DIFFERENT live neurite is flagged and skipped rather than
silently resolved: this is an offline batch tool with no one to ask "keep both, or drop it?",
so it defers instead of guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .log import RecordLog

# repo root = .../proofreading/annotate/review_sheet.py -> parents[2] (mirrors serve.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = os.environ.get(
    "PROOFREAD_ANNOTATE_LOG", str(_REPO_ROOT / "proofread_sessions" / "annotate_records.jsonl")
)

FIELDS = [
    "neurite_id", "link", "seg_count", "note_count", "user", "ts",
    "reviewer", "updated_link", "comment",
]


def decode_link(link: str) -> Optional[dict]:
    """Parse a captured neuroglancer link's ``#!<state>`` fragment back into JSON."""
    if not link or "#!" not in link:
        return None
    frag = link.split("#!", 1)[1]
    try:
        return json.loads(urllib.parse.unquote(frag))
    except json.JSONDecodeError:
        return None


def _encode_fragment(s: str) -> str:
    """Mirrors encodeFragment() in web/src/annotate.ts -- same extra escaping on top of the
    standard percent-encoding neuroglancer's own UrlHashBinding uses."""
    frag = urllib.parse.quote(s, safe="")
    extra = "!'()*;,"
    return "".join("%%%02X" % ord(c) if c in extra else c for c in frag)


def build_link(prefix: str, state: dict) -> str:
    return f"{prefix}#!{_encode_fragment(json.dumps(state))}"


# default enum vocabulary for the review layer's tagging property -- keep in sync with any
# future reviewer-facing docs/tooling that interprets these values.
REVIEW_ENUM_LABELS = ["add", "remove", "merge error", "other"]
REVIEW_DEFAULT_COORD_TRANSFORM = {"x": [2.08e-7, "m"], "y": [2.08e-7, "m"], "z": [1.8e-7, "m"]}

RAW_IMAGE_SOURCE = "brainmaps://73265790802:M808144_LH_S31a_s3_260305a_confocal:subvol1_raw"


def ensure_raw_image_layer(state: dict) -> dict:
    """A handful of early neurites (#1-#3) were captured before ``subvol1_raw`` was added as
    the standard image layer -- they carry only ``subvol1_clahe3d``, which every other neurite
    treats as the secondary/hidden layer (see any normal capture: subvol1_raw is visible,
    subvol1_clahe3d has ``"visible": false``). Swap that single layer for subvol1_raw so these
    few links match what every other neurite's reviewer already sees. No-op if subvol1_raw is
    already present. Deliberately omits shaderControls -- there's no per-capture contrast
    setting to inherit here, so let neuroglancer auto-normalize, same as a fresh session."""
    layers = state.get("layers", [])
    if any(l.get("type") == "image" and l.get("source") == RAW_IMAGE_SOURCE for l in layers):
        return state
    for layer in layers:
        if layer.get("type") == "image" and layer.get("name") == "subvol1_clahe3d":
            layer["source"] = RAW_IMAGE_SOURCE
            layer["name"] = "subvol1_raw"
            layer["tab"] = "rendering"
            layer.pop("shaderControls", None)
            break
    return state


def add_review_layer(state: dict) -> dict:
    """Add a second local annotation layer, ``review``, distinct from ``notes`` -- idempotent,
    a no-op if the link already has one (e.g. a round-trip through --apply). Reviewer marks
    live here instead of in ``notes`` so they never commingle with the original author's.

    Carries a declared enum property (int8: 0=add, 1=remove, 2=merge error, 3=other) and a
    toolBindings entry on "Q" (annotateEnumProperty) so a reviewer can tag a point with two
    clicks instead of typing a convention -- see the module docstring for the caveat that this
    tool type isn't implemented in every neuroglancer build.
    """
    layers = state.get("layers", [])
    if any(layer.get("type") == "annotation" and layer.get("name") == "review" for layer in layers):
        return state

    seg_layer_name = next((l["name"] for l in layers if l.get("type") == "segmentation"), None)
    # reuse whatever coordinate transform "notes" already uses (confirmed identical across every
    # neurite in the log) rather than re-deriving it, falling back to the known constant if this
    # link somehow has no annotation layer yet.
    transform = REVIEW_DEFAULT_COORD_TRANSFORM
    for layer in layers:
        if layer.get("type") == "annotation":
            t = layer.get("source", {}).get("transform", {}).get("outputDimensions")
            if t:
                transform = t
            break

    review_layer = {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {"outputDimensions": transform},
        },
        "tool": "annotatePoint",
        "toolBindings": {"Q": {"type": "annotateEnumProperty", "property": "int8"}},
        "annotationColor": "#33e0ff",  # distinct from notes' yellow (#ffcc00)
        "annotations": [],
        "annotationProperties": [
            {
                "id": "int8",
                "description": "",
                "type": "int8",
                "default": 3,
                "enum_labels": REVIEW_ENUM_LABELS,
                "enum_values": [0, 1, 2, 3],
            }
        ],
        **({"linkedSegmentationLayer": {"segments": seg_layer_name}} if seg_layer_name else {}),
        "filterBySegmentation": ["segments"],
        "name": "review",
    }
    state["layers"] = [*layers, review_layer]
    return state


def with_review_layer(link: str) -> str:
    """Decode a captured link, add the review layer and fix up a legacy missing-raw-image
    layer if needed, re-encode. Returns the original link unchanged if it doesn't decode (e.g.
    an empty/malformed link)."""
    state = decode_link(link)
    if state is None:
        return link
    prefix = link.split("#!", 1)[0]
    state = ensure_raw_image_layer(state)
    state = add_review_layer(state)
    return build_link(prefix, state)


def recompute_from_link(link: str) -> Tuple[Set[str], Dict[str, str]]:
    """(segments, notes) straight from a captured link -- mirrors captureVisibleSegments() +
    the fixed captureNotesBySegment() in web/src/annotate.ts, so a pasted-back link is treated
    exactly like a fresh Record/Update from the app itself, not a bespoke parse.

    Notes are read from every annotation layer EXCEPT ``review``: that layer holds a
    reviewer's raw proposals, never auto-accepted into official notes. Once someone has gone
    through those proposals and written the final decision into the regular annotation layer
    (named ``notes`` via this app, or just ``annotation`` when captured directly on a foreign
    neuroglancer page), that's what gets picked up here."""
    state = decode_link(link)
    if state is None:
        return set(), {}
    segments: Set[str] = set()
    for layer in state.get("layers", []):
        if layer.get("type") != "segmentation":
            continue
        for s in layer.get("segments", []):
            if not s.startswith("!"):  # "!"-prefixed = selected but hidden, not visible
                segments.add(s)
    by_seg: Dict[str, List[str]] = {}
    for layer in state.get("layers", []):
        if layer.get("type") != "annotation" or layer.get("name") == "review":
            continue
        for ann in layer.get("annotations", []):
            desc = ann.get("description")
            if not desc:
                continue
            for seg_list in ann.get("segments", []):  # JSON key is "segments", not "relatedSegments"
                for seg in seg_list:
                    by_seg.setdefault(str(seg), []).append(desc)
    notes = {seg: "; ".join(vals) for seg, vals in by_seg.items()}
    return segments, notes


def export_csv(log_path: str, out_path: str) -> int:
    state = RecordLog.load(log_path)
    rows = []
    for nid, rec in sorted(state.neurites.items()):
        rows.append(
            {
                "neurite_id": nid,
                "link": with_review_layer(rec.link),
                "seg_count": len(rec.ids),
                "note_count": sum(1 for n in rec.ids.values() if n),
                "user": rec.user,
                "ts": rec.ts,
                "reviewer": "",
                "updated_link": "",
                "comment": "",
            }
        )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def apply_csv(
    log_path: str,
    csv_path: str,
    column: str = "updated_link",
    on_conflict: str = "flag",
    only_ids: Optional[Set[int]] = None,
) -> None:
    """Reconcile a filled-in sheet into the log, reading the link from ``column`` (default
    ``updated_link``; pass ``column="verification"`` for a sheet using that newer schema).
    A non-empty value in that column that isn't a real link (no ``#!`` in it -- e.g. "no
    changes", "calling for more votes") is just noted, not treated as an error.

    ``on_conflict`` controls what happens when a segment is already owned by a DIFFERENT live
    neurite: ``"flag"`` (default) skips that row for you to resolve by hand, same as always;
    ``"keep"`` records it under both neurites (an intentionally shared segment), mirroring the
    app's own conflict-prompt "keep" choice. ``only_ids`` restricts processing to specific
    neurite ids -- use it to resolve just the rows a previous flagged run called out, without
    re-writing (harmless but redundant) record events for rows already applied cleanly."""
    log = RecordLog(log_path)  # append-mode writer -- RecordLog.load(log_path) below always re-reads fresh
    updated: List[int] = []
    noted: List[Tuple[int, str]] = []
    flagged: List[Tuple[int, str]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_id = (row.get("neurite_id") or "").strip()
            if not raw_id:
                continue
            nid = int(raw_id)
            if only_ids is not None and nid not in only_ids:
                continue
            value = (row.get(column) or "").strip()
            reviewer = (row.get("reviewer") or "").strip()

            if not value:
                continue
            if "#!" not in value:
                # not a pasted link -- e.g. "no changes", "calling for more votes", a free-text
                # classification. Informational only, never persisted.
                noted.append((nid, value))
                continue

            # reload fresh each time -- so a conflict check sees earlier rows already applied
            # in this same run, same as api.py's record() endpoint reloading before every write
            state = RecordLog.load(log_path)
            rec = state.neurites.get(nid)
            if rec is None:
                flagged.append((nid, "neurite is not currently live (deleted, or never existed) -- skipped"))
                continue

            new_segments, new_notes = recompute_from_link(value)
            if not new_segments:
                flagged.append((nid, f"{column} didn't decode to any segments -- skipped"))
                continue

            conflicts = {}
            for seg in new_segments:
                owner = state.owner_of(seg)
                if owner is not None and owner != nid:
                    conflicts[seg] = owner
            if conflicts and on_conflict != "keep":
                detail = ", ".join(f"{seg}->#{owner}" for seg, owner in sorted(conflicts.items()))
                flagged.append(
                    (nid, f"segments already owned by another neurite ({detail}) -- skipped, resolve by hand")
                )
                continue

            new_ids = {seg: new_notes.get(seg, "") for seg in new_segments}
            log.record(nid, value, new_ids, user=reviewer or rec.user)
            updated.append(nid)

    print(f"{len(updated)} neurites updated: {updated}")
    if noted:
        print(f"\n{len(noted)} rows with a non-link note in {column!r} (not persisted -- read these yourself):")
        for nid, note in noted:
            print(f"  #{nid}: {note}")
    if flagged:
        print(f"\n{len(flagged)} rows need manual attention:")
        for nid, reason in flagged:
            print(f"  #{nid}: {reason}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", default=DEFAULT_LOG, help="path to annotate_records.jsonl")
    ap.add_argument("--export", metavar="CSV", help="write a review sheet")
    ap.add_argument("--apply", metavar="CSV", help="reconcile a filled-in review sheet back into the log")
    ap.add_argument(
        "--column", default="updated_link",
        help="which column --apply reads the link from (default updated_link; e.g. verification)",
    )
    ap.add_argument(
        "--on-conflict", choices=["flag", "keep"], default="flag",
        help="flag (default): skip a row whose segments collide with another neurite, for you "
             "to resolve by hand. keep: record it under both neurites (intentionally shared).",
    )
    ap.add_argument(
        "--only", metavar="IDS",
        help="comma-separated neurite ids to process, skipping every other row -- e.g. to "
             "re-run just the rows a previous --apply flagged, without re-writing the rest",
    )
    args = ap.parse_args(argv)

    if not args.export and not args.apply:
        ap.error("pass --export or --apply")
    if args.export:
        n = export_csv(args.log, args.export)
        print(f"wrote {n} rows to {args.export}")
    if args.apply:
        only_ids = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
        apply_csv(args.log, args.apply, column=args.column, on_conflict=args.on_conflict, only_ids=only_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
