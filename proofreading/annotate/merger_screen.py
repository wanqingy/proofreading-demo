"""Checkpoint merger-screening: batch links + a plain "screen" annotation layer.

    uv run python -m proofreading.annotate.merger_screen --generate \\
        --csv-a denovo_ckpt3102000_sizes.csv --name-a denovo_ckpt3102000 \\
        --seg-source-a "brainmaps://.../subvol1_clahe3d_ffn_v261019_denovo_ckpt3102000" \\
        --seg-layer-a ffn_denovo_ckpt3102000 \\
        --csv-b ft59110972_ckpt2829000_sizes.csv --name-b ft59110972_ckpt2829000 \\
        --seg-source-b "brainmaps://.../subvol1_clahe3d_ffn_v261019_ft59110972_ckpt2829000" \\
        --seg-layer-b ffn_ft59110972_ckpt2829000 \\
        --out proofread_sessions/merger_screen_batches.csv

    uv run python -m proofreading.annotate.merger_screen --tally \\
        --csv-a denovo_ckpt3102000_sizes.csv --name-a denovo_ckpt3102000 \\
        --csv-b ft59110972_ckpt2829000_sizes.csv --name-b ft59110972_ckpt2829000 \\
        --links proofread_sessions/merger_screen_batches_filled.csv

Two retrained segmentation checkpoints get compared by screening each one's top-N largest
segments (large objects are the most merger-prone) for merge errors, to pick the better
checkpoint -- repeatable across future checkpoint rounds.

--generate reads each checkpoint's ``id,num_voxels`` CSV, sorts by size descending, chunks into
batches of ``--batch-size`` (default 200), and builds one neuroglancer link per batch: the known
raw image layer, a segmentation layer for that checkpoint with the batch's ids all ``"!"``-
prefixed (selected but hidden -- nothing visible until a reviewer starts navigating, not all 200
shown at once), and a plain local annotation layer named ``screen`` with the **line** tool
active by default, linked to that batch's own segmentation layer. No declared schema, no
toolBindings -- `"annotateLine"` ships in every neuroglancer build (unlike the ``review`` layer's
``annotateEnumProperty`` in review_sheet.py, which is confirmed absent from this repo's pinned
npm package). A reviewer marks a merger by drawing a line from one true sub-object to another;
a 3+-way merger is multiple lines sharing the same segment id. Writes a small table (batch
index, checkpoint, id count, link) for handing off to whoever's screening each checkpoint.

Navigating within a batch (200 segments, no built-in neuroglancer "next in list" command) is
handled by a companion injectable script -- see web/public/merger-screen-nav.js -- not by this
module; this module only builds and reads back links.

--tally reads the links back (after reviewers have drawn their lines), decodes each, keeps only
the ``screen`` layer's ``type: "line"`` annotations, and groups them by ``relatedSegments`` to
reconstruct each flagged segment's distinct-object count: each line marks one distinct object
(its two endpoints are both sample points on that single object, not a connector between two
different objects), so the count is ``num_lines``. The raw line endpoints are kept alongside for
manual review of any complex multi-way case. Joins ``num_voxels`` back from the original CSVs. Produces
a per-checkpoint summary used to decide which checkpoint is better, and as the template for the
next round.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List, Optional, Tuple

from .review_sheet import (
    RAW_IMAGE_SOURCE,
    REVIEW_DEFAULT_COORD_TRANSFORM,
    build_link,
    decode_link,
)

BATCH_FIELDS = ["batch_index", "checkpoint", "id_count", "link"]

# midpoint of the observed [min, max] extent across all 198 captured positions in
# annotate_records.jsonl at the time this was computed (x:[21,1011] y:[10,1059] z:[1,700]) --
# a data-driven proxy for "center of the volume", not a queried bounding box (no such query
# exists in this repo).
VOLUME_CENTER_POSITION = (515.9, 534.0, 350.0)

# a real 1st-99th-percentile contrast range, computed live (e.g. via the histogram widget's
# "auto" button) rather than guessed -- see build_batch_state's docstring.
RAW_DATA_CONTRAST_RANGE = (1227.0985181970345, 9308.674066688762)


def read_sizes_csv(path: str) -> List[Tuple[str, int]]:
    """Read an ``id,num_voxels`` CSV, sorted by num_voxels descending."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["id"].strip(), int(row["num_voxels"])))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def chunk(items: list, size: int) -> List[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_batch_state(
    seg_source: str,
    seg_layer_name: str,
    ids: List[str],
    image_source: str = RAW_IMAGE_SOURCE,
    coord_transform: Optional[dict] = None,
) -> dict:
    """Fresh neuroglancer state for one batch: image layer, a segmentation layer for this
    checkpoint with every id selected-but-hidden (``"!"``-prefixed -- nothing visible until the
    nav helper's first advance), and a plain ``screen`` annotation layer with the line tool
    active, linked to this batch's own segmentation layer.

    Camera starts at VOLUME_CENTER_POSITION (the midpoint of the observed extent across every
    existing captured position in annotate_records.jsonl -- not a queried bounding box, since no
    such query exists in this repo, but a reasonable data-driven proxy). ``showSlices: False``
    turns off the 2D cross-section plane inside the 3D panel, leaving pure 3D rendering. The raw
    image layer's shaderControls use RAW_DATA_CONTRAST_RANGE for both range and window -- a real
    1st-99th-percentile contrast stretch computed live, not a guessed/default range. The
    segmentation layer's 2D opacity (``selectedAlpha``, not the 3D mesh's ``objectAlpha``) is
    dropped to 0.2 so a reviewer can still see the underlying raw data through a segment."""
    transform = coord_transform or REVIEW_DEFAULT_COORD_TRANSFORM
    return {
        "position": list(VOLUME_CENTER_POSITION),
        "showSlices": False,
        "layers": [
            {
                "type": "image",
                "source": image_source,
                "tab": "rendering",
                "name": "subvol1_raw",
                "shaderControls": {
                    "normalized": {
                        "range": list(RAW_DATA_CONTRAST_RANGE),
                        "window": list(RAW_DATA_CONTRAST_RANGE),
                    }
                },
            },
            {
                "type": "segmentation",
                "source": seg_source,
                "tab": "rendering",
                "segments": [f"!{i}" for i in ids],
                "name": seg_layer_name,
                "selectedAlpha": 0.2,
            },
            {
                "type": "annotation",
                "name": "screen",
                "tool": "annotateLine",
                "source": {
                    "url": "local://annotations",
                    "transform": {"outputDimensions": transform},
                },
                "annotations": [],
                "linkedSegmentationLayer": {"segments": seg_layer_name},
                "filterBySegmentation": ["segments"],
            },
        ],
        "layout": "xy-3d",
    }


def generate_batches(
    csv_path: str,
    checkpoint: str,
    seg_source: str,
    seg_layer_name: str,
    link_prefix: str,
    batch_size: int = 200,
    image_source: str = RAW_IMAGE_SOURCE,
    skip_ids: Optional[set] = None,
    start_index: int = 0,
) -> List[dict]:
    """``skip_ids`` excludes ids already covered by a preserved earlier batch (e.g. a batch
    someone's already started annotating that shouldn't be regenerated/invalidated) --
    ``start_index`` keeps that preserved batch's index out of the new ones' numbering."""
    rows = read_sizes_csv(csv_path)
    ids = [r[0] for r in rows if skip_ids is None or r[0] not in skip_ids]
    rows_out = []
    for i, batch_ids in enumerate(chunk(ids, batch_size), start=start_index):
        state = build_batch_state(seg_source, seg_layer_name, batch_ids, image_source=image_source)
        link = build_link(link_prefix, state)
        rows_out.append(
            {"batch_index": i, "checkpoint": checkpoint, "id_count": len(batch_ids), "link": link}
        )
    return rows_out


def generate_all(
    csv_a: str, name_a: str, seg_source_a: str, seg_layer_a: str,
    csv_b: str, name_b: str, seg_source_b: str, seg_layer_b: str,
    out_path: str,
    batch_size: int = 200,
    link_prefix: str = "https://neuroglancer-demo.appspot.com/",
) -> int:
    rows = generate_batches(csv_a, name_a, seg_source_a, seg_layer_a, link_prefix, batch_size)
    rows += generate_batches(csv_b, name_b, seg_source_b, seg_layer_b, link_prefix, batch_size)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _union_find_components(lines: List[Tuple[tuple, tuple]]) -> int:
    """Distinct-object count for one segment's merger lines: each line marks one distinct object
    (its two endpoints are both sample points on that single object, not a connector between two
    different objects) -- so the count is simply num_lines."""
    return len(lines)


def tally_checkpoint(links: List[str], voxels_by_id: Dict[str, int]) -> dict:
    """Walk a checkpoint's batch links, collect every ``screen``-layer annotation grouped by the
    segment id it's linked to, and summarize. ``voxels_by_id`` comes from that checkpoint's
    original sizes CSV. Line annotations are merger markers (one line per distinct object, see
    module docstring); point annotations are free-text reviewer notes on a segment -- not only
    astrocytes, reviewers also used them for other call-outs (e.g. "dendrite false split",
    "glia (OPC?)") -- kept verbatim rather than filtered to a fixed label set. ``clean_ids`` is
    every screened id with neither a merger line nor a note (screened_ids minus both), on the
    same assumption as above that a returned batch link means every id in it was looked at."""
    lines_by_seg: Dict[str, List[dict]] = {}
    points_by_seg: Dict[str, List[dict]] = {}
    screened_ids: set = set()

    for link in links:
        state = decode_link(link)
        if state is None:
            continue
        seg_layer = next((l for l in state["layers"] if l.get("type") == "segmentation"), None)
        if seg_layer:
            screened_ids.update(s.lstrip("!") for s in seg_layer.get("segments", []))
        screen = next(
            (l for l in state["layers"] if l.get("type") == "annotation" and l.get("name") == "screen"),
            None,
        )
        if not screen:
            continue
        for ann in screen.get("annotations", []):
            segs = [s for seg_list in ann.get("segments", []) for s in seg_list]
            if not segs:
                continue
            seg_id = segs[0]
            if ann.get("type") == "line":
                lines_by_seg.setdefault(seg_id, []).append(
                    {"point_a": ann.get("pointA"), "point_b": ann.get("pointB")}
                )
            elif ann.get("type") == "point":
                points_by_seg.setdefault(seg_id, []).append(
                    {"point": ann.get("point"), "description": ann.get("description")}
                )

    mergers = []
    for seg_id, lines in sorted(lines_by_seg.items(), key=lambda kv: -voxels_by_id.get(kv[0], 0)):
        mergers.append(
            {
                "segment_id": seg_id,
                "num_voxels": voxels_by_id.get(seg_id),
                "estimated_object_count": len(lines),
                "lines": lines,
            }
        )

    notes = []
    for seg_id, points in sorted(points_by_seg.items(), key=lambda kv: -voxels_by_id.get(kv[0], 0)):
        notes.append(
            {
                "segment_id": seg_id,
                "num_voxels": voxels_by_id.get(seg_id),
                "points": points,
            }
        )

    flagged_ids = set(lines_by_seg.keys())
    noted_ids = set(points_by_seg.keys())
    clean_ids = sorted(screened_ids - flagged_ids - noted_ids, key=lambda i: -voxels_by_id.get(i, 0))

    return {
        "segments_screened": len(screened_ids),
        "segments_flagged": len(mergers),
        "segments_noted": len(notes),
        "notes": notes,
        "clean_ids": clean_ids,
        "mergers": mergers,
    }


def tally_csv(
    links_path: str, name_a: str, csv_a: str, name_b: str, csv_b: str,
    link_column: str = "link",
) -> Dict[str, dict]:
    """``link_column`` is which column holds the (reviewed) link -- pass ``"updated_link"`` for
    a sheet that keeps the original ``link`` untouched and returns reviewer edits in a separate
    column (same schema review_sheet.py's apply_csv supports). A row with that column empty is
    not yet reviewed -- it's reported separately per checkpoint, not silently counted as clean
    (its original, un-annotated link has zero screen-layer lines by construction, so blending it
    in would inflate "segments_screened" without anyone having actually looked)."""
    rows = list(csv.DictReader(open(links_path, newline="", encoding="utf-8")))
    voxels_a = dict(read_sizes_csv(csv_a))
    voxels_b = dict(read_sizes_csv(csv_b))

    def reviewed_links(name):
        return [r[link_column] for r in rows if r["checkpoint"] == name and (r.get(link_column) or "").strip()]

    def not_reviewed(name):
        return [
            r["batch_index"] for r in rows
            if r["checkpoint"] == name and not (r.get(link_column) or "").strip()
        ]

    result = {
        name_a: tally_checkpoint(reviewed_links(name_a), voxels_a),
        name_b: tally_checkpoint(reviewed_links(name_b), voxels_b),
    }
    for name in (name_a, name_b):
        result[name]["batches_not_yet_reviewed"] = not_reviewed(name)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generate", action="store_true", help="build the 10 batch links")
    ap.add_argument("--tally", action="store_true", help="summarize returned batch links")
    ap.add_argument("--csv-a", required=True, help="checkpoint A's id,num_voxels CSV")
    ap.add_argument("--name-a", required=True, help="checkpoint A's label")
    ap.add_argument("--seg-source-a", help="checkpoint A's segmentation source URL (--generate)")
    ap.add_argument("--seg-layer-a", help="checkpoint A's segmentation layer name (--generate)")
    ap.add_argument("--csv-b", required=True, help="checkpoint B's id,num_voxels CSV")
    ap.add_argument("--name-b", required=True, help="checkpoint B's label")
    ap.add_argument("--seg-source-b", help="checkpoint B's segmentation source URL (--generate)")
    ap.add_argument("--seg-layer-b", help="checkpoint B's segmentation layer name (--generate)")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--out", help="--generate: where to write the batch-links CSV")
    ap.add_argument("--links", help="--tally: the (filled-in) batch-links CSV to read back")
    ap.add_argument(
        "--link-column", default="link",
        help="--tally: which column holds the reviewed link (default 'link'; pass "
             "'updated_link' if the original link column was kept untouched)",
    )
    args = ap.parse_args(argv)

    if args.generate:
        missing = [
            n for n, v in (
                ("--seg-source-a", args.seg_source_a), ("--seg-layer-a", args.seg_layer_a),
                ("--seg-source-b", args.seg_source_b), ("--seg-layer-b", args.seg_layer_b),
                ("--out", args.out),
            ) if not v
        ]
        if missing:
            ap.error(f"--generate requires: {', '.join(missing)}")
        n = generate_all(
            args.csv_a, args.name_a, args.seg_source_a, args.seg_layer_a,
            args.csv_b, args.name_b, args.seg_source_b, args.seg_layer_b,
            args.out, batch_size=args.batch_size,
        )
        print(f"wrote {n} batch links to {args.out}")
    elif args.tally:
        if not args.links:
            ap.error("--tally requires --links")
        summary = tally_csv(
            args.links, args.name_a, args.csv_a, args.name_b, args.csv_b,
            link_column=args.link_column,
        )
        print(json.dumps(summary, indent=2))
    else:
        ap.error("pass --generate or --tally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
