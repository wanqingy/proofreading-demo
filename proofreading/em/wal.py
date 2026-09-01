"""Append-only write-ahead log -- the session source of truth (ADR 0002).

Every event is appended as one JSON line and ``fsync``'d before anything else, so a
crash at most loses the in-flight keypress; recovery is replaying the file. The
in-memory annotation list and the neuroglancer layers are *views* of this log.

Events
------
- ``annotation`` -- a tagged point: ``uuid, tag, xyz`` (nm), ``root_id``,
  ``mat_version``, ``seed``. ``supervoxel`` is filled later by a ``resolve`` event
  (the click ``xyz`` is the source of truth; supervoxels are derived in batch via
  CloudVolume -- see :mod:`proofreading.em.client`).
- ``resolve`` -- attaches a ``supervoxel`` to an annotation ``uuid``.
- ``tombstone`` -- soft-deletes an annotation ``uuid`` (also reverses any ``omit`` it caused).
- ``ann_status`` -- records whether an annotation has been addressed (``done: bool``).
  Last write per uuid wins; used by the Phase B review queue to track progress.
- ``visit`` -- marks a batch of L2 ids reviewed (coverage).
- ``omit`` -- marks L2 ids omitted, keyed to the merge-error annotation ``uuid`` that caused it.
- ``set_root`` -- the user-chosen review root (``xyz_nm``), so re-rooting survives a reload
  (somaless cells default to an arbitrary tip). Resolution-independent; last one wins.
- ``myelin_tag`` -- a single skeleton NODE tagged myelinated (``uuid, xyz`` (nm, snapped to
  the tagged vertex), ``root_id``, ``mat_version``, ``seed``, ``path_id``). Same *shape* as
  ``annotation``/``Tag`` (one discrete, deliberate point per action -- see ``CONTEXT.md``) but
  a different vocabulary/file: presence of a tag on a node means myelinated; absence means the
  unmyelinated default, exactly like there's no "not a merge error" tag either. ``path_id`` is
  a same-session convenience only (never a durable anchor -- ``xyz`` is, per ADR 0001); it is
  not re-validated against the current tree on replay.
- ``myelin_tag_tombstone`` -- soft-deletes a myelin tag by ``uuid`` (mirrors ``tombstone``).
- ``myelin_visit`` -- marks a batch of L2 ids reviewed **for myelination**. Deliberately a
  SEPARATE set from ``visit``/``visited_l2`` (the merge/split/extend/question review's
  coverage): a branch marked done in the myelin tool has only been checked for myelination,
  not for proofreading errors, so conflating the two would make one tool's progress lie about
  the other's.
- ``cell_done`` -- the whole cell declared finished (``done: bool``), independent of any
  particular branch's state. Reversible like ``ann_status``: last write wins. Written to the
  myelin stream only, by the myelin tool's "cell done" action -- it says nothing about the
  review pass, same separation as ``myelin_visit`` vs ``visit`` above.

One log per cell **per event vocabulary**, named by the **seed supervoxel** (the durable
identity) so a later session over a new root id appends to and resumes the same log(s). The
review tags (``annotation``/``visit``/``omit``/...) and the myelin events
(``myelin_tag``/``myelin_visit``/...) live in SEPARATE files -- see ``WAL.for_cell``'s ``kind``
parameter -- even though they describe the same cell, because they're independent tools with
independent vocabularies and no shared derivation.
"""

from __future__ import annotations

import json
import os
import time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

TAGS = ("merge error", "split error", "extend", "question")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Annotation:
    uuid: str
    tag: str
    xyz: List[float]  # nm
    root_id: int
    mat_version: int
    seed: Optional[int] = None
    ts: str = ""
    supervoxel: Optional[int] = None  # filled by a resolve event


@dataclass
class MyelinTag:
    uuid: str
    xyz: List[float]  # nm -- snapped to the tagged vertex before writing (see service.py)
    root_id: int
    mat_version: int
    seed: Optional[int] = None
    path_id: Optional[int] = None
    ts: str = ""


@dataclass
class WalState:
    """Reconstructed state after replaying a log."""

    annotations: Dict[str, Annotation] = field(default_factory=dict)  # live (un-tombstoned)
    done_uuids: Set[str] = field(default_factory=set)  # annotations marked done in Phase B
    visited_l2: Set[int] = field(default_factory=set)
    omitted_l2: Set[int] = field(default_factory=set)
    omit_by_uuid: Dict[str, Set[int]] = field(default_factory=dict)
    root_xyz: Optional[List[float]] = None  # last chosen review root (nm); None = skeleton default
    myelin_visited_l2: Set[int] = field(default_factory=set)  # separate from visited_l2 -- see wal docstring
    myelin_tags: Dict[str, MyelinTag] = field(default_factory=dict)  # live (un-tombstoned), uuid-keyed
    cell_done: bool = False  # this whole cell declared finished (last cell_done event wins)
    cell_done_ts: Optional[str] = None


class WAL:
    """Append-only JSONL log with synchronous durability."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    @classmethod
    def for_cell(
        cls,
        directory,
        datastack: str,
        seed_supervoxel: int,
        kind: str = "review",
        root_id: Optional[int] = None,
    ) -> "WAL":
        """Open a cell's log, named by BOTH its segment id and its durable seed supervoxel.

        ``kind`` selects which independent event stream for this cell: "review" (default,
        no suffix) is the merge/split/extend/question tag workflow; "myelin" is the myelination
        tool. Same cell, same directory, separate files.

        Naming carries both ids for different reasons, and the distinction is load-bearing:

        * the **segment (root) id** is in the name so a human can tell at a glance which cell a
          log belongs to -- a bare supervoxel id is unrecognisable, and a visits-only log records
          the root id nowhere inside either.
        * the **seed supervoxel** is the durable identity, and is therefore what LOOKUP keys on.
          Root ids change whenever anyone edits the segmentation (see
          ``CellReviewService.resolve``, which re-derives the root FROM the seed for exactly this
          reason). Keying the lookup on the segment id would mean that the first session after
          someone else's edit silently opens a NEW empty log and orphans every existing tag.

        So: find by seed, name by both, and rename in place when the root id has moved. Renaming
        an already-open file is safe on POSIX (handles follow the inode). Nothing is ever deleted,
        merged or overwritten here -- if the destination somehow already exists we use it and leave
        the other file untouched rather than guess which is authoritative.
        """
        suffix = "" if kind == "review" else f"__{kind}"
        d = Path(directory)
        seed = int(seed_supervoxel)
        if root_id is None:
            # No root id available: keep the historical name. Still found later by the seed glob.
            return cls(d / f"{datastack}__seed{seed}{suffix}.jsonl")

        desired = d / f"{datastack}__seg{int(root_id)}__seed{seed}{suffix}.jsonl"
        if desired.exists():
            return cls(desired)

        # Find this cell's existing log by SEED, whatever segment id its name currently carries,
        # including the pre-seg-id legacy name.
        found = [p for p in sorted(d.glob(f"{datastack}__seg*__seed{seed}{suffix}.jsonl"))]
        legacy = d / f"{datastack}__seed{seed}{suffix}.jsonl"
        if legacy.exists():
            found.append(legacy)

        if len(found) > 1:
            # Shouldn't happen. Adopt the largest (most history) and leave the rest alone, loudly.
            found.sort(key=lambda p: p.stat().st_size, reverse=True)
            print(
                f"[wal] WARNING: {len(found)} logs for seed {seed}: "
                f"{[p.name for p in found]}; using {found[0].name} and leaving the others untouched"
            )
        if found:
            try:
                found[0].rename(desired)
            except OSError as e:  # cross-device, permissions, race -- keep using the old name
                print(f"[wal] could not rename {found[0].name} -> {desired.name} ({e}); using as-is")
                return cls(found[0])
        return cls(desired)

    # ----- writing (each call is durable) -------------------------------- #
    def _write(self, obj: dict) -> None:
        obj.setdefault("ts", _now())
        self._fh.write(json.dumps(obj) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def add_annotation(
        self,
        tag: str,
        xyz,
        root_id: int,
        mat_version: int,
        seed: Optional[int] = None,
    ) -> Annotation:
        if tag not in TAGS:
            raise ValueError(f"unknown tag {tag!r}; expected one of {TAGS}")
        ann = Annotation(
            uuid=_uuid.uuid4().hex,
            tag=tag,
            xyz=[float(c) for c in xyz],
            root_id=int(root_id),
            mat_version=int(mat_version),
            seed=None if seed is None else int(seed),
            ts=_now(),
        )
        self._write({"event": "annotation", **ann.__dict__})
        return ann

    def resolve_supervoxel(self, uuid: str, supervoxel: int) -> None:
        self._write({"event": "resolve", "uuid": uuid, "supervoxel": int(supervoxel)})

    def tombstone(self, uuid: str) -> None:
        self._write({"event": "tombstone", "uuid": uuid})

    def set_annotation_done(self, uuid: str, done: bool) -> None:
        self._write({"event": "ann_status", "uuid": uuid, "done": done})

    def mark_visited(self, l2_ids) -> None:
        self._write({"event": "visit", "l2_ids": [int(x) for x in l2_ids]})

    def mark_omitted(self, l2_ids, because_uuid: str) -> None:
        self._write(
            {"event": "omit", "uuid": because_uuid, "l2_ids": [int(x) for x in l2_ids]}
        )

    def set_root(self, xyz_nm) -> None:
        self._write({"event": "set_root", "xyz_nm": [float(c) for c in xyz_nm]})

    def mark_myelin_visited(self, l2_ids, root_id: Optional[int] = None) -> None:
        # root_id is recorded so a visits-only log is still self-describing. `myelin_tag` events
        # already carry it, but a cell reviewed without any myelinated node produces visits only,
        # and such a file used to identify its cell nowhere at all.
        ev = {"event": "myelin_visit", "l2_ids": [int(x) for x in l2_ids]}
        if root_id is not None:
            ev["root_id"] = int(root_id)
        self._write(ev)

    def tag_myelinated(
        self,
        xyz,
        root_id: int,
        mat_version: int,
        seed: Optional[int] = None,
        path_id: Optional[int] = None,
    ) -> MyelinTag:
        tag = MyelinTag(
            uuid=_uuid.uuid4().hex,
            xyz=[float(c) for c in xyz],
            root_id=int(root_id),
            mat_version=int(mat_version),
            seed=None if seed is None else int(seed),
            path_id=None if path_id is None else int(path_id),
            ts=_now(),
        )
        self._write({"event": "myelin_tag", **tag.__dict__})
        return tag

    def delete_myelin_tag(self, uuid: str) -> None:
        self._write({"event": "myelin_tag_tombstone", "uuid": uuid})

    def set_cell_done(self, done: bool) -> None:
        # Reversible, like set_annotation_done -- a misclick costs one more click, not a corrupted
        # log. Written to whichever stream this WAL instance is (myelin, in the one caller today).
        self._write({"event": "cell_done", "done": bool(done)})

    def close(self) -> None:
        self._fh.close()

    # ----- replay -------------------------------------------------------- #
    @staticmethod
    def load(path) -> WalState:
        """Replay a log file into a :class:`WalState` (handles tombstones/resolves)."""
        state = WalState()
        tombstoned: Set[str] = set()
        myelin_tag_tombstoned: Set[str] = set()
        path = Path(path)
        if not path.exists():
            return state
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                kind = ev.get("event")
                if kind == "annotation":
                    a = {k: ev[k] for k in ("uuid", "tag", "xyz", "root_id", "mat_version") }
                    state.annotations[ev["uuid"]] = Annotation(
                        seed=ev.get("seed"), ts=ev.get("ts", ""),
                        supervoxel=ev.get("supervoxel"), **a,
                    )
                elif kind == "resolve":
                    ann = state.annotations.get(ev["uuid"])
                    if ann is not None:
                        ann.supervoxel = int(ev["supervoxel"])
                elif kind == "tombstone":
                    tombstoned.add(ev["uuid"])
                elif kind == "ann_status":
                    if ev.get("done"):
                        state.done_uuids.add(ev["uuid"])
                    else:
                        state.done_uuids.discard(ev["uuid"])
                elif kind == "visit":
                    state.visited_l2.update(int(x) for x in ev["l2_ids"])
                elif kind == "omit":
                    state.omit_by_uuid.setdefault(ev["uuid"], set()).update(
                        int(x) for x in ev["l2_ids"]
                    )
                elif kind == "set_root":
                    state.root_xyz = [float(c) for c in ev["xyz_nm"]]  # last wins
                elif kind == "myelin_visit":
                    state.myelin_visited_l2.update(int(x) for x in ev["l2_ids"])
                elif kind == "myelin_tag":
                    state.myelin_tags[ev["uuid"]] = MyelinTag(
                        uuid=ev["uuid"],
                        xyz=[float(c) for c in ev["xyz"]],
                        root_id=int(ev.get("root_id", 0)),
                        mat_version=int(ev.get("mat_version", 0)),
                        seed=ev.get("seed"),
                        path_id=ev.get("path_id"),
                        ts=ev.get("ts", ""),
                    )
                elif kind == "myelin_tag_tombstone":
                    myelin_tag_tombstoned.add(ev["uuid"])
                elif kind == "cell_done":
                    state.cell_done = bool(ev.get("done"))  # last event wins -- file order
                    state.cell_done_ts = ev.get("ts")
        # apply tombstones: drop annotations and reverse any omissions they caused
        for u in tombstoned:
            state.annotations.pop(u, None)
            state.omit_by_uuid.pop(u, None)
        for ids in state.omit_by_uuid.values():
            state.omitted_l2.update(ids)
        for u in myelin_tag_tombstoned:
            state.myelin_tags.pop(u, None)
        return state
