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
- ``visit`` -- marks a batch of L2 ids reviewed (coverage).
- ``omit`` -- marks L2 ids omitted, keyed to the merge-error annotation ``uuid`` that caused it.
- ``set_root`` -- the user-chosen review root (``xyz_nm``), so re-rooting survives a reload
  (somaless cells default to an arbitrary tip). Resolution-independent; last one wins.

One log per cell, named by the **seed supervoxel** (the durable identity), so a later
session over a new root id appends to and resumes the same log.
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
class WalState:
    """Reconstructed state after replaying a log."""

    annotations: Dict[str, Annotation] = field(default_factory=dict)  # live (un-tombstoned)
    visited_l2: Set[int] = field(default_factory=set)
    omitted_l2: Set[int] = field(default_factory=set)
    omit_by_uuid: Dict[str, Set[int]] = field(default_factory=dict)
    root_xyz: Optional[List[float]] = None  # last chosen review root (nm); None = skeleton default


class WAL:
    """Append-only JSONL log with synchronous durability."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    @classmethod
    def for_cell(cls, directory, datastack: str, seed_supervoxel: int) -> "WAL":
        """Open the log for a cell, named by its durable seed supervoxel."""
        fname = f"{datastack}__seed{seed_supervoxel}.jsonl"
        return cls(Path(directory) / fname)

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

    def mark_visited(self, l2_ids) -> None:
        self._write({"event": "visit", "l2_ids": [int(x) for x in l2_ids]})

    def mark_omitted(self, l2_ids, because_uuid: str) -> None:
        self._write(
            {"event": "omit", "uuid": because_uuid, "l2_ids": [int(x) for x in l2_ids]}
        )

    def set_root(self, xyz_nm) -> None:
        self._write({"event": "set_root", "xyz_nm": [float(c) for c in xyz_nm]})

    def close(self) -> None:
        self._fh.close()

    # ----- replay -------------------------------------------------------- #
    @staticmethod
    def load(path) -> WalState:
        """Replay a log file into a :class:`WalState` (handles tombstones/resolves)."""
        state = WalState()
        tombstoned: Set[str] = set()
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
                elif kind == "visit":
                    state.visited_l2.update(int(x) for x in ev["l2_ids"])
                elif kind == "omit":
                    state.omit_by_uuid.setdefault(ev["uuid"], set()).update(
                        int(x) for x in ev["l2_ids"]
                    )
                elif kind == "set_root":
                    state.root_xyz = [float(c) for c in ev["xyz_nm"]]  # last wins
        # apply tombstones: drop annotations and reverse any omissions they caused
        for u in tombstoned:
            state.annotations.pop(u, None)
            state.omit_by_uuid.pop(u, None)
        for ids in state.omit_by_uuid.values():
            state.omitted_l2.update(ids)
        return state
