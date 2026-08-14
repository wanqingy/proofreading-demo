"""Append-only JSONL log for the simplified proofreading recorder.

One event type, ``record``, covers a full capture: the user proofreads/selects all
segments belonging to one neurite in the embedded neuroglancer viewer, then hits the
"Record" button, which reads the segmentation layer's visible segments, the annotation
layer's per-segment notes, and the current viewer link, and posts them here as one shot.
``neurite_id`` is assigned server-side (the user never types one).

Each event is written as one JSON line, then ``flush()`` + ``os.fsync()`` before returning,
so a crash loses at most the in-flight write. The read model ``{neurite_id: {link, ids}}``
is reconstructed by replaying the file -- there is no separate database.

A ``seg_id`` may be recorded under more than one neurite only when the caller has already
resolved the conflict (see :mod:`proofreading.annotate.api`); this module just persists
whatever ``ids`` mapping it is given per neurite.

A second event, ``delete``, tombstones a ``neurite_id`` -- mirrors the tombstone pattern in
:mod:`proofreading.em.wal`, so deletion is itself a durable, replayable log entry rather than
a destructive rewrite of history. A later ``record`` for the same id (an update, or a fresh
capture reusing the id) simply reappears on replay, since ``record`` is already a
full-overwrite-by-id operation.

Both events carry an optional ``user`` string (self-reported display name for now, not
cryptographically verified -- see the auth discussion for a future hardening pass) for
attribution when multiple people share one log.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class NeuriteRecord:
    link: str = ""
    ids: Dict[str, str] = field(default_factory=dict)  # seg_id -> note
    ts: str = ""
    user: str = ""  # who last recorded/updated this neurite (self-reported)


@dataclass
class RecordState:
    """Reconstructed state after replaying a log."""

    neurites: Dict[int, NeuriteRecord] = field(default_factory=dict)

    def owner_of(self, seg_id: str) -> Optional[int]:
        """The neurite id that already owns ``seg_id``, if any."""
        for neurite_id, rec in self.neurites.items():
            if seg_id in rec.ids:
                return neurite_id
        return None

    def next_id(self) -> int:
        return (max(self.neurites.keys(), default=0)) + 1


class RecordLog:
    """Append-only JSONL log with synchronous durability."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def _write(self, obj: dict) -> None:
        obj.setdefault("ts", _now())
        self._fh.write(json.dumps(obj) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record(self, neurite_id: int, link: str, ids: Dict[str, str], user: str = "") -> None:
        self._write(
            {"event": "record", "neurite_id": int(neurite_id), "link": link, "ids": dict(ids), "user": user}
        )

    def delete(self, neurite_id: int, user: str = "") -> None:
        self._write({"event": "delete", "neurite_id": int(neurite_id), "user": user})

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def load(path) -> RecordState:
        """Replay a log file into a :class:`RecordState`."""
        state = RecordState()
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
                if kind == "record":
                    state.neurites[int(ev["neurite_id"])] = NeuriteRecord(
                        link=ev.get("link", ""),
                        ids=dict(ev.get("ids", {})),
                        ts=ev.get("ts", ""),
                        user=ev.get("user", ""),
                    )
                elif kind == "delete":
                    state.neurites.pop(int(ev["neurite_id"]), None)
        return state
