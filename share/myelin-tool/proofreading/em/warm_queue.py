"""Queue several cells for background tube caching, one cell at a time.

Caching a whole cell takes ~30 min (57 axon branches, ~23GB of EM at mip1 + masks at
``PROOFREAD_TGT_MIP``), so the useful unit of "go make coffee" work is a *list* of cells, not
one. This is that list: submit N cells, they warm in submission order, and each is fully
reviewable long before the last one starts.

Two decisions worth knowing about, because both are easy to get wrong:

**Cells warm strictly serially.** A single branch fill already runs 16 threads against the
tuned connection pool (:mod:`proofreading.em._http_pool`), so warming two cells at once does
not add throughput -- it splits the same bandwidth and pushes *both* finish times out. One
worker thread also means one FIFO, so "cell 1 is ready" happens as early as it possibly can
instead of every cell being 40% done at the same time.

**The queue lives inside the API process.** A standalone warming script would not share
:class:`~proofreading.em.service.CellReviewService`'s per-branch locks, so it would happily
re-fetch a branch the server was already building for the user, at full cost. Keeping one
fetcher in one process makes the existing ``_branch_locks`` / ``_prebuilding`` guards cover
the queue too; :mod:`proofreading.em.warm_cells` is the CLI that drives it over HTTP.

The queue is in-memory and deliberately not persisted: the *durable* record of progress is the
tube cache itself, so re-queueing a cell after a restart skips everything already on disk.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

# a cell that is still going to be worked on -- re-submitting one of these is a no-op
LIVE = ("queued", "opening", "warming", "cancelling")


@dataclass
class _Entry:
    root_id: int
    datastack: str
    compartment: Optional[str]
    open_kw: dict = field(default_factory=dict)
    status: str = "queued"  # queued|opening|warming|cancelling|done|cancelled|error
    total: int = 0          # branches in the compartment
    already_built: int = 0  # cached before we started
    pending: int = 0        # branches this run set out to build
    built: int = 0          # branches attempted so far
    failed: int = 0         # attempted but still not cached (fill hit its time budget, or errored)
    error: str = ""

    def as_dict(self) -> dict:
        d = {
            "root_id": str(self.root_id),  # string: exceeds JS 2^53
            "datastack": self.datastack,
            "compartment": self.compartment,
            "status": self.status,
            "total": self.total,
            "already_built": self.already_built,
            "pending": self.pending,
            "built": self.built,
            "failed": self.failed,
        }
        if self.error:
            d["error"] = self.error
        return d


class WarmQueue:
    """Serial, process-wide background warm-up of whole cells.

    ``open_session(root_id, datastack, **open_kw)`` must return a ready
    :class:`~proofreading.em.service.CellReviewService` (the API layer's session cache), so a
    queued cell the user then opens in the browser is the SAME session -- no second skeleton
    fetch, and no duplicate builds.
    """

    def __init__(self, open_session: Callable[..., object]):
        self._open = open_session
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warmq")
        self._entries: dict[int, _Entry] = {}
        self._order: list[int] = []          # submission order, for a stable status listing
        self._svc: dict[int, object] = {}    # root_id -> service, for live per-chunk progress
        self._cancelled: set[int] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def submit(self, root_id: int, datastack: str, compartment: str | None = None,
               open_kw: dict | None = None) -> dict:
        """Add one cell to the back of the queue (no-op if it's already queued or running)."""
        rid = int(root_id)
        with self._lock:
            e = self._entries.get(rid)
            if e is not None and e.status in LIVE:
                return {"root_id": str(rid), "status": e.status, "accepted": False}
            # a finished/failed/cancelled cell may be re-queued -- it resumes from the cache
            self._entries[rid] = _Entry(rid, datastack, compartment, open_kw or {})
            if rid not in self._order:
                self._order.append(rid)
            self._cancelled.discard(rid)
        self._ex.submit(self._run, rid)
        return {"root_id": str(rid), "status": "queued", "accepted": True}

    def cancel(self, root_id: int) -> dict:
        """Drop a cell from the queue. A cell that is already warming stops after its CURRENT
        branch: a branch fill is not interruptible, and killing one mid-way would leave a
        partially-cached branch with no marker, i.e. wasted bandwidth."""
        rid = int(root_id)
        with self._lock:
            e = self._entries.get(rid)
            if e is None:
                return {"root_id": str(rid), "status": "unknown"}
            self._cancelled.add(rid)
            if e.status == "queued":
                e.status = "cancelled"
            elif e.status in ("opening", "warming"):
                e.status = "cancelling"
            return {"root_id": str(rid), "status": e.status}

    def status(self) -> dict:
        with self._lock:
            entries = [self._entries[r] for r in self._order if r in self._entries]
        cells = []
        for e in entries:
            d = e.as_dict()
            svc = self._svc.get(e.root_id)
            if e.status in ("warming", "cancelling") and svc is not None:
                d["fill"] = svc.warm_status()  # live chunk counts for the cell being built now
            cells.append(d)
        active = next((c["root_id"] for c in cells if c["status"] in ("opening", "warming",
                                                                     "cancelling")), None)
        waiting = sum(1 for c in cells if c["status"] == "queued")
        return {"cells": cells, "active": active, "waiting": waiting}

    def shutdown(self) -> None:
        with self._lock:
            self._cancelled.update(self._entries)  # stop after the branch in flight
        self._ex.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ #
    def _run(self, root_id: int) -> None:
        """The single worker thread: open one cell, then build its branches in review order."""
        e = self._entries[root_id]
        if root_id in self._cancelled:
            e.status = "cancelled"
            return
        try:
            e.status = "opening"
            svc = self._open(root_id, e.datastack, **e.open_kw)  # CAVE skeleton fetch
            self._svc[root_id] = svc
            targets = svc.warm_targets(e.compartment)
            e.total = int(targets["total"])
            e.already_built = int(targets["already_built"])
            e.pending = len(targets["pending"])
            e.status = "warming"
            for pid in targets["pending"]:
                if root_id in self._cancelled:
                    e.status = "cancelled"
                    return
                svc.warm_branch(pid)
                e.built += 1
                # warm_branch is best-effort (a fill can hit its time budget); the marker is the
                # only honest answer to "is this branch actually cached?"
                if not svc.branch_built(pid):
                    e.failed += 1
            e.status = "done"
        except Exception as exc:  # a bad root_id / CAVE outage must not kill the worker thread
            e.status = "error"
            e.error = f"{type(exc).__name__}: {exc}"
