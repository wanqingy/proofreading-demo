"""FastAPI backend for the simplified proofreading recorder.

Endpoints:
    GET    /api/neurites             -- list all recorded neurites (most recent first)
    GET    /api/neurites/{id}        -- single neurite detail
    POST   /api/record               -- record a capture, or update an existing neurite if
                                         ``neurite_id`` is given (may return a conflict list
                                         instead of writing -- see below)
    DELETE /api/neurites/{id}        -- tombstone a neurite (idempotent)
    GET    /healthz

POST /api/record body: ``{link, ids: {seg_id: note}, resolutions?: {seg_id: "keep"|"drop"},
neurite_id?: int, user?: str}``. Without ``neurite_id``, a new neurite id is assigned (a fresh capture).
With ``neurite_id``, that existing neurite's record is overwritten in place (an update) --
segments it already owns are exempt from conflict checks against itself.

For each ``seg_id`` already owned by a *different* neurite: if ``resolutions`` says "keep",
it's recorded under this neurite too (an intentionally shared segment); "drop" excludes it;
absent, the id is returned in the response's ``conflicts`` map (``seg_id -> owning neurite_id``)
and NOTHING is written -- the frontend re-POSTs with ``resolutions`` filled in once the user
has confirmed each one.

Single-user, bind to localhost only. Run via :mod:`proofreading.annotate.serve`.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .log import RecordLog


class RecordRequest(BaseModel):
    link: str = ""
    ids: Dict[str, str] = {}
    resolutions: Dict[str, str] = {}  # seg_id -> "keep" | "drop"
    neurite_id: Optional[int] = None  # present -> update this neurite instead of allocating one
    user: str = ""  # self-reported display name, for attribution only (not verified)


def create_app(log_path: str) -> FastAPI:
    log_path = os.path.abspath(log_path)
    app = FastAPI(title="proofreading-annotate backend", version="0.1.0")

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    log = RecordLog(log_path)
    lock = threading.Lock()

    def neurite_summary(neurite_id: int, rec) -> dict:
        return {
            "id": neurite_id,
            "link": rec.link,
            "ids": rec.ids,
            "seg_count": len(rec.ids),
            "note_count": sum(1 for n in rec.ids.values() if n),
            "ts": rec.ts,
            "user": rec.user,
        }

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/neurites")
    def list_neurites():
        state = RecordLog.load(log_path)
        items = [neurite_summary(nid, rec) for nid, rec in state.neurites.items()]
        items.sort(key=lambda x: x["id"], reverse=True)
        return {"neurites": items}

    @app.get("/api/neurites/{neurite_id}")
    def get_neurite(neurite_id: int):
        state = RecordLog.load(log_path)
        rec = state.neurites.get(neurite_id)
        if rec is None:
            return {"error": f"neurite {neurite_id} not found"}
        return neurite_summary(neurite_id, rec)

    @app.post("/api/record")
    def record(req: RecordRequest):
        with lock:
            state = RecordLog.load(log_path)
            target_id = req.neurite_id
            final_ids: Dict[str, str] = {}
            conflicts: Dict[str, int] = {}
            for seg_id, note in req.ids.items():
                owner = state.owner_of(seg_id)
                if owner is None or owner == target_id:
                    final_ids[seg_id] = note
                    continue
                resolution = req.resolutions.get(seg_id)
                if resolution == "keep":
                    final_ids[seg_id] = note
                elif resolution == "drop":
                    continue
                else:
                    conflicts[seg_id] = owner
            if conflicts:
                return {"status": "conflict", "conflicts": conflicts}
            neurite_id = target_id if target_id is not None else state.next_id()
            log.record(neurite_id, req.link, final_ids, user=req.user)
            return {
                "status": "ok",
                "neurite_id": neurite_id,
                "link": req.link,
                "ids": final_ids,
            }

    @app.delete("/api/neurites/{neurite_id}")
    def delete_neurite(neurite_id: int, user: str = ""):
        with lock:
            log.delete(neurite_id, user=user)
        return {"status": "ok"}

    return app
