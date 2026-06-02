"""FastAPI backend for the browser-native EM proofreading tool (M1: read-only glide).

Exposes the headless :class:`~proofreading.em.service.CellReviewService` over HTTP and serves
the sparse tube precomputed volumes from the SAME origin (a ``StaticFiles`` mount), so the
frontend reaches the JSON API, the chunks, and (later) the SPA all at ``http://localhost:8000``.

Endpoints (M1):
    POST /api/cells                                   -- open or resume a cell -> header
    GET  /api/cells/{root_id}/branches                -- branch checklist + summary
    GET  /api/cells/{root_id}/branches/{pid}/camera   -- build tube + camera path payload
    GET  /healthz
    /tube/<datastack>/<root_id>/{em,tgt}/...          -- precomputed chunks (CORS, no-store)

Run via :mod:`proofreading.em.serve`. Single-user, bind to localhost only.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .client import EMClient
from .service import CellReviewService


class OpenCellRequest(BaseModel):
    # root_id may arrive as a string (it exceeds JS 2^53); pydantic coerces "..." -> int exactly
    root_id: int
    datastack: Optional[str] = None
    version: Optional[int] = None
    step_nm: float = 500.0
    tube_mip: int = 1
    tube_radius_nm: float = 1000.0
    orient_to_path: bool = False


class AnnotateRequest(BaseModel):
    tag: str
    xyz_nm: tuple[float, float, float]  # click position in nm


class SetRootRequest(BaseModel):
    xyz_nm: tuple[float, float, float]  # clicked marker position in nm


def create_app(wal_dir: str, default_datastack: str = "minnie65_public") -> FastAPI:
    wal_dir = os.path.abspath(wal_dir)
    app = FastAPI(title="proofreading-em backend", version="0.1.0")

    # cross-origin during dev (Vite on :5173). In prod the SPA is same-origin.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # the tube cache is mutated in place as branches fill -> never cache its responses
    @app.middleware("http")
    async def _no_store_tube(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/tube/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ----- shared state ------------------------------------------------- #
    clients: dict[tuple, EMClient] = {}
    clients_lock = threading.Lock()
    sessions: dict[int, CellReviewService] = {}
    sessions_lock = threading.Lock()

    def get_client(datastack: str, version: Optional[int]) -> EMClient:
        key = (datastack, version)
        with clients_lock:
            c = clients.get(key)
            if c is None:
                c = EMClient(datastack, version=version)
                clients[key] = c
            return c

    def get_session(root_id: int) -> CellReviewService:
        s = sessions.get(int(root_id))
        if s is None:
            raise HTTPException(404, f"cell {root_id} not open; POST /api/cells first")
        return s

    # ----- endpoints (sync def -> blocking CAVE/CloudVolume runs in the threadpool) --- #
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "cells_open": len(sessions)}

    @app.post("/api/cells")
    def open_cell(req: OpenCellRequest):
        ds = req.datastack or default_datastack
        rid = int(req.root_id)
        with sessions_lock:
            s = sessions.get(rid)
            if s is None or s.datastack != ds:
                client = get_client(ds, req.version)
                s = CellReviewService(
                    client, rid, wal_dir,
                    step_nm=req.step_nm, tube_mip=req.tube_mip,
                    tube_radius_nm=req.tube_radius_nm, orient_to_path=req.orient_to_path,
                )
                sessions[rid] = s
        return s.header()

    @app.get("/api/cells/{root_id}/branches")
    def list_branches(root_id: int):
        return get_session(root_id).branches()

    @app.get("/api/cells/{root_id}/branches/{path_id}/camera")
    def branch_camera(root_id: int, path_id: int, request: Request, orient: bool = False):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        payload = s.camera_path(path_id, orient=orient)
        base = str(request.base_url).rstrip("/")  # e.g. http://localhost:8000
        payload["em_source"] = f"precomputed://{base}/{payload.pop('em_rel')}"
        payload["tgt_source"] = f"precomputed://{base}/{payload.pop('tgt_rel')}"
        return payload

    @app.post("/api/cells/{root_id}/annotations")
    def add_annotation(root_id: int, req: AnnotateRequest):
        s = get_session(root_id)
        try:
            return s.add_annotation(req.tag, list(req.xyz_nm))
        except ValueError as e:  # unknown tag
            raise HTTPException(400, str(e))

    @app.get("/api/cells/{root_id}/annotations")
    def list_annotations(root_id: int):
        return {"annotations": get_session(root_id).list_annotations()}

    @app.delete("/api/cells/{root_id}/annotations/{uuid}")
    def delete_annotation(root_id: int, uuid: str):
        return get_session(root_id).delete_annotation(uuid)

    @app.post("/api/cells/{root_id}/branches/{path_id}/done")
    def mark_branch_done(root_id: int, path_id: int):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        return s.mark_done(path_id)

    @app.get("/api/cells/{root_id}/live-sources")
    def live_sources(root_id: int):
        return get_session(root_id).live_sources()

    @app.get("/api/cells/{root_id}/skeleton-features")
    def skeleton_features(root_id: int):
        return get_session(root_id).skeleton_features()

    @app.post("/api/cells/{root_id}/root")
    def set_root(root_id: int, req: SetRootRequest):
        return get_session(root_id).set_root(list(req.xyz_nm))

    @app.on_event("shutdown")
    def _close_sessions():
        for s in sessions.values():
            s.close()

    # ----- static: serve the precomputed tube cache from this origin ---- #
    tube_dir = os.path.join(wal_dir, "tube_cache")
    os.makedirs(tube_dir, exist_ok=True)
    app.mount("/tube", StaticFiles(directory=tube_dir), name="tube")

    return app
