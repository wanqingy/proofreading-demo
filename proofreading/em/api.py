"""FastAPI backend for the browser-native EM proofreading tool (M1: read-only glide).

Exposes the headless :class:`~proofreading.em.service.CellReviewService` over HTTP and serves
the sparse tube precomputed volumes from the SAME origin (a ``StaticFiles`` mount), so the
frontend reaches the JSON API, the chunks, and the SPA all at ``http://localhost:8000``. Pass
``web_dist`` (a prebuilt ``vite build`` output) to serve the UI itself from this origin too --
see ``share/myelin-tool`` for the no-Node packaged form; without it, run the Vite dev server
separately as this repo's own dev loop does.

Endpoints (M1):
    POST /api/cells                                   -- open or resume a cell -> header
                                                          (warm_compartment=axon also kicks off a
                                                          whole-cell background tube warm-up)
    GET  /api/cells/{root_id}/branches                -- branch checklist + summary
                                                          (?compartment=axon restricts to that type)
    GET  /api/cells/{root_id}/branches/{pid}/camera   -- build tube + camera path payload
                                                          (includes nodes_nm, the TRUE sparse
                                                          skeleton vertices for the node overlay;
                                                          ?compartment=axon also scopes background
                                                          pre-build to the myelin tool's own
                                                          axon-only, myelin-coverage sequence)
    POST /api/cells/{root_id}/myelin/tag              -- tag the nearest skeleton node myelinated
    DELETE /api/cells/{root_id}/myelin/tag/{uuid}     -- remove a myelin tag
    GET  /api/cells/{root_id}/myelin/tags             -- live myelin tags (?path_id= restricts
                                                          to one branch)
    POST /api/cells/{root_id}/branches/{pid}/myelin-done -- mark a branch myelin-reviewed
                                                          (separate coverage dim from /done)
    GET  /api/cells/{root_id}/warm-status             -- live chunk-level caching progress
                                                          (+ stalled_s to spot a wedged read)
    POST   /api/warm-queue                            -- queue N cells to cache in the
                                                          background, ONE AT A TIME
    GET    /api/warm-queue                            -- queue status (+ live fill of the cell
                                                          being built now)
    DELETE /api/warm-queue/{root_id}                  -- drop a cell from the queue
    POST /api/crash-report                            -- append a renderer-kill breadcrumb to
                                                          <wal_dir>/crash_reports/<tool>.log
    GET  /healthz
    /tube/<datastack>/<root_id>/{em,tgt}/...          -- precomputed chunks (CORS, no-store)

Run via :mod:`proofreading.em.serve`. Single-user, bind to localhost only.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .client import EMClient
from .service import CellReviewService
from .warm_queue import WarmQueue


class OpenCellRequest(BaseModel):
    # root_id may arrive as a string (it exceeds JS 2^53); pydantic coerces "..." -> int exactly
    root_id: int
    datastack: Optional[str] = None
    version: Optional[int] = None
    step_nm: float = 500.0
    tube_mip: int = 1
    tube_radius_nm: float = 1000.0
    orient_to_path: bool = False
    # resolution of the red target-mask overlay. None -> CellTube.DEFAULT_TGT_MIP (env
    # PROOFREAD_TGT_MIP, default 4 = coarse but ~15x less data than the EM mip). Raise toward
    # tube_mip for a sharper mask at a large caching cost -- see CellTube's class comment.
    tgt_mip: Optional[int] = None
    # when set (e.g. "axon"), background-build EVERY remaining to-review branch of that
    # compartment right after opening, instead of only the reactive next-2 lookahead. Opt-in per
    # request so the myelin tool can warm a whole cell without changing main.ts's behaviour.
    warm_compartment: Optional[str] = None


class AnnotateRequest(BaseModel):
    tag: str
    xyz_nm: tuple[float, float, float]  # click position in nm


class SetRootRequest(BaseModel):
    xyz_nm: tuple[float, float, float]  # clicked marker position in nm


class MyelinTagRequest(BaseModel):
    xyz_nm: tuple[float, float, float]
    path_id: Optional[int] = None


class WarmQueueRequest(BaseModel):
    root_ids: list[int]  # warmed one cell at a time, in this order
    datastack: Optional[str] = None
    version: Optional[int] = None
    compartment: Optional[str] = "axon"  # matches the myelin tool's axon-only sweep
    tgt_mip: Optional[int] = None


class CrashReportRequest(BaseModel):
    tool: str  # picks the log FILE, not a path -- validated against _CRASH_TOOL_RE below
    report: str


# renderer-kill breadcrumbs (flykernel's crashwatch) land in <wal_dir>/crash_reports/<tool>.log --
# `tool` selects the filename, so it's validated against a whitelist rather than trusted as a path.
_CRASH_TOOL_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_CRASH_REPORT_MAX_BYTES = 64_000  # one breadcrumb report; a bad client can't force a huge write
_CRASH_LOG_MAX_BYTES = 1_000_000  # per-tool log cap -- a crash LOOP must not be able to fill disk


def create_app(
    wal_dir: str, default_datastack: str = "minnie65_public", web_dist: Optional[str] = None,
) -> FastAPI:
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

    def open_session(root_id: int, datastack: str, version: Optional[int] = None,
                     **kw) -> CellReviewService:
        """Get-or-create a cell session. Shared by /api/cells and the warm queue, so a cell the
        queue has already opened is the SAME session the browser then gets -- one skeleton
        fetch, and one set of per-branch build locks covering both."""
        rid = int(root_id)
        with sessions_lock:
            s = sessions.get(rid)
            if s is None or s.datastack != datastack:
                s = CellReviewService(get_client(datastack, version), rid, wal_dir, **kw)
                sessions[rid] = s
            return s

    warm_queue = WarmQueue(open_session)

    # ----- endpoints (sync def -> blocking CAVE/CloudVolume runs in the threadpool) --- #
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "cells_open": len(sessions)}

    @app.post("/api/cells")
    def open_cell(req: OpenCellRequest):
        ds = req.datastack or default_datastack
        s = open_session(
            req.root_id, ds, req.version,
            step_nm=req.step_nm, tube_mip=req.tube_mip,
            tube_radius_nm=req.tube_radius_nm, orient_to_path=req.orient_to_path,
            tgt_mip=req.tgt_mip,
        )
        header = s.header()
        if req.warm_compartment:
            header["warming"] = s.warm_cell(compartment=req.warm_compartment)
        return header

    @app.get("/api/cells/{root_id}/branches")
    def list_branches(root_id: int, compartment: Optional[str] = None):
        return get_session(root_id).branches(compartment=compartment)

    @app.get("/api/cells/{root_id}/branches/{path_id}/camera")
    def branch_camera(
        root_id: int, path_id: int, request: Request,
        orient: bool = False, compartment: Optional[str] = None,
    ):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        payload = s.camera_path(path_id, orient=orient, compartment=compartment)
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

    @app.post("/api/cells/{root_id}/annotations/{uuid}/status")
    def toggle_annotation_status(root_id: int, uuid: str):
        try:
            return get_session(root_id).toggle_annotation_status(uuid)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/cells/{root_id}/branches/{path_id}/done")
    def mark_branch_done(root_id: int, path_id: int):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        return s.mark_done(path_id)

    @app.post("/api/cells/{root_id}/branches/{path_id}/omit")
    def omit_branch(root_id: int, path_id: int):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        return s.omit_branch(path_id)

    @app.post("/api/cells/{root_id}/branches/{path_id}/myelin-done")
    def mark_branch_myelin_done(root_id: int, path_id: int):
        s = get_session(root_id)
        if path_id < 0 or path_id >= len(s.tree.branch_paths):
            raise HTTPException(404, f"path {path_id} out of range (0..{len(s.tree.branch_paths) - 1})")
        return s.myelin_mark_done(path_id)

    @app.get("/api/cells/{root_id}/warm-status")
    def warm_status(root_id: int):
        return get_session(root_id).warm_status()

    @app.post("/api/warm-queue")
    def warm_queue_submit(req: WarmQueueRequest):
        ds = req.datastack or default_datastack
        open_kw = {"version": req.version, "tgt_mip": req.tgt_mip}
        submitted = [warm_queue.submit(r, ds, req.compartment, open_kw) for r in req.root_ids]
        return {"submitted": submitted, **warm_queue.status()}

    @app.get("/api/warm-queue")
    def warm_queue_status():
        return warm_queue.status()

    @app.delete("/api/warm-queue/{root_id}")
    def warm_queue_cancel(root_id: int):
        return warm_queue.cancel(root_id)

    @app.get("/api/cells/{root_id}/live-sources")
    def live_sources(root_id: int):
        return get_session(root_id).live_sources()

    @app.get("/api/cells/{root_id}/skeleton-features")
    def skeleton_features(root_id: int):
        return get_session(root_id).skeleton_features()

    @app.post("/api/cells/{root_id}/root")
    def set_root(root_id: int, req: SetRootRequest):
        return get_session(root_id).set_root(list(req.xyz_nm))

    @app.post("/api/cells/{root_id}/resolve")
    def resolve_supervoxels(root_id: int):
        return get_session(root_id).resolve()

    @app.post("/api/cells/{root_id}/myelin/tag")
    def tag_myelinated_node(root_id: int, req: MyelinTagRequest):
        return get_session(root_id).tag_myelinated_node(list(req.xyz_nm), req.path_id)

    @app.delete("/api/cells/{root_id}/myelin/tag/{uuid}")
    def delete_myelin_tag(root_id: int, uuid: str):
        return get_session(root_id).delete_myelin_tag(uuid)

    @app.get("/api/cells/{root_id}/myelin/tags")
    def myelin_tags(root_id: int, path_id: Optional[int] = None):
        return get_session(root_id).myelin_tags(path_id)

    @app.post("/api/crash-report")
    def crash_report(req: CrashReportRequest):
        # A renderer kill (blank page) leaves no console and no in-page handler -- flykernel's
        # crashwatch instead persists a breadcrumb to localStorage and POSTs it here on the next
        # load, so it's readable from the terminal on a remote/headless machine.
        if not _CRASH_TOOL_RE.match(req.tool):
            raise HTTPException(400, "invalid tool name")
        body = req.report.encode("utf-8", errors="replace")
        if len(body) > _CRASH_REPORT_MAX_BYTES:
            raise HTTPException(413, "report too large")
        crash_dir = os.path.join(wal_dir, "crash_reports")
        os.makedirs(crash_dir, exist_ok=True)
        path = os.path.join(crash_dir, f"{req.tool}.log")
        entry = f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n{req.report}\n".encode(
            "utf-8", errors="replace"
        )
        existing = b""
        if os.path.isfile(path):
            with open(path, "rb") as f:
                existing = f.read()
        data = existing + entry
        if len(data) > _CRASH_LOG_MAX_BYTES:
            data = data[-_CRASH_LOG_MAX_BYTES:]  # oldest entries drop first
        with open(path, "wb") as f:
            f.write(data)
        return {"ok": True}

    @app.on_event("shutdown")
    def _close_sessions():
        warm_queue.shutdown()
        for s in sessions.values():
            s.close()

    # ----- static: serve the precomputed tube cache from this origin ---- #
    tube_dir = os.path.join(wal_dir, "tube_cache")
    os.makedirs(tube_dir, exist_ok=True)
    app.mount("/tube", StaticFiles(directory=tube_dir), name="tube")

    # ----- static: serve the prebuilt SPA from this origin (no Node needed) --------- #
    # Opt-in via `web_dist` so the normal dev loop (Vite on :5173, no dist/ present) is
    # unaffected -- this only activates for a packaged build (see share/myelin-tool). Mounted
    # LAST, after every /api and /tube route above, so it only ever catches what nothing else
    # claimed (Starlette matches routes in registration order, not by specificity).
    if web_dist and os.path.isdir(web_dist):
        if not os.path.isfile(os.path.join(web_dist, "index.html")):
            # a myelin-only build (see the trimmed rollupOptions.input in
            # share/myelin-tool/web/vite.config.ts) has no index.html for StaticFiles(html=True)
            # to fall back on at "/" -- go straight to the one page that exists instead of a bare
            # 404. Skipped entirely when index.html IS present (this repo's own 4-tool dist),
            # so it never shadows that page.
            @app.get("/")
            def _root():
                return RedirectResponse(url="/myelin.html")

        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
