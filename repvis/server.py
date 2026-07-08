"""FastAPI server: content-addressed sources + multi-source runs (joint PCA).

Sources are stored once by content hash (re-processing never grows disk). A run
group processes one or more sources with a shared PCA basis; the matrix UI fills
a (source × model) grid as each run finishes. Completed runs persist on disk
(meta.json + pca.mp4) so the workspace survives page reloads and restarts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AUTH_TOKEN, DEVICES, RUNS_DIR, SOURCES_DIR, STATIC_DIR, REGISTRY
from .extract import flush_vram
from .pipeline import drop_seg_cache, refit_and_render, run_group, segment_and_render

app = FastAPI(title="repvis")

SOURCES: dict[str, dict] = {}   # source_id -> {id, name, ext, size}
GROUPS: dict[str, dict] = {}    # group_id  -> live group/progress state
LOCK = threading.Lock()
# rids of in-flight /segment|/refit mutations (guarded by LOCK). Deletes and the
# _persist_run supersede skip/refuse these so we never clobber a run mid-mutation;
# a second /segment|/refit on an rid already in-flight is rejected with 409 (a plain
# set can't tell two same-rid mutations apart, so the first finisher would otherwise
# discard the marker while the second is still queued/running on the single worker).
ACTIVE_RUN_MUTATIONS: set[str] = set()
# Serialize GPU work so a single group gets the whole machine (and we never OOM-race).
EXEC = ThreadPoolExecutor(max_workers=1)

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_ID_RE = re.compile(r"[0-9a-f]{12,16}\Z")  # source hash (16 hex) / run-group id (12 hex)


def _safe_id(x: str) -> bool:
    return bool(x and _ID_RE.match(x))


def _scan_sources():
    """Rebuild the in-memory source index from disk (survives restarts)."""
    for d in sorted(SOURCES_DIR.glob("*")):
        if not d.is_dir():
            continue
        vids = list(d.glob("video.*"))
        if not vids:
            continue
        v = vids[0]
        meta_p = d / "meta.json"
        name = d.name
        if meta_p.exists():
            try:
                name = json.loads(meta_p.read_text()).get("name", d.name)
            except Exception:  # noqa: BLE001
                pass
        SOURCES[d.name] = {"id": d.name, "name": name, "ext": v.suffix,
                           "size": v.stat().st_size}


def _run_record(d: Path) -> dict | None:
    """meta.json of a *completed* run dir, or None if the dir isn't one."""
    try:
        m = json.loads((d / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return None
    return m if (d / "pca.mp4").exists() else None


def _read_meta(rid: str) -> dict | None:
    """meta.json of a run dir (completed or in-flight), or None if unreadable."""
    try:
        return json.loads((RUNS_DIR / rid / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return None


def _seg_client(meta: dict | None) -> dict:
    """The per-run seg object the client uses to place +/- points and map them
    to source pixels. Missing seg material / meta -> {available: False}."""
    if not meta or "seg" not in meta:
        return {"available": False}
    return {**meta["seg"], "grid": meta.get("grid"),
            "frames": meta.get("frames"), "fps": meta.get("fps")}


def _parse_points(payload: dict) -> list:
    """Validate a segment request body: `points` must be a list (possibly empty)
    of [x, y, label, frame] with x,y finite numbers, label in {0,1} and frame an
    int >= 0 (bools rejected). Run-specific bounds (x<W, y<H, frame<T) are enforced
    in the pipeline (ValueError -> 422). Returns the point list or raises 400."""
    pts = payload.get("points")
    if not isinstance(pts, list):
        raise HTTPException(400, "points (list of [x,y,label,frame]) required")
    out = []
    for p in pts:
        if (not isinstance(p, (list, tuple)) or len(p) != 4
                or any(isinstance(v, bool) for v in p)
                or not all(isinstance(v, (int, float)) for v in p[:2])
                or not all(math.isfinite(v) for v in p[:2])
                or p[2] not in (0, 1)
                or not isinstance(p[3], int) or p[3] < 0):
            raise HTTPException(400, "each point must be [number, number, 0|1, frame>=0]")
        out.append([float(p[0]), float(p[1]), int(p[2]), int(p[3])])
    return out


def _prune_runs():
    """Startup GC: drop incomplete, orphaned (source gone) and superseded run
    dirs. Completed runs persist so the workspace survives restarts."""
    latest: dict[tuple[str, str], tuple[float, Path]] = {}
    for d in RUNS_DIR.glob("*"):
        if not d.is_dir():
            continue
        m = _run_record(d)
        if not m or m.get("source_id") not in SOURCES:
            shutil.rmtree(d, ignore_errors=True)
            continue
        key = (m["source_id"], m["model"])
        prev = latest.get(key)
        if prev and prev[0] >= m.get("created", 0.0):
            shutil.rmtree(d, ignore_errors=True)
        else:
            if prev:
                shutil.rmtree(prev[1], ignore_errors=True)
            latest[key] = (m.get("created", 0.0), d)


_scan_sources()
_prune_runs()


# ---------------------------------------------------------------- progress emit
def _persist_run(meta: dict):
    """Write a completed run's meta.json and retire the previous result for the
    same (source, model) cell — a re-run supersedes it."""
    rid, sid, model = meta["run_id"], meta["source_id"], meta["model"]
    rmeta = meta.pop("run_meta", None)
    if rmeta:  # lift grid/feat_dim/frames/fps/frame_indices/seg to meta top-level
        meta.update(rmeta)
    (RUNS_DIR / rid / "meta.json").write_text(json.dumps(meta))
    with LOCK:
        for d in RUNS_DIR.glob("*"):
            if d.name == rid or not d.is_dir() or d.name in ACTIVE_RUN_MUTATIONS:
                continue
            m = _run_record(d)
            if m and m.get("source_id") == sid and m.get("model") == model:
                drop_seg_cache(d.name)   # a re-run supersedes this run; its cache is stale
                shutil.rmtree(d, ignore_errors=True)


def _emit(group: dict, **kw):
    persist = None
    with LOCK:
        rid = kw.pop("run_id", None)
        rstatus = kw.pop("run_status", None)
        rresult = kw.pop("result", None)
        rmeta = kw.pop("run_meta", None)
        if rid is not None and rid in group["runs"]:
            r = group["runs"][rid]
            if rresult is not None:
                r["result"] = rresult
            if rmeta is not None:
                r["run_meta"] = rmeta
            if rstatus is not None:
                r["status"] = rstatus
            if rstatus == "done":
                persist = {"run_id": rid, "source_id": r["source_id"],
                           "model": group["model"], "opts": group["opts"],
                           "result": r["result"], "run_meta": r.get("run_meta"),
                           "created": time.time()}
        group.update(kw)
        if kw.get("error"):
            group["status"] = "error"
            for r in group["runs"].values():   # don't leave runs stuck "running"
                if r["status"] == "running":
                    r["status"] = "error"
        elif kw.get("stage") == "done":
            group["status"] = "done"
        elif "stage" in kw or "progress" in kw:
            group["status"] = "running"
        group["rev"] += 1
    if persist:   # disk I/O outside the lock; only the single worker gets here
        _persist_run(persist)


# --------------------------------------------------------------- access control
# Shared-token gate. When AUTH_TOKEN is set (non-empty) every request must carry
# the token via `Authorization: Bearer <t>`, an `X-Repvis-Token` header, or the
# `repvis_token` cookie; POST /api/login trades the token for that cookie so the
# browser (video/img/EventSource, which can't set headers) authenticates too.
# When AUTH_TOKEN is unset the gate is disabled (open) — see the startup warning.
# AUTH_TOKEN is read as a module global on every request so tests can toggle it.
AUTH_COOKIE = "repvis_token"
_AUTH_EXEMPT = frozenset({"/", "/api/login", "/api/auth"})

if not AUTH_TOKEN:
    print("repvis: REPVIS_TOKEN is not set — running WITHOUT access control; "
          "every source video, run and API route is publicly reachable.",
          file=sys.stderr)


def _authorized(request: Request, token: str) -> bool:
    """True iff the request presents `token` via Bearer / X-Repvis-Token / cookie
    (constant-time compare; any one matching source is enough)."""
    tb = token.encode()
    cands = []
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        cands.append(auth[7:])
    xt = request.headers.get("x-repvis-token")
    if xt is not None:
        cands.append(xt)
    ck = request.cookies.get(AUTH_COOKIE)
    if ck is not None:
        cands.append(ck)
    return any(secrets.compare_digest(c.encode(), tb) for c in cands)


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    token = AUTH_TOKEN
    if token and request.url.path not in _AUTH_EXEMPT and not _authorized(request, token):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    return await call_next(request)


@app.get("/api/auth")
def auth_status():
    """Whether the server enforces the shared token (so the UI knows to prompt)."""
    return {"required": bool(AUTH_TOKEN)}


@app.post("/api/login")
def login(request: Request, payload: dict = Body(default={})):
    """Trade a valid token (JSON {"token": …} or a Bearer / X-Repvis-Token header)
    for the httpOnly `repvis_token` cookie. No-op when auth is disabled."""
    token = AUTH_TOKEN
    if not token:
        return {"ok": True, "required": False}
    body_tok = payload.get("token")
    ok = ((body_tok is not None
           and secrets.compare_digest(str(body_tok).encode(), token.encode()))
          or _authorized(request, token))
    if not ok:
        raise HTTPException(401, "invalid token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(AUTH_COOKIE, token, httponly=True, samesite="strict", path="/")
    return resp


# ------------------------------------------------------------------ static/info
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
def models():
    out = [{"key": k, "label": s.label, "family": s.family,
            "available": s.is_available(), "note": s.note,
            "patch": s.patch, "max_side": s.max_side}
           for k, s in REGISTRY.items()]
    return {"models": out, "gpus": len(DEVICES)}


@app.post("/api/flush")
def flush():
    """Free cached GPU memory from the previous job (models stay loaded)."""
    return flush_vram()


# ----------------------------------------------------------------------- sources
@app.get("/api/sources")
def list_sources():
    with LOCK:
        return {"sources": list(SOURCES.values())}


def _materialize_source(sid: str, tmp: Path, ext: str, name: str, size: int, rec: dict):
    """Register a streamed upload atomically with delete_source (both take LOCK).
    Reuse the on-disk dir only if it actually holds the video (a half-materialized
    empty dir left by an interrupted upload is re-created); otherwise move the temp
    file into place. Runs on a worker thread — NEVER the event loop — so acquiring the
    blocking LOCK here can't stall SSE/async requests while a delete holds it."""
    with LOCK:
        d = SOURCES_DIR / sid
        if not any(d.glob("video.*")):   # dir absent or missing its video -> (re)materialize
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(d / ("video" + ext)))
            (d / "meta.json").write_text(json.dumps({"name": name, "ext": ext, "size": size}))
        SOURCES[sid] = rec


@app.post("/api/sources")
async def upload_source(file: UploadFile = File(...)):
    """Store an upload once, keyed by content hash (sha256). Re-uploading the
    same bytes returns the existing source — disk never grows for duplicates."""
    ext = Path(file.filename or "input.mp4").suffix.lower()
    if ext not in _VIDEO_EXTS:
        ext = ".mp4"

    # Stream to a temp file while hashing, so we never hold the whole video in RAM.
    h = hashlib.sha256()
    tmp = Path(tempfile.mkstemp(suffix=ext, dir=SOURCES_DIR)[1])
    size = 0
    try:
        with tmp.open("wb") as out:
            while True:
                buf = await file.read(1 << 20)
                if not buf:
                    break
                h.update(buf)
                out.write(buf)
                size += len(buf)
        if size == 0:
            raise HTTPException(400, "empty upload")
        sid = h.hexdigest()[:16]
        rec = {"id": sid, "name": file.filename or sid, "ext": ext, "size": size}
        # Materialize + register off the event loop: this section takes the blocking
        # LOCK (shared with delete_source, which holds it across an rmtree). Acquiring
        # it on the loop thread would freeze every SSE/async request for the delete's
        # duration; a worker thread blocks alone. Still one critical section, so a dup
        # upload can never register sid after a concurrent delete removed the dir.
        await asyncio.to_thread(_materialize_source, sid, tmp, ext, file.filename or sid, size, rec)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return rec


def _source_video(sid: str) -> Path | None:
    if not _safe_id(sid):
        return None
    d = SOURCES_DIR / sid
    if not d.is_dir():
        return None
    for p in d.glob("video.*"):
        return p
    return None


@app.get("/api/sources/{sid}/video")
def source_video(sid: str):
    p = _source_video(sid)
    if not p or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


def _active_run_ids() -> set[str]:
    """Run ids belonging to groups still queued/running (call under LOCK)."""
    return {rid for g in GROUPS.values() if g["status"] in ("queued", "running")
            for rid in g["runs"]}


@app.delete("/api/sources/{sid}")
def delete_source(sid: str):
    """Remove a source and every result derived from it."""
    if not _safe_id(sid):
        raise HTTPException(404)
    with LOCK:
        if _active_run_ids():
            raise HTTPException(409, "a run is in progress")
        if sid not in SOURCES:
            raise HTTPException(404)
        derived = [d for d in RUNS_DIR.glob("*")
                   if d.is_dir() and (_run_record(d) or {}).get("source_id") == sid]
        if any(d.name in ACTIVE_RUN_MUTATIONS for d in derived):
            raise HTTPException(409, "a derived run is being segmented/refit")
        SOURCES.pop(sid)
        shutil.rmtree(SOURCES_DIR / sid, ignore_errors=True)
        for d in derived:
            drop_seg_cache(d.name)
            shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


# --------------------------------------------------------------------- workspace
@app.get("/api/workspace")
def workspace():
    """Everything the client needs to rebuild the matrix: sources, completed
    runs (latest per source×model, oldest first) and still-active groups the
    client can re-attach to after a reload."""
    with LOCK:
        srcs = list(SOURCES.values())
        have = set(SOURCES)
        active_ids = _active_run_ids()
        active = [{"group_id": g["id"], "model": g["model"],
                   "runs": [{"run_id": rid, "source_id": r["source_id"],
                             "pca_url": f"/api/runs/{rid}/pca",
                             "original_url": f"/api/sources/{r['source_id']}/video"}
                            for rid, r in g["runs"].items()]}
                  for g in GROUPS.values() if g["status"] in ("queued", "running")]
    for a in active:   # best-effort seg (in-flight meta may not exist yet)
        for run in a["runs"]:
            run["seg"] = _seg_client(_read_meta(run["run_id"]))
    metas = []
    for d in RUNS_DIR.glob("*"):
        if d.is_dir() and d.name not in active_ids:
            m = _run_record(d)
            if m and m.get("source_id") in have:
                metas.append(m)
    metas.sort(key=lambda m: m.get("created", 0.0))
    runs = [{"run_id": m["run_id"], "source_id": m["source_id"], "model": m["model"],
             "result": m["result"], "seg": _seg_client(m),
             "pca_url": f"/api/runs/{m['run_id']}/pca",
             "original_url": f"/api/sources/{m['source_id']}/video"} for m in metas]
    return {"sources": srcs, "runs": runs, "active": active}


# -------------------------------------------------------------------------- runs
@app.post("/api/runs")
def create_runs(payload: dict = Body(...)):
    """Start a run group: process `source_ids` with `model` (+ opts).
    With >1 source the PCA basis is shared (joint, cross-video colors)."""
    source_ids = payload.get("source_ids") or []
    model = payload.get("model")
    opts = payload.get("opts") or {}
    if not source_ids:
        raise HTTPException(400, "no sources")
    if model not in REGISTRY:
        raise HTTPException(400, "unknown model")
    if not REGISTRY[model].is_available():
        raise HTTPException(400, f"model '{model}' weights are not available")

    # Validate sources, mkdir the run dirs and register the group in ONE critical
    # section: a concurrent delete_runs/delete_source can't rmtree a just-validated
    # source or a nascent run dir between the check and the group going live.
    with LOCK:
        items = []
        for sid in source_ids:
            inp = _source_video(sid)
            if not inp:
                raise HTTPException(404, f"unknown source {sid}")
            rid = uuid.uuid4().hex[:12]
            rd = RUNS_DIR / rid
            rd.mkdir(parents=True, exist_ok=True)
            items.append({"run_id": rid, "source_id": sid, "input": inp, "out": rd / "pca.mp4"})

        gid = uuid.uuid4().hex[:12]
        runs = {it["run_id"]: {"run_id": it["run_id"], "source_id": it["source_id"],
                               "status": "running", "result": None} for it in items}
        group = {"id": gid, "model": model, "opts": opts, "status": "queued", "stage": "queued",
                 "progress": 0.0, "message": "Queued…", "error": None, "rev": 0, "runs": runs}
        GROUPS[gid] = group

    def task():
        try:
            run_group(items, model, opts, lambda **kw: _emit(group, **kw))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _emit(group, stage="error", error=str(e), message=f"Error: {e}")
        finally:
            for it in items:   # failed/aborted runs (no meta.json) leave no dirs behind
                rd = RUNS_DIR / it["run_id"]
                if not (rd / "meta.json").exists():
                    shutil.rmtree(rd, ignore_errors=True)
            try:
                flush_vram()
            except Exception:  # noqa: BLE001
                pass

    EXEC.submit(task)
    return {"group_id": gid, "model": model,
            "runs": [{"run_id": it["run_id"], "source_id": it["source_id"],
                      "seg": _seg_client(_read_meta(it["run_id"])),
                      "pca_url": f"/api/runs/{it['run_id']}/pca",
                      "original_url": f"/api/sources/{it['source_id']}/video"}
                     for it in items]}


def _events_tick(gid: str, last: int):
    """Snapshot a group's SSE payload under LOCK, off the event loop (via to_thread).
    Returns None if the group is gone, (None, rev) if unchanged since `last`, or
    (payload, rev) with a fresh payload. Keeps the loop thread from blocking on LOCK
    while a delete holds it across an rmtree."""
    with LOCK:
        g = GROUPS.get(gid)
        if not g:
            return None
        if g["rev"] == last:
            return None, g["rev"]
        payload = {k: g[k] for k in ("status", "stage", "progress", "message", "error")}
        payload["runs"] = {rid: {"source_id": r["source_id"], "status": r["status"],
                                 "result": r["result"], "seg": _seg_client(r.get("run_meta"))}
                           for rid, r in g["runs"].items()}
        return payload, g["rev"]


@app.get("/api/runs/{gid}/events")
async def events(gid: str):
    if gid not in GROUPS:
        raise HTTPException(404)

    async def gen():
        last = -1
        while True:
            tick = await asyncio.to_thread(_events_tick, gid, last)
            if tick is None:            # group gone
                break
            payload, last = tick
            if payload is not None:
                yield f"data: {json.dumps(payload)}\n\n"
                if payload["status"] in ("done", "error"):
                    break
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{rid}/pca")
def run_pca(rid: str):
    if not _safe_id(rid):
        raise HTTPException(404)
    p = RUNS_DIR / rid / "pca.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/runs/{rid}/segment")
def run_segment(rid: str, payload: dict = Body(...)):
    """(Re)segment the foreground from client +/- points and re-bake the run's
    PCA video (blocking, on the single GPU worker). Empty `points` re-runs the
    DINO-saliency auto-seed."""
    if not _safe_id(rid):
        raise HTTPException(404)
    rd = RUNS_DIR / rid
    points = _parse_points(payload)
    with LOCK:
        if not (rd / "feats.f16").exists():
            raise HTTPException(404)
        if _active_run_ids():
            raise HTTPException(409, "a run is in progress")
        if rid in ACTIVE_RUN_MUTATIONS:
            raise HTTPException(409, "this run is already being segmented/refit")
        ACTIVE_RUN_MUTATIONS.add(rid)
    try:
        seg = EXEC.submit(segment_and_render, rd, points).result()
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404)
    except ValueError as e:   # out-of-bounds/non-finite points (run dims known in pipeline)
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        with LOCK:
            ACTIVE_RUN_MUTATIONS.discard(rid)
        try:
            flush_vram()
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "seg": seg}


@app.post("/api/runs/{rid}/refit")
def run_refit(rid: str):
    """Re-fit the display basis over the current foreground mask and re-render
    the run's PCA video (blocking, on the single GPU worker)."""
    if not _safe_id(rid):
        raise HTTPException(404)
    rd = RUNS_DIR / rid
    with LOCK:
        if not (rd / "feats.f16").exists():
            raise HTTPException(404)
        if _active_run_ids():
            raise HTTPException(409, "a run is in progress")
        if rid in ACTIVE_RUN_MUTATIONS:
            raise HTTPException(409, "this run is already being segmented/refit")
        ACTIVE_RUN_MUTATIONS.add(rid)
    try:
        seg = EXEC.submit(refit_and_render, rd).result()
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        with LOCK:
            ACTIVE_RUN_MUTATIONS.discard(rid)
        try:
            flush_vram()
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "seg": seg}


@app.delete("/api/runs")
def delete_runs():
    """Clear all completed results (in-flight groups and mutating runs untouched)."""
    with LOCK:
        skip = _active_run_ids() | ACTIVE_RUN_MUTATIONS
        removed = 0
        for d in RUNS_DIR.glob("*"):
            if d.is_dir() and d.name not in skip:
                drop_seg_cache(d.name)
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return {"ok": True, "removed": removed}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
