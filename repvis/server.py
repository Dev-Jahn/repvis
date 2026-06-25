"""FastAPI server: content-addressed sources + multi-source runs (joint PCA).

Sources are stored once by content hash (re-processing never grows disk). A run
group processes one or more sources with a shared PCA basis; the matrix UI fills
a (source × model) grid as each run finishes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DEVICES, RUNS_DIR, SOURCES_DIR, STATIC_DIR, REGISTRY
from .extract import flush_vram
from .pipeline import run_group

app = FastAPI(title="repvis")

SOURCES: dict[str, dict] = {}   # source_id -> {id, name, ext, size}
GROUPS: dict[str, dict] = {}    # group_id  -> live group/progress state
LOCK = threading.Lock()
# Serialize GPU work so a single group gets the whole machine (and we never OOM-race).
EXEC = ThreadPoolExecutor(max_workers=1)

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_ID_RE = re.compile(r"[0-9a-f]{12,16}\Z")  # source hash (16 hex) / run-group id (12 hex)


def _safe_id(x: str) -> bool:
    return bool(x and _ID_RE.match(x))


def _clear_runs():
    """Run outputs are session-scoped (GROUPS is in-memory, never reloaded), so
    wipe stale run dirs on startup to keep disk bounded across restarts."""
    for d in RUNS_DIR.glob("*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


_clear_runs()


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


_scan_sources()


# ---------------------------------------------------------------- progress emit
def _emit(group: dict, **kw):
    with LOCK:
        rid = kw.pop("run_id", None)
        rstatus = kw.pop("run_status", None)
        rresult = kw.pop("result", None)
        if rid is not None and rid in group["runs"]:
            r = group["runs"][rid]
            if rresult is not None:
                r["result"] = rresult
            if rstatus is not None:
                r["status"] = rstatus
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
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, "empty upload")
        sid = h.hexdigest()[:16]
        d = SOURCES_DIR / sid
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(d / ("video" + ext)))
            (d / "meta.json").write_text(json.dumps(
                {"name": file.filename or sid, "ext": ext, "size": size}))
        else:
            tmp.unlink(missing_ok=True)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    rec = {"id": sid, "name": file.filename or sid, "ext": ext, "size": size}
    with LOCK:
        SOURCES[sid] = rec
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
    group = {"id": gid, "model": model, "status": "queued", "stage": "queued",
             "progress": 0.0, "message": "Queued…", "error": None, "rev": 0, "runs": runs}
    with LOCK:
        GROUPS[gid] = group

    def task():
        try:
            run_group(items, model, opts, lambda **kw: _emit(group, **kw))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _emit(group, stage="error", error=str(e), message=f"Error: {e}")
        finally:
            try:
                flush_vram()
            except Exception:  # noqa: BLE001
                pass

    EXEC.submit(task)
    return {"group_id": gid, "model": model,
            "runs": [{"run_id": it["run_id"], "source_id": it["source_id"],
                      "pca_url": f"/api/runs/{it['run_id']}/pca",
                      "original_url": f"/api/sources/{it['source_id']}/video"}
                     for it in items]}


@app.get("/api/runs/{gid}/events")
async def events(gid: str):
    if gid not in GROUPS:
        raise HTTPException(404)

    async def gen():
        last = -1
        while True:
            with LOCK:
                g = GROUPS.get(gid)
                if not g:
                    break
                if g["rev"] != last:
                    last = g["rev"]
                    payload = {k: g[k] for k in ("status", "stage", "progress", "message", "error")}
                    payload["runs"] = {rid: {"source_id": r["source_id"], "status": r["status"],
                                             "result": r["result"]}
                                       for rid, r in g["runs"].items()}
                    status = g["status"]
                else:
                    payload = status = None
            if payload is not None:
                yield f"data: {json.dumps(payload)}\n\n"
                if status in ("done", "error"):
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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
