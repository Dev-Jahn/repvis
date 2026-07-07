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
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DEVICES, RUNS_DIR, SOURCES_DIR, STATIC_DIR, REGISTRY
from .extract import flush_vram
from .pipeline import refit_and_render, run_group, segment_and_render

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
    of [number, number, 0|1]. Returns the point list or raises 400."""
    pts = payload.get("points")
    if not isinstance(pts, list):
        raise HTTPException(400, "points (list of [x,y,label]) required")
    out = []
    for p in pts:
        if (not isinstance(p, (list, tuple)) or len(p) != 3
                or any(isinstance(v, bool) for v in p)
                or not all(isinstance(v, (int, float)) for v in p[:2])
                or p[2] not in (0, 1)):
            raise HTTPException(400, "each point must be [number, number, 0|1]")
        out.append([float(p[0]), float(p[1]), int(p[2])])
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
    for d in RUNS_DIR.glob("*"):
        if d.name == rid or not d.is_dir():
            continue
        m = _run_record(d)
        if m and m.get("source_id") == sid and m.get("model") == model:
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
        SOURCES.pop(sid)
    shutil.rmtree(SOURCES_DIR / sid, ignore_errors=True)
    for d in RUNS_DIR.glob("*"):
        if d.is_dir():
            m = _run_record(d)
            if m and m.get("source_id") == sid:
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
    with LOCK:
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
                                             "result": r["result"],
                                             "seg": _seg_client(r.get("run_meta"))}
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


@app.post("/api/runs/{rid}/segment")
def run_segment(rid: str, payload: dict = Body(...)):
    """(Re)segment the foreground from client +/- points and re-bake the run's
    PCA video (blocking, on the single GPU worker). Empty `points` re-runs the
    DINO-saliency auto-seed."""
    if not _safe_id(rid):
        raise HTTPException(404)
    rd = RUNS_DIR / rid
    if not (rd / "feats.f16").exists():
        raise HTTPException(404)
    points = _parse_points(payload)
    with LOCK:
        if _active_run_ids():
            raise HTTPException(409, "a run is in progress")
    try:
        seg = EXEC.submit(segment_and_render, rd, points).result()
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
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
    if not (rd / "feats.f16").exists():
        raise HTTPException(404)
    with LOCK:
        if _active_run_ids():
            raise HTTPException(409, "a run is in progress")
    try:
        seg = EXEC.submit(refit_and_render, rd).result()
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        try:
            flush_vram()
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "seg": seg}


@app.delete("/api/runs")
def delete_runs():
    """Clear all completed results (in-flight groups are left untouched)."""
    with LOCK:
        active_ids = _active_run_ids()
    removed = 0
    for d in RUNS_DIR.glob("*"):
        if d.is_dir() and d.name not in active_ids:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return {"ok": True, "removed": removed}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
