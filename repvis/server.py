"""FastAPI server: upload -> background processing -> SSE progress -> side-by-side videos."""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DEVICES, JOBS_DIR, REGISTRY, STATIC_DIR
from .extract import flush_vram
from .pipeline import run_job

app = FastAPI(title="repvis")

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
# Serialize GPU jobs so a single job gets the full machine (and we never OOM-race).
EXEC = ThreadPoolExecutor(max_workers=1)


def _emit(job: dict, **kw):
    with JOBS_LOCK:
        job.update(kw)
        job["rev"] += 1
        if kw.get("error"):
            job["status"] = "error"
        elif kw.get("stage") == "done":
            job["status"] = "done"
        else:
            job["status"] = "running"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
def models():
    out = []
    for k, s in REGISTRY.items():
        out.append({
            "key": k, "label": s.label, "family": s.family,
            "available": s.is_available(), "note": s.note,
            "patch": s.patch, "max_side": s.max_side,
        })
    return {"models": out, "gpus": len(DEVICES)}


@app.post("/api/flush")
def flush():
    """Free cached GPU memory from the previous job (models stay loaded)."""
    return flush_vram()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(...),
    remove_bg: bool = Form(False),
    l2norm: bool = Form(False),
    fps: float = Form(24.0),
    max_frames: int = Form(900),
    max_side: int = Form(0),
):
    if model not in REGISTRY:
        raise HTTPException(400, "unknown model")
    spec = REGISTRY[model]
    if not spec.is_available():
        raise HTTPException(400, f"model '{model}' weights are not available")

    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "queued", "stage": "queued", "progress": 0.0,
           "message": "Queued…", "error": None, "result": None, "rev": 0, "model": model}
    with JOBS_LOCK:
        JOBS[jid] = job

    jd = JOBS_DIR / jid
    jd.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    inp = jd / ("input" + suffix)
    inp.write_bytes(await file.read())

    opts = {"remove_bg": remove_bg, "l2norm": l2norm, "fps": fps,
            "max_frames": max_frames, "max_side": max_side}

    def task():
        try:
            run_job(jd, inp, model, opts, lambda **kw: _emit(job, **kw))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _emit(job, stage="error", error=str(e), message=f"Error: {e}")
        finally:
            # release this job's cached GPU memory right away (models stay loaded),
            # so a finished OR failed job never sits holding VRAM.
            try:
                flush_vram()
            except Exception:  # noqa: BLE001
                pass

    EXEC.submit(task)
    return {"job_id": jid,
            "input_url": f"/api/jobs/{jid}/original",
            "pca_url": f"/api/jobs/{jid}/pca"}


@app.get("/api/jobs/{jid}/events")
async def events(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)

    async def gen():
        last = -1
        while True:
            with JOBS_LOCK:
                job = JOBS.get(jid)
                snap = dict(job) if job else None
            if snap and snap["rev"] != last:
                last = snap["rev"]
                payload = {k: snap[k] for k in
                           ("status", "stage", "progress", "message", "error", "result")}
                yield f"data: {json.dumps(payload)}\n\n"
                if snap["status"] in ("done", "error"):
                    break
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _find_input(jid: str) -> Path | None:
    jd = JOBS_DIR / jid
    if not jd.exists():
        return None
    for p in jd.glob("input.*"):
        return p
    return None


@app.get("/api/jobs/{jid}/original")
def original(jid: str):
    p = _find_input(jid)
    if not p or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/jobs/{jid}/pca")
def pca(jid: str):
    p = JOBS_DIR / jid / "pca.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
