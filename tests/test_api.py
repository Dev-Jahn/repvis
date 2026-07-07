"""API tests. The fast tests need no GPU; the full pipeline test is opt-in:

    uv run pytest                      # API tests only
    REPVIS_TEST_GPU=1 uv run pytest    # + full joint-PCA run on GPU
"""
import asyncio
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
import pytest

# Redirect sources/ + runs/ away from the repo before repvis.config is imported.
os.environ.setdefault("REPVIS_DATA_DIR", tempfile.mkdtemp(prefix="repvis-test-"))

from fastapi.testclient import TestClient  # noqa: E402

from fastapi import HTTPException, UploadFile  # noqa: E402

import repvis.server as srv  # noqa: E402
from repvis import pipeline as pl  # noqa: E402
from repvis.config import RUNS_DIR, SOURCES_DIR  # noqa: E402

client = TestClient(srv.app)

GPU = pytest.mark.skipif(os.environ.get("REPVIS_TEST_GPU") != "1",
                         reason="set REPVIS_TEST_GPU=1 to run the full GPU pipeline test")


def _make_clip(path, src):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src,
                    "-t", "1", "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    a, b = d / "a.mp4", d / "b.mp4"
    _make_clip(a, "testsrc2=size=192x108:rate=8")
    _make_clip(b, "mandelbrot=size=192x108:rate=8")
    return a, b


def _upload(p):
    with p.open("rb") as f:
        r = client.post("/api/sources", files={"file": (p.name, f, "video/mp4")})
    r.raise_for_status()
    return r.json()["id"]


def test_empty_upload_rejected():
    r = client.post("/api/sources", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
    assert r.status_code == 400


def test_bad_ids_are_404():
    assert client.get("/api/sources/..%2f..%2fetc/video").status_code == 404
    assert client.get("/api/runs/notahexid/pca").status_code == 404
    assert client.delete("/api/sources/notahexid").status_code == 404


def test_upload_dedup_and_listing(clips):
    a, b = clips
    sa, sb = _upload(a), _upload(b)
    assert sa != sb
    assert _upload(a) == sa                      # same bytes -> same source
    ids = {s["id"] for s in client.get("/api/sources").json()["sources"]}
    assert {sa, sb} <= ids
    assert client.get(f"/api/sources/{sa}/video").status_code == 200


def test_run_validation(clips):
    sa = _upload(clips[0])
    assert client.post("/api/runs", json={"source_ids": [], "model": "dinov2-base"}).status_code == 400
    assert client.post("/api/runs", json={"source_ids": [sa], "model": "nope"}).status_code == 400
    assert client.post("/api/runs", json={"source_ids": ["0" * 16], "model": "dinov2-base"}).status_code == 404


def test_delete_source(tmp_path):
    p = tmp_path / "solo.mp4"
    _make_clip(p, "testsrc=size=160x90:rate=8")   # unique content, unused by other tests
    sid = _upload(p)
    assert client.delete(f"/api/sources/{sid}").json()["ok"] is True
    assert client.get(f"/api/sources/{sid}/video").status_code == 404
    assert sid not in {s["id"] for s in client.get("/api/sources").json()["sources"]}


OPTS = {"max_frames": 8, "fps": 8, "max_side": 126, "remove_bg": False, "l2norm": False}


def _run_and_wait(source_ids, model="dinov2-base", timeout=300):
    r = client.post("/api/runs", json={"source_ids": source_ids, "model": model, "opts": OPTS})
    r.raise_for_status()
    body = r.json()
    deadline = time.time() + timeout
    while time.time() < deadline:
        g = srv.GROUPS[body["group_id"]]
        if g["status"] in ("done", "error"):
            assert g["status"] == "done", f"group failed: {g.get('error')}"
            return body
        time.sleep(0.5)
    raise TimeoutError("run did not finish")


@GPU
def test_full_joint_run_and_persistence(clips, monkeypatch):
    sa, sb = _upload(clips[0]), _upload(clips[1])
    body = _run_and_wait([sa, sb])
    rids = [x["run_id"] for x in body["runs"]]
    assert len(rids) == 2
    for rid in rids:
        d = RUNS_DIR / rid
        assert (d / "pca.mp4").exists() and (d / "meta.json").exists()
        # every run persists features + PCA state (+ SAM masks) for later re-seg/refit
        names = {p.name for p in d.iterdir()}
        assert {"pca.mp4", "meta.json", "feats.f16", "state.pt"} <= names
        assert names <= {"pca.mp4", "meta.json", "feats.f16", "state.pt", "masks.u1"}
        assert client.get(f"/api/runs/{rid}/pca").status_code == 200

    ws = client.get("/api/workspace").json()
    assert set(rids) <= {r["run_id"] for r in ws["runs"]}

    # SAM2 foreground: seg meta, mask sanity, deterministic re-decode, re-seg from
    # a 4-tuple point, point validation, SAM-failure (no clobber), auto reset + refit
    r0 = rids[0]
    meta = json.loads((RUNS_DIR / r0 / "meta.json").read_text())
    assert all(k in meta for k in ("grid", "frames", "fps", "frame_indices", "seg"))
    assert len(meta["frame_indices"]) == meta["frames"]

    # the initial auto-seg produced a usable, *partial* foreground mask
    res = meta["result"]
    m0 = pl._load_masks(RUNS_DIR / r0, meta["frames"], res["height"], res["width"])
    assert 0.0 < float(m0.float().mean()) < 1.0
    assert meta["seg"]["available"] is True

    # frame_indices re-decode is deterministic (masks align 1:1 with feats/video)
    sp = pl._source_path(meta["source_id"])
    f1 = pl._decode_source_frames(sp, meta["frame_indices"])
    f2 = pl._decode_source_frames(sp, meta["frame_indices"])
    assert len(f1) == len(f2) == meta["frames"]
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))

    # re-segment from a 4-tuple point [x, y, label, frame]
    seg = client.post(f"/api/runs/{r0}/segment", json={"points": [[96, 54, 1, 0]]})
    assert seg.status_code == 200 and seg.json()["ok"]

    # point validation — 400 on malformed shape, 422 on out-of-run bounds
    def _seg(pts):
        return client.post(f"/api/runs/{r0}/segment", json={"points": pts}).status_code
    # NaN can't ride httpx's JSON encoder (allow_nan=False) — send a raw body; the
    # server's stdlib json.loads accepts the literal, then math.isfinite rejects it.
    nan_r = client.post(f"/api/runs/{r0}/segment", content='{"points": [[NaN, 0, 1, 0]]}',
                        headers={"Content-Type": "application/json"})
    assert nan_r.status_code == 400                      # non-finite coordinate
    assert _seg([[10, 10, 2, 0]]) == 400                 # label not in {0, 1}
    assert _seg([[1, 2]]) == 400                         # wrong arity
    assert client.post(f"/api/runs/{r0}/segment", json={"points": "x"}).status_code == 400
    assert _seg([[1e12, 0, 1, 0]]) == 422                # x outside frame bounds
    assert _seg([[10, 10, 1, 999]]) == 422               # frame index out of range

    # a SAM failure on a refine must NOT clobber the existing artifacts (5xx, bytes intact)
    before_masks = (RUNS_DIR / r0 / "masks.u1").read_bytes()
    before_pca = (RUNS_DIR / r0 / "pca.mp4").read_bytes()

    def _boom(*a, **k):
        raise RuntimeError("sam down")

    monkeypatch.setattr("repvis.pipeline.sam.segment_session", _boom)  # re-seg path
    fail = client.post(f"/api/runs/{r0}/segment", json={"points": [[96, 54, 1, 0]]})
    assert 500 <= fail.status_code < 600
    assert (RUNS_DIR / r0 / "masks.u1").read_bytes() == before_masks
    assert (RUNS_DIR / r0 / "pca.mp4").read_bytes() == before_pca
    monkeypatch.undo()

    assert client.post(f"/api/runs/{r0}/segment", json={"points": []}).status_code == 200   # auto reset
    assert client.post(f"/api/runs/{r0}/refit", json={}).status_code == 200                 # no threshold
    assert client.get(f"/api/runs/{r0}/pca").status_code == 200

    # a re-run of the same (source, model) supersedes the old result on disk
    old = next(x["run_id"] for x in body["runs"] if x["source_id"] == sa)
    new = _run_and_wait([sa])["runs"][0]["run_id"]
    assert (RUNS_DIR / new / "meta.json").exists()
    assert not (RUNS_DIR / old).exists()
    ws = client.get("/api/workspace").json()
    pairs = [(r["source_id"], r["model"]) for r in ws["runs"]]
    assert len(pairs) == len(set(pairs))          # one run per matrix cell

    # Clear removes every completed result
    assert client.delete("/api/runs").json()["ok"] is True
    assert client.get("/api/workspace").json()["runs"] == []
    assert not any(d.is_dir() for d in RUNS_DIR.glob("*"))


# --------------------------------------------------------- delete/mutation race
def _fake_source(sid):
    sd = SOURCES_DIR / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "video.mp4").write_bytes(b"\x00")
    (sd / "meta.json").write_text(json.dumps({"name": sid, "ext": ".mp4", "size": 1}))
    srv.SOURCES[sid] = {"id": sid, "name": sid, "ext": ".mp4", "size": 1}


def _fake_run(sid, rid, model="dinov2-base", created=1.0):
    d = RUNS_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    for name in ("feats.f16", "state.pt", "masks.u1", "pca.mp4"):
        (d / name).write_bytes(b"\x00")
    (d / "meta.json").write_text(json.dumps({
        "run_id": rid, "source_id": sid, "model": model, "created": created,
        "result": {"width": 16, "height": 16}, "grid": [4, 4],
        "frames": 1, "fps": 8.0, "frame_indices": [0], "seg": {"available": True}}))
    return d


def _cleanup(sid, *rids):
    for rid in rids:
        shutil.rmtree(RUNS_DIR / rid, ignore_errors=True)
    srv.SOURCES.pop(sid, None)
    shutil.rmtree(SOURCES_DIR / sid, ignore_errors=True)


def _race_delete_vs_segment(monkeypatch, rid, destroy):
    """Force the stale-snapshot interleaving: pause `destroy()` at its FIRST rmtree,
    then fire /segment on `rid`. Returns dict(stub_ran, dir_present, seg_status).
    Invariant the fix must uphold: the segment stub never runs against a dir the
    delete then removes."""
    at_seam = threading.Event()
    seam_go = threading.Event()
    started = threading.Event()
    release = threading.Event()
    st = {"stub_ran": False, "dir_present": None, "seg_status": None}

    def _stub(run_dir, points, emit=None):
        st["stub_ran"] = True
        started.set()
        release.wait(5)
        st["dir_present"] = (run_dir / "feats.f16").exists()
        return {"available": True, "empty": False, "points": []}

    monkeypatch.setattr(srv, "segment_and_render", _stub)

    real_rmtree = srv.shutil.rmtree

    def _gated_rmtree(path, *a, **k):
        if not at_seam.is_set():
            at_seam.set()
            seam_go.wait(5)
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(srv.shutil, "rmtree", _gated_rmtree)

    def _destroy():
        try:
            destroy()
        except HTTPException:
            pass

    dth = threading.Thread(target=_destroy)
    dth.start()
    assert at_seam.wait(5), "destroy never reached its rmtree"

    def _segment():
        try:
            srv.run_segment(rid, {"points": [[1, 1, 1, 0]]})
        except HTTPException as e:
            st["seg_status"] = e.status_code

    sth = threading.Thread(target=_segment)
    sth.start()
    started.wait(1.0)          # buggy path reaches the stub fast; the fix blocks on LOCK -> times out
    seam_go.set()
    dth.join(5)
    release.set()
    sth.join(5)
    assert not dth.is_alive() and not sth.is_alive()
    return st


def _assert_no_clobber(st):
    # either the stub never ran (segment cleanly 404'd after the dir was removed) OR
    # it ran and the dir was still present the whole time. NEVER: stub ran + dir gone.
    if st["stub_ran"]:
        assert st["dir_present"] is True, "segment ran against a deleted run dir (race)"
    else:
        assert st["seg_status"] == 404


def test_race_delete_runs_vs_segment(monkeypatch):
    sid, rid = "a" * 16, "a" * 12
    _fake_source(sid)
    _fake_run(sid, rid)
    try:
        _assert_no_clobber(_race_delete_vs_segment(monkeypatch, rid, lambda: srv.delete_runs()))
    finally:
        _cleanup(sid, rid)


def test_race_delete_source_vs_segment(monkeypatch):
    sid, rid = "b" * 16, "b" * 12
    _fake_source(sid)
    _fake_run(sid, rid)
    try:
        _assert_no_clobber(_race_delete_vs_segment(monkeypatch, rid, lambda: srv.delete_source(sid)))
    finally:
        _cleanup(sid, rid)


def test_race_supersede_vs_segment(monkeypatch):
    sid, old, new = "c" * 16, "c" * 12, "d" * 12
    _fake_source(sid)
    _fake_run(sid, old, created=1.0)
    _fake_run(sid, new, created=2.0)          # the freshly-completed run that supersedes `old`
    meta = {"run_id": new, "source_id": sid, "model": "dinov2-base",
            "result": {"width": 16, "height": 16}, "created": 2.0}
    try:
        _assert_no_clobber(_race_delete_vs_segment(monkeypatch, old, lambda: srv._persist_run(dict(meta))))
    finally:
        _cleanup(sid, old, new)


def test_refit_404s_when_run_dir_absent(monkeypatch):
    # sibling guard for the /refit path (feats.f16 check now inside the lock):
    # a refit on a nonexistent run dir must 404, not proceed.
    rid = "e" * 12
    try:
        srv.run_refit(rid)
        assert False, "expected 404"
    except HTTPException as e:
        assert e.status_code == 404


def test_race_dup_upload_vs_delete_source(monkeypatch):
    """Phantom-source barrier: a dup (identical-bytes) upload must not register sid
    into SOURCES after a concurrent delete_source popped it and rmtree'd the dir.
    Force the buggy interleaving: pause delete at its source-dir rmtree (holding
    LOCK), then fire the dup upload so its dedup-check sees the still-present dir.
    Invariant the fix upholds: SOURCES has sid IFF its dir exists -- never a phantom
    entry pointing at a missing dir."""
    data = b"upload-delete-phantom-regression\x00\x01\x02"
    sid = hashlib.sha256(data).hexdigest()[:16]
    # a prior identical upload already exists on disk + in the registry
    sd = SOURCES_DIR / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "video.mp4").write_bytes(data)
    (sd / "meta.json").write_text(json.dumps({"name": sid, "ext": ".mp4", "size": len(data)}))
    srv.SOURCES[sid] = {"id": sid, "name": sid, "ext": ".mp4", "size": len(data)}

    at_seam = threading.Event()
    seam_go = threading.Event()
    real_rmtree = srv.shutil.rmtree

    def _gated_rmtree(path, *a, **k):
        if str(path) == str(SOURCES_DIR / sid) and not at_seam.is_set():
            at_seam.set()
            seam_go.wait(5)
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(srv.shutil, "rmtree", _gated_rmtree)

    def _delete():
        try:
            srv.delete_source(sid)
        except HTTPException:
            pass

    dth = threading.Thread(target=_delete)
    dth.start()
    assert at_seam.wait(5), "delete never reached its source-dir rmtree"

    # delete is paused mid-rmtree while HOLDING LOCK; fire the dup upload.
    def _upload():
        uf = UploadFile(filename="dup.mp4", file=io.BytesIO(data))
        try:
            asyncio.run(srv.upload_source(uf))
        except HTTPException:
            pass

    uth = threading.Thread(target=_upload)
    uth.start()
    time.sleep(0.3)            # let the upload reach (and, under the fix, block on) LOCK
    seam_go.set()
    dth.join(5)
    uth.join(5)
    assert not dth.is_alive() and not uth.is_alive()

    has_entry = sid in srv.SOURCES
    has_dir = (SOURCES_DIR / sid).is_dir()
    _cleanup(sid)
    assert has_entry == has_dir, f"phantom source: SOURCES={has_entry} dir={has_dir}"


# ------------------------------------------------------- shared-token access control
TOKEN = "s3cr3t-test-token"


@pytest.fixture
def auth_on():
    """Enforce the shared token for the duration of a test, then restore open mode
    (default) so the other tests are unaffected."""
    prev = srv.AUTH_TOKEN
    srv.AUTH_TOKEN = TOKEN
    try:
        yield TOKEN
    finally:
        srv.AUTH_TOKEN = prev


def test_auth_blocks_without_credentials(auth_on):
    sid, rid = "f" * 16, "f" * 12
    _fake_source(sid)
    _fake_run(sid, rid)
    c = TestClient(srv.app)   # isolated cookie jar (no repvis_token yet)
    try:
        # every protected route (API, source video, run PCA, run POST) is 401
        assert c.get("/api/workspace").status_code == 401
        assert c.get("/api/sources").status_code == 401
        assert c.get(f"/api/sources/{sid}/video").status_code == 401
        assert c.get(f"/api/runs/{rid}/pca").status_code == 401
        assert c.post("/api/runs", json={"source_ids": [sid], "model": "dinov2-base"}).status_code == 401
        # the login page + auth-probe stay reachable so the UI can bootstrap
        assert c.get("/").status_code == 200
        assert c.get("/api/auth").json() == {"required": True}

        # a valid Authorization: Bearer header authenticates
        bearer = {"Authorization": f"Bearer {TOKEN}"}
        assert c.get("/api/workspace", headers=bearer).status_code == 200
        assert c.get(f"/api/sources/{sid}/video", headers=bearer).status_code == 200
        assert c.get(f"/api/runs/{rid}/pca", headers=bearer).status_code == 200
        # so does the X-Repvis-Token header
        assert c.get("/api/sources", headers={"X-Repvis-Token": TOKEN}).status_code == 200
        # a wrong token is still rejected
        assert c.get("/api/sources", headers={"X-Repvis-Token": "nope"}).status_code == 401
        assert c.get("/api/sources", headers={"Authorization": "Bearer nope"}).status_code == 401
    finally:
        _cleanup(sid, rid)


def test_login_cookie_flow(auth_on):
    sid, rid = "f" * 16, "f" * 12
    _fake_source(sid)
    _fake_run(sid, rid)
    c = TestClient(srv.app)
    try:
        # wrong token -> 401, no cookie
        assert c.post("/api/login", json={"token": "wrong"}).status_code == 401
        assert "repvis_token" not in c.cookies
        # correct token -> 200 and an httpOnly, samesite=strict, path=/ cookie
        r = c.post("/api/login", json={"token": TOKEN})
        assert r.status_code == 200 and r.json()["ok"] is True
        sc = r.headers["set-cookie"].lower()
        assert "repvis_token=" in sc and "httponly" in sc and "samesite=strict" in sc and "path=/" in sc
        # the client now carries the cookie -> media + API authenticate automatically
        assert "repvis_token" in c.cookies
        assert c.get("/api/workspace").status_code == 200
        assert c.get(f"/api/sources/{sid}/video").status_code == 200
        assert c.get(f"/api/runs/{rid}/pca").status_code == 200
    finally:
        _cleanup(sid, rid)


def test_auth_disabled_is_open():
    prev = srv.AUTH_TOKEN
    srv.AUTH_TOKEN = None            # disabled (the default for the whole suite)
    try:
        c = TestClient(srv.app)
        assert c.get("/api/sources").status_code == 200
        assert c.get("/api/auth").json() == {"required": False}
        assert c.post("/api/login", json={"token": "anything"}).status_code == 200   # no-op
    finally:
        srv.AUTH_TOKEN = prev
