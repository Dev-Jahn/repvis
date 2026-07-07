"""API tests. The fast tests need no GPU; the full pipeline test is opt-in:

    uv run pytest                      # API tests only
    REPVIS_TEST_GPU=1 uv run pytest    # + full joint-PCA run on GPU
"""
import io
import json
import os
import subprocess
import tempfile
import time

import pytest

# Redirect sources/ + runs/ away from the repo before repvis.config is imported.
os.environ.setdefault("REPVIS_DATA_DIR", tempfile.mkdtemp(prefix="repvis-test-"))

from fastapi.testclient import TestClient  # noqa: E402

import repvis.server as srv  # noqa: E402
from repvis.config import RUNS_DIR  # noqa: E402

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
def test_full_joint_run_and_persistence(clips):
    sa, sb = _upload(clips[0]), _upload(clips[1])
    body = _run_and_wait([sa, sb])
    rids = [x["run_id"] for x in body["runs"]]
    assert len(rids) == 2
    for rid in rids:
        d = RUNS_DIR / rid
        assert (d / "pca.mp4").exists() and (d / "meta.json").exists()
        # every run now persists its features + PCA state for later refit; bgscore
        # only when the fg/bg split was non-degenerate.
        names = {p.name for p in d.iterdir()}
        assert {"pca.mp4", "meta.json", "feats.f16", "state.pt"} <= names
        assert names <= {"pca.mp4", "meta.json", "feats.f16", "state.pt", "bgscore.u8"}
        assert client.get(f"/api/runs/{rid}/pca").status_code == 200

    ws = client.get("/api/workspace").json()
    assert set(rids) <= {r["run_id"] for r in ws["runs"]}

    # per-cell background threshold + refit (reuses persisted features, no re-extract)
    r0 = rids[0]
    meta = json.loads((RUNS_DIR / r0 / "meta.json").read_text())
    assert all(k in meta for k in ("grid", "frames", "fps", "bg"))
    if meta["bg"]["available"]:
        assert client.get(f"/api/runs/{r0}/bgscore").status_code == 200
        rf = client.post(f"/api/runs/{r0}/refit", json={"threshold": 0.9})  # high -> ample fg
        assert rf.status_code == 200 and rf.json()["ok"]
        assert client.get(f"/api/runs/{r0}/pca").status_code == 200
    else:
        assert client.get(f"/api/runs/{r0}/bgscore").status_code == 404
    assert client.post(f"/api/runs/{r0}/threshold", json={"threshold": 0.5}).status_code == 200
    assert client.post(f"/api/runs/{r0}/refit", json={}).status_code == 400   # missing threshold

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
