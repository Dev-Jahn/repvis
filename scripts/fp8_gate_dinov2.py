#!/usr/bin/env python
"""GPU-side FP8 fidelity gate for spike/fp8-attention (Phase 0 + Phase 1).

Companion to the offline harness in ``scripts/fp8_fidelity.py``: this is the part
that must run on the GPU box. It drives ``dinov2-base`` through repvis' OWN
extract path (``repvis.extract._Extractor``) on identical clips under four numeric
configs, then feeds the dumped grid features through the REAL PCA render path
(``repvis.pca.fit_pca_state`` / ``project_chunk`` / ``refit_display``) so the color
metrics reflect what the user actually sees.

Configs (one model on the GPU at a time; features copied to host between them):

  * ``bf16_a`` / ``bf16_b`` — the server's bf16 eager SDPA load, run twice
    (self-noise floor; identical in-process because the forward is deterministic).
  * ``fp16``  — same weights cast to fp16 (a dtype repvis already treats as
    interchangeable with bf16 in ``_pre``); this is the Phase-0 *free-variance*
    floor the fp8 candidate is calibrated against.
  * ``fp8``   — torchao full-FP8: ``quantize_(model,
    Float8DynamicActivationFloat8WeightConfig())`` over every ``nn.Linear``
    (qkv/proj + MLP), EXACTLY the deployment candidate the backbone bench timed
    at 1.4-1.65x. LayerNorm and the softmax stay bf16. This is the Phase-1 probe.

For each clip and each (base, candidate) pair it reports the three gate metrics
(per-token cosine, top-3 PCA subspace principal angle, rendered-RGB delta-E) on
raw features, through a FIXED bf16 basis (isolates feature error), through each
config's OWN refit basis (end-to-end user-visible), on the few-token
``refit_display`` foreground path, and a temporal frame-to-frame flicker delta.

Run (needs torchao, GPU):
    CUDA_VISIBLE_DEVICES=5 HF_HOME=/var/cache/huggingface \
        uv run --with torchao python scripts/fp8_gate_dinov2.py

Add ``--video PATH:SECONDS`` (repeatable) to include real footage; without it the
gate runs on two lavfi clips (mandelbrot, testsrc2). The reported spike numbers
used two real clips (1080p + 720p) plus those two lavfi clips.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repvis.config import REGISTRY, proc_hw           # noqa: E402
from repvis.extract import _Extractor, flush_vram     # noqa: E402
from repvis import pca                                 # noqa: E402
import fp8_fidelity as fid                             # noqa: E402

DEV = "cuda:0"   # CUDA_VISIBLE_DEVICES already selects the physical GPU
NFR = 8          # consecutive frames per clip (gives the temporal-flicker delta)
PATCH, MAX_SIDE = 14, 1024


def _make_frames(kind: str, src: str, ss: float, w: int, h: int, raw: Path) -> torch.Tensor:
    """(NFR, 3, H, W) uint8 CUDA tensor of consecutive frames via ffmpeg rawvideo."""
    if not raw.exists():
        if kind == "real":
            cmd = ["ffmpeg", "-y", "-ss", str(ss), "-i", src, "-frames:v", str(NFR),
                   "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw)]
        else:
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-frames:v", str(NFR),
                   "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw)]
        subprocess.run(cmd, check=True, capture_output=True)
    buf = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(NFR, h, w, 3).copy()
    return torch.from_numpy(buf).permute(0, 3, 1, 2).contiguous().to(DEV)


def _build(cfg: str):
    """Fresh dinov2-base extractor under the requested numeric config."""
    spec = REGISTRY["dinov2-base"]
    ext = _Extractor(spec, DEV)                       # eager bf16, the server load
    if cfg == "fp16":
        ext.model = ext.model.half()
        ext.in_dtype = torch.float16
        return ext, spec, "fp16"
    if cfg == "fp8":
        from torchao.quantization import (quantize_,
                                          Float8DynamicActivationFloat8WeightConfig)
        quantize_(ext.model, Float8DynamicActivationFloat8WeightConfig())
        return ext, spec, "torchao-Float8DynamicActivationFloat8Weight"
    return ext, spec, "bf16"


def _dump(clips: list[dict]) -> tuple[dict, dict]:
    """Extract (T,gh,gw,D) float grid features per config per clip (host numpy)."""
    frames, grids = {}, {}
    for cl in clips:
        frames[cl["name"]] = _make_frames(cl["kind"], cl["src"], cl["ss"],
                                          cl["w"], cl["h"], cl["raw"])
        ph, pw, gh, gw = proc_hw(cl["h"], cl["w"], PATCH, MAX_SIDE)
        grids[cl["name"]] = ((ph, pw), (gh, gw))
    feats: dict = {}
    for cfg in ["bf16_a", "bf16_b", "fp16", "fp8"]:
        ext, spec, status = _build("bf16" if cfg.startswith("bf16") else cfg)
        print(f"[{cfg}] {status}", flush=True)
        feats[cfg] = {"_status": status}
        for name, fr in frames.items():
            proc, grid = grids[name]
            out = ext.process(fr, proc, grid, spec.batch_size)
            feats[cfg][name] = out.float().cpu().numpy()
        del ext
        flush_vram()
    return feats, grids


def _render(feat_base: np.ndarray, feat_cand: np.ndarray) -> dict:
    """Run the real pca.py render and gather every color/subspace metric."""
    tb = torch.from_numpy(feat_base).to(DEV)
    tc = torch.from_numpy(feat_cand).to(DEV)
    T, gh, gw, D = tb.shape
    st_b = pca.fit_pca_state(tb.reshape(-1, D))
    st_c = pca.fit_pca_state(tc.reshape(-1, D))
    rgb_b = pca.project_chunk(tb, st_b).cpu().numpy()          # base, own basis
    rgb_c_shared = pca.project_chunk(tc, st_b).cpu().numpy()   # cand thru BASE basis
    rgb_c_own = pca.project_chunk(tc, st_c).cpu().numpy()      # cand, own basis

    r0, r1, c0, c1 = gh // 4, gh - gh // 4, gw // 4, gw - gw // 4   # central pseudo-fg
    fg_b = tb[:, r0:r1, c0:c1, :].reshape(-1, D)
    fg_c = tc[:, r0:r1, c0:c1, :].reshape(-1, D)
    rst_b = pca.refit_display(fg_b, st_b)
    rst_c = pca.refit_display(fg_c, st_c)
    rgb_rb = pca.project_chunk(tb, rst_b).cpu().numpy()[:, r0:r1, c0:c1, :]
    rgb_rc = pca.project_chunk(tc, rst_c).cpu().numpy()[:, r0:r1, c0:c1, :]

    def flick(rgb):
        d = [fid.color_delta_e(rgb[t], rgb[t + 1])["mean"] for t in range(rgb.shape[0] - 1)]
        return float(np.mean(d)) if d else 0.0

    return {
        "deltaE_shared_basis": fid.color_delta_e(rgb_b, rgb_c_shared),
        "deltaE_own_basis": fid.color_delta_e(rgb_b, rgb_c_own),
        "refit_subspace_deg": fid.subspace_principal_angles(
            fg_b.cpu().numpy(), fg_c.cpu().numpy())["max_deg"],
        "refit_deltaE": fid.color_delta_e(rgb_rb, rgb_rc),
        "flicker_base": flick(rgb_b), "flicker_cand": flick(rgb_c_own),
    }


def _compare(fb: np.ndarray, fc: np.ndarray) -> dict:
    return {"cosine": fid.feature_cosine(fb, fc),
            "subspace": fid.subspace_principal_angles(fb, fc),
            **_render(fb, fc)}


def _summary(report: dict, clips: list[str]) -> None:
    def worst(pair, path, agg):
        vals = []
        for c in clips:
            d = report["pairs"][pair][c]
            for k in path.split("."):
                d = d[k]
            vals.append(d)
        return max(vals) if agg == "max" else min(vals)

    print("\n" + "=" * 92)
    print("FP8 FIDELITY GATE — dinov2-base — worst over clips: " + ", ".join(clips))
    print("=" * 92)
    print(f"{'metric':<28}{'bf16-self':>13}{'fp16-floor':>13}{'fp8-cand':>13}{'doc gate':>12}")
    rows = [("cosine mean", "cosine.mean", "min", 0.9995),
            ("cosine p1", "cosine.p1", "min", 0.995),
            ("subspace max deg", "subspace.max_deg", "max", 2.0),
            ("refit subspace deg", "refit_subspace_deg", "max", 2.0),
            ("deltaE shared p95", "deltaE_shared_basis.p95", "max", 3.0),
            ("deltaE own p95", "deltaE_own_basis.p95", "max", 3.0),
            ("refit deltaE p95", "refit_deltaE.p95", "max", 3.0)]
    for label, path, agg, gate in rows:
        cells = [worst(p, path, agg) for p in ("bf16_self", "fp16_floor", "fp8_cand")]
        print(f"{label:<28}{cells[0]:>13.5f}{cells[1]:>13.5f}{cells[2]:>13.5f}{gate:>12.4f}")
    print("-" * 92)
    for p in ("fp16_floor", "fp8_cand"):
        print(f"temporal flicker {p:<11} base={worst(p,'flicker_base','max'):.3f}  "
              f"cand={worst(p,'flicker_cand','max'):.3f}  (mean frame-to-frame deltaE)")
    print("=" * 92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[],
                    help="PATH:SECONDS real clip (repeatable)")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="fp8gate_"))
    clips = []
    for v in args.video:
        path, _, ss = v.rpartition(":")
        w, h = _probe(path)
        clips.append({"name": Path(path).stem, "kind": "real", "src": path,
                      "ss": float(ss), "w": w, "h": h, "raw": tmp / f"{Path(path).stem}.raw"})
    for name, src, w, h in [("mandelbrot", "mandelbrot=size=1280x720:rate=25", 1280, 720),
                            ("testsrc2", "testsrc2=size=1280x720:rate=25", 1280, 720)]:
        clips.append({"name": name, "kind": "lavfi", "src": src, "ss": 0.0,
                      "w": w, "h": h, "raw": tmp / f"{name}.raw"})

    torch.manual_seed(0)
    feats, grids = _dump(clips)
    statuses = {k: feats[k].pop("_status") for k in list(feats)}
    names = [c["name"] for c in clips]
    pairs = [("bf16_self", "bf16_a", "bf16_b"),
             ("fp16_floor", "bf16_a", "fp16"),
             ("fp8_cand", "bf16_a", "fp8")]
    report = {"statuses": statuses, "pairs": {}}
    for pn, a, b in pairs:
        report["pairs"][pn] = {c: _compare(feats[a][c], feats[b][c]) for c in names}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    _summary(report, names)
    print("statuses:", statuses)


def _probe(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                         check=True, capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


if __name__ == "__main__":
    main()
