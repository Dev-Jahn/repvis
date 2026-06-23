"""End-to-end job, streamed in bounded chunks so peak GPU memory is O(one chunk).

Phase 1: decode -> extract features per chunk -> offload to CPU, gathering a
         capped token subsample for the PCA fit.
Phase 2: fit PCA once (temporally consistent colors), then project + render +
         encode each cached chunk, freeing it as we go.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import REGISTRY, STREAM_CHUNK, proc_hw, select_devices
from .extract import extract_features
from .pca import fit_pca_state, project_chunk
from .video_io import VideoSource, encode_rgb_frames

_FIT_MAX = 300_000  # max tokens used to fit the PCA basis


def _even(v: int) -> int:
    v = int(v)
    return max(2, v - (v % 2))


def run_job(job_dir: Path, input_path: Path, model_key: str, opts: dict, emit):
    spec = REGISTRY[model_key]

    emit(stage="decoding", progress=0.02, message="Opening video…")
    src = VideoSource(input_path,
                      target_fps=float(opts.get("fps", 24.0)),
                      max_frames=int(opts.get("max_frames", 900)))
    n, meta = src.n, src.meta
    if n == 0:
        raise RuntimeError("no frames decoded")

    devices = select_devices()
    primary = devices[0]
    max_side = int(opts.get("max_side") or 0) or spec.max_side
    proc_h, proc_w, gh, gw = proc_hw(meta["height"], meta["width"], spec.patch, max_side)

    chunk = max(1, STREAM_CHUNK)
    if spec.family == "vjepa":  # align chunk to clip length
        chunk = max(spec.chunk_frames, (chunk // spec.chunk_frames) * spec.chunk_frames)
    n_chunks = math.ceil(n / chunk)
    per_chunk_quota = max(1, math.ceil(_FIT_MAX / n_chunks))

    ndev = len(devices)
    emit(stage="extracting", progress=0.05,
         message=(f"{n} frames · {meta['width']}×{meta['height']} → proc {proc_w}×{proc_h} "
                  f"(grid {gw}×{gh}) · {spec.label} · {ndev} GPU(s) · streaming ×{n_chunks}"))

    # ---- Phase 1: extract per chunk, offload to CPU, gather PCA fit sample ----
    feat_chunks: list[torch.Tensor] = []   # CPU fp16 (k, gh, gw, D)
    fit_parts: list[torch.Tensor] = []     # on `primary`
    bs = int(opts.get("batch_size", 32))
    done = 0
    for _start, frames in src.iter_chunks(chunk):
        cf = extract_features(spec, frames, devices, (proc_h, proc_w), (gh, gw), bs, lambda _f: None)
        flat = cf.reshape(-1, cf.shape[-1])
        take = min(flat.shape[0], per_chunk_quota)
        sel = torch.randperm(flat.shape[0], device=flat.device)[:take]
        fit_parts.append(flat[sel].float())
        feat_chunks.append(cf.to("cpu"))
        done += int(frames.shape[0])
        emit(stage="extracting", progress=0.05 + 0.55 * done / n,
             message=f"Extracting dense features… {int(100 * done / n)}%  ({done}/{n})")
        del cf, flat, sel
    src.close()

    emit(stage="pca", progress=0.62, message="Fitting PCA basis…")
    fit_buf = torch.cat(fit_parts, 0)
    del fit_parts
    state = fit_pca_state(fit_buf, remove_bg=bool(opts.get("remove_bg", False)),
                          l2norm=bool(opts.get("l2norm", False))).to(primary)
    del fit_buf
    if primary.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- Phase 2: project + render + encode each cached chunk ----
    ow, oh = _even(meta["width"]), _even(meta["height"])
    out_path = job_dir / "pca.mp4"
    emit(stage="encoding", progress=0.65, message="Rendering & encoding PCA video…")

    def gen():
        rendered = 0
        while feat_chunks:
            cf = feat_chunks.pop(0).to(primary)
            rgb = project_chunk(cf, state).permute(0, 3, 1, 2)  # (k,3,gh,gw)
            k = rgb.shape[0]
            for j in range(0, k, 24):
                up = F.interpolate(rgb[j:j + 24], size=(oh, ow),
                                   mode="bilinear", align_corners=False)
                up = (up.clamp(0, 1) * 255).round().to(torch.uint8)
                up = up.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                for f in range(up.shape[0]):
                    yield up[f]
            rendered += k
            emit(stage="encoding", progress=0.65 + 0.33 * rendered / n,
                 message=f"Encoding PCA video… {int(100 * rendered / n)}%")
            del cf, rgb

    # encoders/containers mishandle sub-1 fps; clamp and let the player sync
    # proportionally by time-fraction (durations need not match).
    enc_fps = max(1.0, meta["fps_out"])
    encode_rgb_frames(gen(), out_path, ow, oh, enc_fps)

    emit(stage="done", progress=1.0, message="Done",
         result={"width": ow, "height": oh, "frames": n,
                 "src_fps": round(meta["src_fps"], 2),
                 "out_fps": round(enc_fps, 2),
                 "grid": f"{gw}×{gh}", "proc": f"{proc_w}×{proc_h}",
                 "gpus": ndev, "chunks": n_chunks})
