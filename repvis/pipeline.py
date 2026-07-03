"""End-to-end job, streamed in bounded chunks so peak GPU memory is O(one chunk).

A "group" is one or more source videos processed together. With >1 source the
PCA basis is fit once over the *pooled* token sample from every source (equal
per-source contribution), so the same color means the same feature direction
*across* videos — that's the cross-video semantic comparison. A single source is
just a group of one (identical to per-video PCA).

Phase 1: for each source, decode -> extract features per chunk -> spill each
         chunk to disk (the run dir), gathering a capped token subsample into
         one shared fit buffer. Host RAM stays O(one chunk) even for joint runs.
Phase 2: fit the shared PCA once (temporally + cross-video consistent colors),
         then stream the spilled chunks back to project + render + encode each
         source's own video, deleting each spill file as it is consumed.
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

_FIT_MAX = 300_000  # max tokens used to fit the PCA basis (pooled across sources)


def _even(v: int) -> int:
    v = int(v)
    return max(2, v - (v % 2))


def run_group(items: list[dict], model_key: str, opts: dict, emit):
    """Process a group of sources with one shared PCA basis.

    `items`: list of {"run_id", "source_id", "input": Path, "out": Path}.
    `emit`: group-level progress; per-source completion is reported via
            emit(run_id=..., run_status="done", result={...}).
    """
    spec = REGISTRY[model_key]
    devices = select_devices()
    primary = devices[0]
    ndev = len(devices)
    max_side = int(opts.get("max_side") or 0) or spec.max_side
    bs = int(opts.get("batch_size", 32))
    fps = float(opts.get("fps") or 24.0)
    max_frames = max(1, int(opts.get("max_frames") or 900))

    chunk = max(1, STREAM_CHUNK)
    if spec.family == "vjepa":  # align chunk to clip length
        chunk = max(spec.chunk_frames, (chunk // spec.chunk_frames) * spec.chunk_frames)

    nsrc = len(items)
    per_source_fit = max(1, _FIT_MAX // nsrc)  # equal contribution to the shared basis

    # Decode every source up front so we know total frame count for progress.
    emit(stage="decoding", progress=0.02, message="Opening video(s)…")
    srcs = []
    total_n = 0
    for it in items:
        s = VideoSource(it["input"], target_fps=fps, max_frames=max_frames)
        if s.n == 0:
            s.close()
            raise RuntimeError(f"no frames decoded ({it['source_id']})")
        srcs.append((it, s))
        total_n += s.n

    multi = nsrc > 1
    emit(stage="extracting", progress=0.05,
         message=(f"{nsrc} source(s) · {total_n} frames · {spec.label} · {ndev} GPU(s)"
                  + (" · shared (joint) PCA" if multi else "")))

    # ---- Phase 1: extract every source, spill features to disk, pool fit sample ----
    # The shared basis must see every source before any projection, so per-source
    # feature chunks are spilled to the run dir (fp16) instead of held in host RAM.
    # RAM stays O(one chunk); disk usage is transient and freed as phase 2 consumes.
    per_src: list[dict] = []
    src_fit: list[torch.Tensor] = []      # one pooled (mi, D) fit sample per source
    done_frames = 0
    for it, s in srcs:
        meta = s.meta
        proc_h, proc_w, gh, gw = proc_hw(meta["height"], meta["width"], spec.patch, max_side)
        n = s.n
        n_chunks = math.ceil(n / chunk)
        per_chunk_quota = max(1, math.ceil(per_source_fit / n_chunks))
        spill_dir: Path = it["out"].parent
        feat_paths: list[Path] = []           # spilled fp16 (k, gh, gw, D) chunks
        parts: list[torch.Tensor] = []
        for _start, frames in s.iter_chunks(chunk):
            cf = extract_features(spec, frames, devices, (proc_h, proc_w), (gh, gw), bs, lambda _f: None)
            flat = cf.reshape(-1, cf.shape[-1])
            take = min(flat.shape[0], per_chunk_quota)
            sel = torch.randperm(flat.shape[0], device=flat.device)[:take]
            parts.append(flat[sel].float())
            p = spill_dir / f"feat_{len(feat_paths):04d}.pt"
            torch.save(cf.to("cpu"), p)
            feat_paths.append(p)
            done_frames += int(frames.shape[0])
            emit(stage="extracting", progress=0.05 + 0.55 * done_frames / total_n,
                 message=f"Extracting dense features… {int(100 * done_frames / total_n)}%  ({done_frames}/{total_n})")
            del cf, flat, sel
        s.close()
        src_fit.append(torch.cat(parts, 0))
        per_src.append({"it": it, "meta": meta, "grid": (gw, gh), "proc": (proc_w, proc_h),
                        "n": n, "feat_paths": feat_paths})

    # ---- Fit ONE PCA basis over the pooled sample (shared colors across sources) ----
    emit(stage="pca", progress=0.62,
         message="Fitting shared PCA basis…" if multi else "Fitting PCA basis…")
    # Enforce equal per-source contribution: a small/short source must not be
    # out-weighted by a long one, or the shared basis (and thus cross-video colors)
    # would be dominated by the larger source. Subsample every source down to the
    # smallest available token count before pooling.
    if len(src_fit) > 1:
        m = min(int(t.shape[0]) for t in src_fit)
        src_fit = [t[torch.randperm(t.shape[0], device=t.device)[:m]] for t in src_fit]
    fit_buf = torch.cat(src_fit, 0).to(primary)
    del src_fit
    state = fit_pca_state(fit_buf, remove_bg=bool(opts.get("remove_bg", False)),
                          l2norm=bool(opts.get("l2norm", False))).to(primary)
    del fit_buf
    if primary.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- Phase 2: project + render + encode each source with the shared basis ----
    rendered_total = 0
    for item in per_src:
        it, meta = item["it"], item["meta"]
        gw, gh = item["grid"]; pw, ph = item["proc"]; n = item["n"]
        feat_paths = item["feat_paths"]
        ow, oh = _even(meta["width"]), _even(meta["height"])

        def gen(feat_paths=feat_paths, ow=ow, oh=oh):
            nonlocal rendered_total
            for p in feat_paths:
                cf = torch.load(p, map_location=primary, weights_only=True)
                p.unlink()
                rgb = project_chunk(cf, state).permute(0, 3, 1, 2)  # (k,3,gh,gw)
                k = rgb.shape[0]
                for j in range(0, k, 24):
                    up = F.interpolate(rgb[j:j + 24], size=(oh, ow),
                                       mode="bilinear", align_corners=False)
                    up = (up.clamp(0, 1) * 255).round().to(torch.uint8)
                    up = up.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    for f in range(up.shape[0]):
                        yield up[f]
                rendered_total += k
                emit(stage="encoding", progress=0.65 + 0.33 * rendered_total / total_n,
                     message=f"Encoding PCA video… {int(100 * rendered_total / total_n)}%")
                del cf, rgb

        # encoders/containers mishandle sub-1 fps; clamp and let the player sync
        # proportionally by time-fraction (durations need not match).
        enc_fps = max(1.0, meta["fps_out"])
        encode_rgb_frames(gen(), it["out"], ow, oh, enc_fps)
        emit(run_id=it["run_id"], run_status="done",
             result={"width": ow, "height": oh, "frames": n,
                     "src_fps": round(meta["src_fps"], 2), "out_fps": round(enc_fps, 2),
                     "grid": f"{gw}×{gh}", "proc": f"{pw}×{ph}", "gpus": ndev})

    emit(stage="done", progress=1.0, message="Done")
