"""End-to-end group processing, fully pipelined so the GPU never idles.

A "group" is one or more source videos processed together under ONE shared PCA
basis (joint PCA: same color = same feature direction across videos; a single
source is a group of one).

Work is split into *units* — contiguous segments of a source's sampled frames —
bin-packed across the available GPUs. Each unit runs a self-contained phase-1
pipeline on its device:

    [decode thread]  NVDEC -> uint8 RGB chunks on the GPU   (queue, depth 2)
    [worker thread]  preprocess + model forward (bf16)      -> fp16 features
    [offload thread] features -> host RAM cache             (queue, depth 2)

so decode, compute and D2H run concurrently. After the shared PCA fit, phase 2
renders each source on its own GPU:

    [prefetch thread] host cache -> GPU                     (queue, depth 2)
    [render thread]   project -> upsample -> RGB
    [NvencSink]       RGB->NV12 on GPU -> pinned D2H -> ffmpeg h264_nvenc
                      (csc/copy/encode overlap the render loop internally)

Peak GPU memory is O(one chunk); host RAM holds the fp16 feature cache
(bounded by max_frames per source).
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import sam
from .config import (REGISTRY, SOURCES_DIR, STREAM_CHUNK, proc_hw,
                     select_devices)
from .extract import extract_unit_chunk, warm_model
from .pca import fit_pca_state, project_chunk, refit_display
from .video_io import GpuEncoder, GpuVideoSource, compute_indices, probe_video

_FIT_MAX = 300_000   # max tokens used to fit the PCA basis (pooled across sources)
_MIN_SEG = 192       # don't split a source into segments smaller than this


def _even(v: int) -> int:
    v = int(v)
    return max(2, v - (v % 2))


class _Cancel(Exception):
    pass


class _Group:
    """Shared mutable state for one run_group call."""

    def __init__(self, emit, total_frames: int):
        self.emit = emit
        self.total = total_frames
        self.lock = threading.Lock()
        self.done_extract = 0
        self.done_render = 0
        self.error: BaseException | None = None
        self.cancel = threading.Event()

    def fail(self, e: BaseException):
        with self.lock:
            if self.error is None:
                self.error = e
        self.cancel.set()

    def check(self):
        if self.cancel.is_set():
            raise _Cancel()

    def tick_extract(self, k: int):
        with self.lock:
            self.done_extract += k
            d = self.done_extract
        self.emit(stage="extracting", progress=0.05 + 0.50 * d / self.total,
                  message=f"Extracting dense features… {int(100 * d / self.total)}%  ({d}/{self.total})")

    def tick_render(self, k: int):
        with self.lock:
            self.done_render += k
            d = self.done_render
        self.emit(stage="encoding", progress=0.62 + 0.36 * d / self.total,
                  message=f"Rendering PCA video… {int(100 * d / self.total)}%")


def _plan_units(sources: list[dict], ndev: int, align: int) -> list[dict]:
    """Split sources into contiguous index segments and bin-pack them onto
    devices (largest first, least-loaded device), so decode+extract scale
    across GPUs even for a single long video."""
    units = []
    for s in sources:
        n = len(s["indices"])
        # Several units per device: NVDEC sessions run in parallel (the card
        # has multiple decode engines), and unit B's decode overlaps unit A's
        # extraction on the same device. Segments align to the model's clip
        # length so V-JEPA never pads mid-video.
        nseg = max(1, min(math.ceil(4 * ndev / len(sources)), n // _MIN_SEG))
        per = -(-math.ceil(n / nseg) // align) * align
        for k in range(0, n, per):
            units.append({"src": s, "lo": k, "hi": min(n, k + per)})
    load = [0] * ndev
    for u in sorted(units, key=lambda u: u["lo"] - u["hi"]):   # largest first
        d = min(range(ndev), key=lambda i: load[i])
        u["dev_idx"] = d
        load[d] += u["hi"] - u["lo"]
    return units


# ---- cancellation-safe queue plumbing --------------------------------------
# Every producer/consumer handoff below is bounded and cancel-aware: a thread
# blocked on a full/empty queue must ALWAYS wake up when the group is cancelled
# (g.fail from any thread) or its stop Event fires, so one thread's death can
# never wedge another forever. This is the invariant that keeps the pipeline
# deadlock-free — see _cput/_cget/_drain.
_CANCELLED = object()


def _cput(q: queue.Queue, item, g: _Group, *stops: threading.Event) -> bool:
    """Bounded put that yields to cancellation. True if enqueued, False if we
    bailed because the group was cancelled or a stop fired."""
    while not (g.cancel.is_set() or any(s.is_set() for s in stops)):
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _cget(q: queue.Queue, g: _Group):
    """Bounded get that yields to cancellation. Returns _CANCELLED if the group
    was cancelled while the queue stayed empty."""
    while True:
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            if g.cancel.is_set():
                return _CANCELLED


def _drain(q: queue.Queue):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


# ------------------------------------------------------------------- phase 1
# Stream discipline (measured, not theoretical): torchcodec's CUDA decoder
# synchronizes against the DEFAULT stream — any compute queued there starves
# decode by >10x. So the default stream is reserved for torchcodec, model
# compute runs on a side stream, and the feature D2H on another. Handoffs
# carry a CUDA event (consumers wait on the event, never the device) and
# cross-stream tensors are record_stream()-ed so the caching allocator
# doesn't recycle them early.

def _start_decode(unit: dict, dev: str, chunk: int, align: int, g: _Group):
    """Kick off a unit's NVDEC decode thread (bounded queue -> self-throttled).

    All of a device's units decode concurrently: the GPU has multiple NVDEC
    engines, and unit B's decode overlaps unit A's extraction.
    """
    src = unit["src"]
    indices = src["indices"][unit["lo"]:unit["hi"]]
    vs = GpuVideoSource(src["input"], dev, indices)
    dq: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()

    def decode_loop():   # default stream: torchcodec only
        try:
            torch.cuda.set_device(dev)
            for _i, frames in vs.iter_chunks(chunk, align):
                if stop.is_set() or g.cancel.is_set():
                    break
                ev = torch.cuda.Event()
                ev.record()
                if not _cput(dq, (frames, ev), g, stop):
                    break
        except BaseException as e:  # noqa: BLE001
            g.fail(e)
        finally:
            _cput(dq, None, g, stop)

    dt = threading.Thread(target=decode_loop, daemon=True)
    dt.start()
    unit["_decode"] = (dt, vs, stop)
    unit["_dq"] = dq
    unit["cache"] = []


def _stop_decode(unit: dict):
    """Stop + join a unit's decode thread if it's still around (idempotent)."""
    dec = unit.get("_decode")
    if not dec:
        return
    dt, vs, stop = dec
    stop.set()
    dq = unit.get("_dq")
    if dq is not None:
        _drain(dq)              # unblock a decode thread parked on a full queue
    dt.join()
    vs.close()
    unit["_decode"] = unit["_dq"] = None


def _offload_loop(oq: queue.Queue, dev: str, g: _Group):
    """Shared per-device D2H worker: (unit, feats, event) -> pinned host cache."""
    try:
        torch.cuda.set_device(dev)
        s_off = torch.cuda.Stream()
        while True:
            item = _cget(oq, g)
            if item is None or item is _CANCELLED:
                return
            unit, feats, ev = item
            s_off.wait_event(ev)                     # after the producer kernels
            feats.record_stream(s_off)
            with torch.cuda.stream(s_off):
                pinned = torch.empty(feats.shape, dtype=feats.dtype, pin_memory=True)
                pinned.copy_(feats, non_blocking=True)
            s_off.synchronize()                      # feats stays referenced until here
            unit["cache"].append(pinned)
            del feats
    except BaseException as e:  # noqa: BLE001
        g.fail(e)


def _extract_unit(unit: dict, spec, dev: str, chunk: int, bs: int, g: _Group, oq: queue.Queue):
    """Drain a unit's decode queue through the model; hand features to the
    device's offload worker."""
    dq = unit["_dq"]
    fit_parts: list[torch.Tensor] = []   # PCA-fit token samples
    n_chunks = math.ceil((unit["hi"] - unit["lo"]) / chunk)
    quota = max(1, math.ceil(unit["fit_quota"] / max(n_chunks, 1)))
    s_cmp = torch.cuda.Stream(device=dev)
    try:
        with torch.cuda.stream(s_cmp):
            while True:
                item = _cget(dq, g)
                if item is None or item is _CANCELLED:
                    break
                frames, ev = item
                s_cmp.wait_event(ev)                     # decode done for this chunk
                frames.record_stream(s_cmp)
                feats = extract_unit_chunk(spec, frames, dev, unit["proc"], unit["grid"], bs)
                flat = feats.reshape(-1, feats.shape[-1])
                take = min(flat.shape[0], quota)
                sel = torch.randperm(flat.shape[0], device=flat.device)[:take]
                fit_parts.append(flat[sel].float())
                fev = torch.cuda.Event()
                fev.record(s_cmp)
                if not _cput(oq, (unit, feats, fev), g):   # offload dead/cancelled
                    break
                g.tick_extract(int(frames.shape[0]))
                del frames, feats, flat, sel
    finally:
        _stop_decode(unit)      # always stop+join this unit's decoder
    g.check()
    with torch.cuda.stream(s_cmp):
        unit["fit"] = torch.cat(fit_parts, 0).to("cpu") if fit_parts else None
    s_cmp.synchronize()


# ----------------------------------------------------- SAM2 foreground masks
def _source_path(source_id: str) -> Path:
    """Resolve a source_id back to its on-disk video (SOURCES_DIR/<id>/video.*)."""
    vids = sorted((SOURCES_DIR / source_id).glob("video.*"))
    if not vids:
        raise FileNotFoundError(f"source video missing for {source_id}")
    return vids[0]


def _decode_source_frames(path, indices) -> list[np.ndarray]:
    """Decode EXACTLY `indices` from `path` (torchcodec, approximate seek) as a
    list of (H, W, 3) uint8 RGB np arrays at SOURCE resolution, frame order —
    so SAM masks align 1:1 with feats.f16 / the encoded video."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(path), device="cpu", seek_mode="approximate")
    data = dec.get_frames_at(indices=[int(i) for i in indices]).data  # (T,3,H,W) uint8
    arr = data.permute(0, 2, 3, 1).contiguous().numpy()               # (T,H,W,3)
    return [arr[i] for i in range(arr.shape[0])]


def _auto_seed(grid0: torch.Tensor, width: int, height: int) -> list[tuple[float, float, int, int]]:
    """DINO-saliency auto seed: from a frame-0 grid feature (gh, gw, D), pick the
    patch whose feature is farthest (L2) from the patch mean and map its centre to
    a source pixel. Returns a single positive point [(x, y, 1, 0)] on seed frame 0."""
    gh, gw, D = grid0.shape
    x = grid0.reshape(-1, D).float()
    sal = (x - x.mean(0)).norm(dim=1)          # (gh*gw,)
    row, col = divmod(int(sal.argmax()), gw)
    px = (col + 0.5) / gw * float(width)
    py = (row + 0.5) / gh * float(height)
    return [(px, py, 1, 0)]


def _resize_masks(m: torch.Tensor, oh: int, ow: int) -> torch.Tensor:
    """(T, H, W) bool -> (T, oh, ow) bool via nearest resize (identity if already)."""
    if m.shape[1] == oh and m.shape[2] == ow:
        return m.bool()
    return F.interpolate(m.float()[:, None], size=(oh, ow), mode="nearest")[:, 0] > 0.5


def _seg_meta(points: list, *, available: bool, error, empty: bool) -> dict:
    """Build the C2 seg meta object. `points` are 4-tuples (x, y, label, frame)."""
    return {"available": bool(available), "error": error, "empty": bool(empty),
            "seed_frame": 0,
            "points": [[float(x), float(y), int(lab), int(fr)] for (x, y, lab, fr) in points]}


def _run_sam(frames_np: list, points: list, dev: str, oh: int, ow: int) -> tuple[torch.Tensor, bool]:
    """Run SAM2 from per-frame `points` over `frames_np` and resize to (oh, ow).
    RAISES on SAM failure (callers decide whether to swallow). Returns
    ((T, oh, ow) bool CPU mask, empty) where empty is True iff no foreground."""
    raw = sam.segment(frames_np, points, dev)          # (T, H, W) bool
    masks = _resize_masks(raw, oh, ow)
    return masks, not bool(masks.any())


def _segment(frames_np: list, points: list, dev: str, oh: int, ow: int) -> tuple[torch.Tensor, dict]:
    """INITIAL-run segmentation: run SAM2 and build the C2 seg meta. On a SAM hard
    error, fall back to an all-foreground mask (available=False, error=<msg>) so the
    video still renders; an empty result is available=True, empty=True."""
    T = len(frames_np)
    try:
        masks, empty = _run_sam(frames_np, points, dev, oh, ow)
        seg = _seg_meta(points, available=True, error=None, empty=empty)
    except Exception as e:  # noqa: BLE001
        masks = torch.ones(T, oh, ow, dtype=torch.bool)
        seg = _seg_meta(points, available=False, error=str(e), empty=False)
    return masks, seg


def _save_masks(run_dir: Path, masks: torch.Tensor):
    """Persist (T, oh, ow) bool masks as packed bits (masks.u1)."""
    np.packbits(masks.numpy().reshape(-1)).tofile(str(run_dir / "masks.u1"))


def _load_masks(run_dir: Path, T: int, oh: int, ow: int) -> torch.Tensor:
    """Load masks.u1 back into a (T, oh, ow) bool CPU tensor."""
    raw = np.fromfile(str(run_dir / "masks.u1"), dtype=np.uint8)
    bits = np.unpackbits(raw, count=T * oh * ow).reshape(T, oh, ow)
    return torch.from_numpy(bits.astype(bool))


def _mk_tick(emit, total: int, label: str):
    """Progress callback for the re-render helpers (or None when emit is None)."""
    if emit is None:
        return None
    prog = [0]

    def tick(k):
        prog[0] += k
        emit(stage="encoding", progress=prog[0] / max(total, 1),
             message=f"{label} {int(100 * prog[0] / max(total, 1))}%")

    return tick


# ------------------------------------------------------------------- phase 2
def _bake_encode(chunks: list, state, masks: torch.Tensor, out_path, ow: int, oh: int,
                 fps: float, dev: str, g: _Group, tick=None):
    """Prefetch host feature chunks -> GPU, project with `state` -> rgb, upsample
    to (oh, ow), MULTIPLY IN the per-frame SAM mask (resized nearest to (oh, ow)),
    then NVENC-encode to `out_path`. The single reusable "feats + state + masks ->
    baked mp4" encoder, shared by run_group, segment_and_render and refit_and_render.

    `chunks` : list of CPU fp16 (k, gh, gw, D) tensors in frame order; each is
               freed (set to None) once copied to the GPU.
    `masks`  : (T, H, W) bool CPU tensor in frame order (True = foreground); its
               per-frame slice is baked onto each rendered frame.
    `tick(k)`: optional progress callback, rendered-frame count per chunk.

    Same stream discipline as phase 1: render math on a side stream, the default
    stream stays reserved for torchcodec (NVENC submit).
    """
    pq: queue.Queue = queue.Queue(maxsize=2)
    stop_pre = threading.Event()

    def prefetch_loop():   # host -> GPU, own stream
        try:
            torch.cuda.set_device(dev)
            s_pre = torch.cuda.Stream()
            with torch.cuda.stream(s_pre):
                for i, c in enumerate(chunks):
                    if stop_pre.is_set() or g.cancel.is_set():
                        break
                    gpu = c.to(dev, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(s_pre)
                    if not _cput(pq, (gpu, ev), g, stop_pre):
                        break
                    chunks[i] = None
        except BaseException as e:  # noqa: BLE001
            g.fail(e)
        finally:
            _cput(pq, None, g, stop_pre)

    pt = threading.Thread(target=prefetch_loop, daemon=True)
    s_r = torch.cuda.Stream(device=dev)
    default = torch.cuda.default_stream(dev)
    sink = None
    pt_started = False
    base = 0
    try:
        # NOTE: state.to(dev) and GpuEncoder() must be inside the try so the
        # finally always runs (either can raise: H2D OOM / NVENC init failure).
        st = state.to(dev)
        sink = GpuEncoder(out_path, ow, oh, fps, dev)
        pt.start()
        pt_started = True
        while True:
            item = _cget(pq, g)
            if item is _CANCELLED:
                raise _Cancel()
            if item is None:
                break
            cf, ev = item
            s_r.wait_event(ev)
            with torch.cuda.stream(s_r):
                cf.record_stream(s_r)
                rgb = project_chunk(cf, st).permute(0, 3, 1, 2)   # (k,3,gh,gw) float
                k = rgb.shape[0]
                mchunk = masks[base:base + k].to(dev).float()     # (k,H,W)
                for j in range(0, k, 24):
                    up = F.interpolate(rgb[j:j + 24], size=(oh, ow), mode="bilinear", align_corners=False)
                    mb = F.interpolate(mchunk[j:j + 24, None], size=(oh, ow), mode="nearest")  # (b,1,oh,ow)
                    up = up * mb                              # bake the foreground mask
                    up = (up.clamp_(0, 1) * 255).round_().to(torch.uint8).contiguous()
                    rev = torch.cuda.Event()
                    rev.record(s_r)
                    default.wait_event(rev)               # encoder consumes on default
                    up.record_stream(default)
                    sink.submit(up)
            if tick is not None:
                tick(k)
            base += k
            del cf, rgb, mchunk
        sink.close()
    except BaseException:
        if sink is not None:
            sink.abort()
        raise
    finally:
        stop_pre.set()
        _drain(pq)               # unblock prefetch parked on a full queue
        if pt_started:
            pt.join()


def _render_source(src: dict, state, dev: str, g: _Group, emit_done):
    """Persist this source's dense features + PCA state, auto-seed + run SAM2
    foreground segmentation, then encode the PCA video with the mask baked in."""
    torch.cuda.set_device(dev)
    meta = src["meta"]
    ow, oh = _even(meta["width"]), _even(meta["height"])
    run_dir = src["out"].parent
    chunks = [c for u in src["units"] for c in u["cache"]]   # units are in order
    for u in src["units"]:
        u["cache"] = None

    # full dense features -> disk (fp16, C-contiguous, frame order) for later refit
    with open(run_dir / "feats.f16", "wb") as fh:
        for c in chunks:
            fh.write(c.numpy().tobytes())

    # auto-seed from frame-0 grid saliency, decode source frames, run SAM2
    points = _auto_seed(chunks[0][0], meta["width"], meta["height"])
    frames_np = _decode_source_frames(src["input"], src["indices"])
    masks, seg = _segment(frames_np, points, dev, oh, ow)
    del frames_np

    _bake_encode(chunks, state, masks, src["out"], ow, oh, src["fps_out"], dev, g,
                 tick=g.tick_render)
    _save_masks(run_dir, masks)
    torch.save(state.to("cpu"), run_dir / "state.pt")
    emit_done(src, seg)


# ---------------------------------------------------------------------- main
def run_group(items: list[dict], model_key: str, opts: dict, emit):
    """Process a group of sources with one shared PCA basis.

    `items`: list of {"run_id", "source_id", "input": Path, "out": Path}.
    """
    spec = REGISTRY[model_key]
    devices = select_devices()
    ndev = len(devices)
    max_side = int(opts.get("max_side") or 0) or spec.max_side
    fps = float(opts.get("fps") or 24.0)
    max_frames = max(1, int(opts.get("max_frames") or 900))

    chunk = max(1, STREAM_CHUNK)
    if spec.family == "vjepa":   # align chunk to clip length
        chunk = max(spec.chunk_frames, (chunk // spec.chunk_frames) * spec.chunk_frames)

    # ---- probe sources, plan work ----
    emit(stage="decoding", progress=0.02, message="Opening video(s)…")
    sources = []
    total_n = 0
    for it in items:
        meta = probe_video(it["input"])
        indices = compute_indices(meta["n_total"], meta["src_fps"], fps, max_frames)
        if not indices:
            raise RuntimeError(f"no frames to decode ({it['source_id']})")
        ph, pw, gh, gw = proc_hw(meta["height"], meta["width"], spec.patch, max_side)
        sources.append({"it": it, "input": it["input"], "out": it["out"], "meta": meta,
                        "indices": indices, "grid": (gh, gw), "proc": (ph, pw),
                        "fps_out": len(indices) / meta["duration"] if meta["duration"] > 0 else meta["src_fps"],
                        "units": []})
        total_n += len(indices)

    align = spec.chunk_frames if spec.family == "vjepa" else 1
    units = _plan_units(sources, ndev, align)
    per_source_fit = max(1, _FIT_MAX // len(sources))
    for u in units:
        s = u["src"]
        u["proc"], u["grid"] = s["proc"], s["grid"]
        u["fit_quota"] = max(1, per_source_fit * (u["hi"] - u["lo"]) // len(s["indices"]))
        s["units"].append(u)
    for s in sources:
        s["units"].sort(key=lambda u: u["lo"])

    bs = int(opts.get("batch_size") or 0) or spec.batch_size
    multi = len(sources) > 1
    emit(stage="extracting", progress=0.05,
         message=(f"{len(sources)} source(s) · {total_n} frames · {spec.label} · {ndev} GPU(s)"
                  + (" · shared (joint) PCA" if multi else "")))

    g = _Group(emit, total_n)

    # ---- warm models on every device that has work (parallel, once) ----
    # both the feature extractor and SAM2 (used in phase-2 segmentation).
    used = sorted({devices[u["dev_idx"]] for u in units})
    threads = [threading.Thread(target=warm_model, args=(spec, d), daemon=True) for d in used]
    threads += [threading.Thread(target=sam.warm_model, args=(d,), daemon=True) for d in used]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ---- phase 1: per-device workers — decode all units ahead, extract in order ----
    def dev_worker(di: int):
        dev = devices[di]
        mine = [u for u in units if u["dev_idx"] == di]
        oq: queue.Queue = queue.Queue(maxsize=2)
        ot = threading.Thread(target=_offload_loop, args=(oq, dev, g), daemon=True)
        ot_started = False
        try:
            torch.cuda.set_device(dev)
            ot.start()
            ot_started = True
            for u in mine:
                _start_decode(u, dev, chunk, align, g)
            for u in mine:
                g.check()
                _extract_unit(u, spec, dev, chunk, bs, g, oq)
        except _Cancel:
            pass
        except BaseException as e:  # noqa: BLE001
            g.fail(e)
        finally:
            for u in mine:                  # stop+join any decoder _extract_unit didn't reach
                _stop_decode(u)
            if ot_started:
                if not g.cancel.is_set():    # clean shutdown; on cancel offload exits via _cget
                    _cput(oq, None, g)
                ot.join()

    workers = [threading.Thread(target=dev_worker, args=(i,), daemon=True)
               for i in range(ndev) if any(u["dev_idx"] == i for u in units)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    if g.error:
        raise g.error

    # ---- shared PCA fit (equal per-source contribution) ----
    emit(stage="pca", progress=0.58,
         message="Fitting shared PCA basis…" if multi else "Fitting PCA basis…")
    primary = devices[0]
    src_fit = []   # pooled fit tokens per source
    for s in sources:
        parts = [u["fit"] for u in s["units"] if u["fit"] is not None]
        for u in s["units"]:
            u["fit"] = None
        src_fit.append(torch.cat(parts, 0))
    if len(src_fit) > 1:   # a long source must not out-weight a short one
        m = min(int(t.shape[0]) for t in src_fit)
        src_fit = [t[torch.randperm(t.shape[0])[:m]] for t in src_fit]
    fit_buf = torch.cat(src_fit, 0).to(primary)
    del src_fit
    state = fit_pca_state(fit_buf, l2norm=bool(opts.get("l2norm", False)))
    del fit_buf

    # ---- phase 2: render sources in parallel across devices ----
    emit(stage="encoding", progress=0.62, message="Rendering PCA video…")

    def emit_done(src, seg):
        meta, it = src["meta"], src["it"]
        gh, gw = src["grid"]
        ph, pw = src["proc"]
        emit(run_id=it["run_id"], run_status="done",
             result={"width": _even(meta["width"]), "height": _even(meta["height"]),
                     "frames": len(src["indices"]),
                     "src_fps": round(meta["src_fps"], 2), "out_fps": round(max(1.0, src["fps_out"]), 2),
                     "grid": f"{gw}×{gh}", "proc": f"{pw}×{ph}", "gpus": ndev},
             # lifted to meta top-level by the server (see segment/refit + server persist)
             run_meta={"grid": [gh, gw], "feat_dim": int(state.mean.shape[0]),
                       "frames": len(src["indices"]), "fps": float(src["fps_out"]),
                       "frame_indices": [int(i) for i in src["indices"]], "seg": seg})

    def render_worker(src, dev):
        try:
            _render_source(src, state, dev, g, emit_done)
        except _Cancel:
            pass
        except BaseException as e:  # noqa: BLE001
            g.fail(e)

    rthreads = []
    for i, s in enumerate(sources):
        t = threading.Thread(target=render_worker, args=(s, devices[i % ndev]), daemon=True)
        rthreads.append(t)
        t.start()
    for t in rthreads:
        t.join()
    if g.error:
        raise g.error

    emit(stage="done", progress=1.0, message="Done")


def _load_run(run_dir: Path):
    """Shared loader for the live segment/refit paths: parse meta, load the dense
    features + PCA state, pick a device. Raises FileNotFoundError if feats.f16 is
    missing (-> 404 upstream)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    gh, gw = int(meta["grid"][0]), int(meta["grid"][1])
    D, T = int(meta["feat_dim"]), int(meta["frames"])
    fps = float(meta["fps"])
    res = meta["result"]
    ow, oh = int(res["width"]), int(res["height"])

    feats_p = run_dir / "feats.f16"
    if not feats_p.exists():
        raise FileNotFoundError(str(feats_p))
    feats = torch.from_numpy(
        np.fromfile(str(feats_p), dtype=np.float16).reshape(T, gh, gw, D))
    state = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=False)

    dev = select_devices()[0]
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
    return meta, feats, state, dev, (gh, gw, D, T, fps, ow, oh)


def _replace_video(fn, tmp: Path, dst: Path):
    """Encode into `tmp` (a real .mp4 so torchcodec infers the container), then
    atomically os.replace it onto `dst`; clean up the temp on failure."""
    try:
        fn(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)


# ------------------------------------------------------------ live segment / refit
def segment_and_render(run_dir: Path, points: list, emit=None) -> dict:
    """Re-segment a run from client point prompts and re-bake pca.mp4.

    `points`: list of [x, y, label, frame] (label 1 = foreground/+, 0 = background/-)
    in SOURCE pixel coords, `frame` the sampled-frame index the click was placed on.
    EMPTY -> recompute the DINO-saliency auto-seed from the persisted frame-0 grid
    features. The source's exact sampled frame_indices are re-decoded so the SAM
    masks align 1:1 with feats.f16 / the video. On a SAM failure this RAISES and
    leaves pca.mp4 / masks.u1 / meta['seg'] untouched (never a partial write).
    Out-of-bounds / non-finite points raise ValueError. Persists masks.u1 +
    meta['seg'], atomically replaces pca.mp4, returns meta['seg'].
    """
    run_dir = Path(run_dir)
    meta, feats, state, dev, (gh, gw, D, T, fps, ow, oh) = _load_run(run_dir)

    frames_np = _decode_source_frames(_source_path(meta["source_id"]), meta["frame_indices"])
    H, W = frames_np[0].shape[:2]
    pts = []
    for p in points:   # bounds-check client points against this run's W/H/T
        x, y, lab, fr = float(p[0]), float(p[1]), int(p[2]), int(p[3])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"non-finite point coord ({x}, {y})")
        if not (0.0 <= x < W and 0.0 <= y < H):
            raise ValueError(f"point ({x}, {y}) out of frame bounds {W}x{H}")
        if not (0 <= fr < T):
            raise ValueError(f"point frame {fr} out of range [0, {T})")
        pts.append((x, y, lab, fr))
    if not pts:   # reset -> recompute the auto-seed from persisted feats frame-0 grid
        pts = _auto_seed(feats[0], W, H)
    # SAM failure propagates here (no fallback) so we never clobber the existing run.
    masks, empty = _run_sam(frames_np, pts, dev, oh, ow)
    seg = _seg_meta(pts, available=True, error=None, empty=empty)
    del frames_np

    chunks = [feats[i:i + STREAM_CHUNK].contiguous() for i in range(0, T, STREAM_CHUNK)]
    del feats
    g = _Group(lambda **kw: None, max(T, 1))
    _replace_video(
        lambda tmp: _bake_encode(chunks, state, masks, tmp, ow, oh, fps, dev, g,
                                 tick=_mk_tick(emit, T, "Segmenting…")),
        run_dir / "pca.seg.mp4", run_dir / "pca.mp4")

    _save_masks(run_dir, masks)
    meta["seg"] = seg
    # atomic replace: a concurrent delete_source reads meta.json via _run_record to
    # decide whether a run derives from the source, and a torn (truncated) read there
    # would drop this in-flight run from that check and delete its source out from under us.
    meta_tmp = run_dir / "meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta))
    os.replace(meta_tmp, run_dir / "meta.json")
    return meta["seg"]


def refit_and_render(run_dir: Path, emit=None) -> dict:
    """Re-fit the display basis over the current mask's foreground grid tokens and
    re-bake pca.mp4 with the SAME masks (no threshold). Overwrites state.pt and
    returns meta['seg'] unchanged. Runs on a GPU chosen via select_devices."""
    run_dir = Path(run_dir)
    meta, feats, state, dev, (gh, gw, D, T, fps, ow, oh) = _load_run(run_dir)
    masks = _load_masks(run_dir, T, oh, ow)

    # downsample the pixel masks to the feature grid -> per-token fg FRACTION in
    # [0,1], used as soft weights (no hard >0.5 gate) so thin structures survive
    # proportionally. Keep every token with any foreground; drop only w == 0.
    grid_frac = F.adaptive_avg_pool2d(masks.float().unsqueeze(1), (gh, gw)).squeeze(1)
    w = grid_frac.reshape(-1)                       # (T*gh*gw,)
    keep = w > 0
    fg_tokens = feats.reshape(-1, D)[keep]
    fg_w = w[keep]
    if fg_tokens.shape[0] < 16:
        raise ValueError("mask leaves too little foreground to refit")
    if fg_tokens.shape[0] > _FIT_MAX:
        sel = torch.randperm(fg_tokens.shape[0])[:_FIT_MAX]
        fg_tokens, fg_w = fg_tokens[sel], fg_w[sel]
    new_state = refit_display(fg_tokens.to(dev), state.to(dev), weights=fg_w.to(dev))

    chunks = [feats[i:i + STREAM_CHUNK].contiguous() for i in range(0, T, STREAM_CHUNK)]
    del feats
    g = _Group(lambda **kw: None, max(T, 1))
    _replace_video(
        lambda tmp: _bake_encode(chunks, new_state, masks, tmp, ow, oh, fps, dev, g,
                                 tick=_mk_tick(emit, T, "Re-rendering…")),
        run_dir / "pca.refit.mp4", run_dir / "pca.mp4")

    torch.save(new_state.to("cpu"), run_dir / "state.pt")
    return meta.get("seg", {"available": False})
