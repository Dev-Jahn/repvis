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

from .config import REGISTRY, STREAM_CHUNK, proc_hw, select_devices
from .extract import extract_unit_chunk, gray_field, warm_model
from .pca import (BgCtx, _has_mask, fit_pca_state, project_chunk, refit_display,
                  score_chunk)
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
    fit_parts: list[tuple[torch.Tensor, torch.Tensor]] = []   # (tokens, grid positions)
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
                npatch = unit["grid"][0] * unit["grid"][1]
                # keep each token's grid position: remove_bg debiases by position
                fit_parts.append((flat[sel].float(), (sel % npatch).to(torch.int32)))
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
        unit["fit"] = (torch.cat([t for t, _ in fit_parts], 0).to("cpu"),
                       torch.cat([p for _, p in fit_parts], 0).to("cpu")) if fit_parts else None
    s_cmp.synchronize()


# ------------------------------------------------------------------- phase 2
def _render_encode(chunks: list, state, out_path, ow: int, oh: int, fps: float,
                   dev: str, g: _Group, tick=None, score_out: list | None = None):
    """Prefetch host feature chunks -> GPU, project with `state` -> UNMASKED rgb,
    upsample to (oh, ow) and NVENC-encode to `out_path`. Shared by the normal
    phase-2 render and the live refit re-render.

    `chunks`   : list of CPU fp16 (k, gh, gw, D) tensors in frame order; each is
                 freed (set to None) once copied to the GPU.
    `tick(k)`  : optional progress callback, rendered-frame count per chunk.
    `score_out`: if a list, the per-chunk uint8 background score (k, gh, gw) is
                 appended in frame order (requires mask material in `state`).

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
                for j in range(0, k, 24):
                    up = F.interpolate(rgb[j:j + 24], size=(oh, ow), mode="bilinear", align_corners=False)
                    up = (up.clamp_(0, 1) * 255).round_().to(torch.uint8).contiguous()
                    rev = torch.cuda.Event()
                    rev.record(s_r)
                    default.wait_event(rev)               # encoder consumes on default
                    up.record_stream(default)
                    sink.submit(up)
                if score_out is not None:                 # per-patch bg score, same stream
                    sc = (score_chunk(cf, st) * 255.0).round_().clamp_(0, 255).to(torch.uint8)
                    score_out.append(sc.to("cpu"))        # blocking D2H after the score kernels
            if tick is not None:
                tick(k)
            del cf, rgb
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
    """Persist this source's dense features / bg score / PCA state, then encode
    the UNMASKED PCA video from its host feature cache."""
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

    has_mask = _has_mask(state)
    score_out: list | None = [] if has_mask else None
    _render_encode(chunks, state, src["out"], ow, oh, src["fps_out"], dev, g,
                   tick=g.tick_render, score_out=score_out)

    if score_out is not None:   # per-patch background score for the live slider
        with open(run_dir / "bgscore.u8", "wb") as fh:
            for s in score_out:
                fh.write(s.numpy().tobytes())
    torch.save(state.to("cpu"), run_dir / "state.pt")
    emit_done(src)


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
    used = sorted({devices[u["dev_idx"]] for u in units})
    threads = [threading.Thread(target=warm_model, args=(spec, d), daemon=True) for d in used]
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
    src_fit = []   # (tokens, positions) per source
    for s in sources:
        parts = [u["fit"] for u in s["units"] if u["fit"] is not None]
        for u in s["units"]:
            u["fit"] = None
        src_fit.append((torch.cat([t for t, _ in parts], 0),
                        torch.cat([p for _, p in parts], 0)))
    if len(src_fit) > 1:   # a long source must not out-weight a short one
        m = min(int(t.shape[0]) for t, _ in src_fit)
        sel = [torch.randperm(t.shape[0])[:m] for t, _ in src_fit]
        src_fit = [(t[i], p[i]) for (t, p), i in zip(src_fit, sel)]
    fit_buf = torch.cat([t for t, _ in src_fit], 0).to(primary)
    # Always compute mask material so every run can drive the live threshold
    # slider (masking is applied at display time, not baked into the video). If
    # the fg/bg split is degenerate, fit_pca_state leaves the state maskless
    # (_has_mask false) and the slider stays disabled downstream.
    fields = {}    # gray positional fields (one model forward per distinct grid, cached)
    for s in sources:
        gr = tuple(s["grid"])
        if gr not in fields:
            fields[gr] = gray_field(spec, primary, tuple(s["proc"]), gr)
    bg = BgCtx(
        pos=torch.cat([p for _, p in src_fit], 0).long().to(primary),
        grid_id=torch.cat([torch.full((t.shape[0],), si, dtype=torch.long)
                           for si, (t, _) in enumerate(src_fit)], 0).to(primary),
        grids=[tuple(s["grid"]) for s in sources], fields=fields)
    del src_fit
    state = fit_pca_state(fit_buf, remove_bg=True,
                          l2norm=bool(opts.get("l2norm", False)), bg=bg)
    del fit_buf, bg

    # ---- phase 2: render sources in parallel across devices ----
    emit(stage="encoding", progress=0.62, message="Rendering PCA video…")

    has_mask = _has_mask(state)
    t0 = float(state.bg_threshold)

    def emit_done(src):
        meta, it = src["meta"], src["it"]
        gh, gw = src["grid"]
        ph, pw = src["proc"]
        emit(run_id=it["run_id"], run_status="done",
             result={"width": _even(meta["width"]), "height": _even(meta["height"]),
                     "frames": len(src["indices"]),
                     "src_fps": round(meta["src_fps"], 2), "out_fps": round(max(1.0, src["fps_out"]), 2),
                     "grid": f"{gw}×{gh}", "proc": f"{pw}×{ph}", "gpus": ndev},
             # lifted to meta top-level by the server (see refit_and_render / server persist)
             run_meta={"grid": [gh, gw], "feat_dim": int(state.mean.shape[0]),
                       "frames": len(src["indices"]), "fps": float(src["fps_out"]),
                       "bg": {"available": has_mask, "threshold": t0,
                              "fitted_threshold": t0}})

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


# ------------------------------------------------------------------- live refit
def refit_and_render(run_dir: Path, threshold: float, emit=None) -> dict:
    """Re-fit ONLY the display basis over the current foreground (patches with
    background score < `threshold`) and re-render the UNMASKED PCA video from the
    persisted dense features, atomically replacing runs/<rid>/pca.mp4.

    Reuses the run's mask material (bg score is unchanged — only colors refit),
    overwrites state.pt with the new state, updates meta's bg thresholds, and
    returns the updated meta["bg"] dict. Runs on a GPU chosen via select_devices.
    """
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
    score = torch.from_numpy(
        np.fromfile(str(run_dir / "bgscore.u8"), dtype=np.uint8).reshape(T, gh, gw))
    state = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=False)

    dev = select_devices()[0]
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)

    # foreground tokens under the requested threshold, subsampled for the fit
    fg = (score.reshape(-1).float() / 255.0) < float(threshold)
    fg_tokens = feats.reshape(-1, D)[fg]
    if fg_tokens.shape[0] < 16:
        raise ValueError(f"threshold {float(threshold):.3f} leaves too little foreground to refit")
    if fg_tokens.shape[0] > _FIT_MAX:
        sel = torch.randperm(fg_tokens.shape[0])[:_FIT_MAX]
        fg_tokens = fg_tokens[sel]
    new_state = refit_display(fg_tokens.to(dev), state.to(dev))

    # re-render the UNMASKED video from ALL features with the new display basis
    chunks = [feats[i:i + STREAM_CHUNK].contiguous() for i in range(0, T, STREAM_CHUNK)]
    del feats
    g = _Group(lambda **kw: None, max(T, 1))
    tick = None
    if emit is not None:
        prog = [0]

        def tick(k):
            prog[0] += k
            emit(stage="encoding", progress=prog[0] / max(T, 1),
                 message=f"Re-rendering… {int(100 * prog[0] / max(T, 1))}%")

    tmp = run_dir / "pca.refit.mp4"   # must keep a real video extension (encoder infers container)
    try:
        _render_encode(chunks, new_state, tmp, ow, oh, fps, dev, g, tick=tick)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, run_dir / "pca.mp4")

    torch.save(new_state.to("cpu"), run_dir / "state.pt")
    meta["bg"]["fitted_threshold"] = float(threshold)
    meta["bg"]["threshold"] = float(threshold)
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return meta["bg"]
