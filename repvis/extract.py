"""Dense feature extraction (DINOv2/v3 per-frame, V-JEPA 2.1 spatio-temporal).

Frames are resized (aspect-preserving) to a per-video patch-multiple resolution,
so the dense grid stays close to the source resolution instead of a tiny square.
All heavy work runs in bf16 on GPU; long clips are sharded across every GPU.
"""
from __future__ import annotations

import gc
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import AutoModel

from .config import COMPILE, DEVICES, ModelSpec

_DINO_UNIT = 96  # frames per GPU shard threshold for image models


class _Extractor:
    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device

        if spec.family == "dino":
            # DINO is numerically clean in pure bf16.
            self.model = AutoModel.from_pretrained(
                spec.resolve_source(), dtype=torch.bfloat16, attn_implementation="sdpa",
            ).to(device).eval()
            cfg = self.model.config
            self.prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
            # DINOv2 uses learned position embeddings -> must interpolate for any
            # resolution other than its native one. (DINOv3 uses RoPE; no-op.)
            self._interp = "interpolate_pos_encoding" in inspect.signature(self.model.forward).parameters
            self.autocast = False
            self.in_dtype = torch.bfloat16
        else:
            # V-JEPA 2.1's custom RoPE upcasts q/k to fp32; keep fp32 weights and
            # get bf16 compute speed via autocast (pure-bf16 mismatches in SDPA).
            from vjepa21_hf import VJEPA21Model  # vendored, on sys.path via config
            try:
                self.model = VJEPA21Model.from_pretrained(
                    spec.resolve_source(), attn_implementation="sdpa",
                ).to(device).eval()
            except Exception:  # noqa: BLE001 - some builds reject the kwarg
                self.model = VJEPA21Model.from_pretrained(spec.resolve_source()).to(device).eval()
            self.prefix = 0
            self._interp = False
            self.autocast = True
            self.in_dtype = torch.float32

        self.mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)

        if COMPILE:
            try:
                self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
            except Exception:  # noqa: BLE001
                pass

    def _pre(self, frames_u8: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """(n,C,H,W) uint8 -> (n,3,*hw) normalized tensor in self.in_dtype."""
        x = frames_u8.to(self.device, non_blocking=True).float().div_(255.0)
        x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return x.to(self.in_dtype)

    @torch.inference_mode()
    def process(self, frames_u8, hw, grid, batch_size, tick) -> torch.Tensor:
        if self.spec.family == "dino":
            return self._dino(frames_u8, hw, grid, batch_size, tick)
        if self.device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._vjepa(frames_u8, hw, grid, tick)
        return self._vjepa(frames_u8, hw, grid, tick)

    def _dino(self, frames_u8, hw, grid, bs, tick) -> torch.Tensor:
        gh, gw = grid
        n, npatch = frames_u8.shape[0], gh * gw
        kw = {"interpolate_pos_encoding": True} if self._interp else {}
        outs = []
        for i in range(0, n, bs):
            x = self._pre(frames_u8[i:i + bs], hw)
            lhs = self.model(pixel_values=x, **kw).last_hidden_state  # (b, prefix+gh*gw, D)
            patches = lhs[:, self.prefix:self.prefix + npatch, :]
            outs.append(patches.reshape(x.shape[0], gh, gw, -1).to(torch.float16))
            tick(x.shape[0])
        return torch.cat(outs, 0)

    def _vjepa(self, frames_u8, hw, grid, tick) -> torch.Tensor:
        gh, gw = grid
        n = frames_u8.shape[0]
        tub, cf = self.spec.tubelet, self.spec.chunk_frames
        outs, i = [], 0
        while i < n:
            chunk = frames_u8[i:i + cf]
            m = chunk.shape[0]
            pad = (-m) % tub
            if pad:
                chunk = torch.cat([chunk, chunk[-1:].repeat(pad, 1, 1, 1)], 0)
            x = self._pre(chunk, hw)                       # (m+pad, 3, *hw)
            mm = x.shape[0]
            vid = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # (1,3,T,H,W)
            feats = self.model(pixel_values_videos=vid, skip_predictor=True).last_hidden_state
            t_tok = mm // tub
            gridf = feats.reshape(t_tok, gh, gw, -1)       # row-major [T,H,W]
            gridf = gridf.repeat_interleave(tub, dim=0)[:m]  # tubelet -> frames
            outs.append(gridf.to(torch.float16))
            tick(m)
            i += cf
        return torch.cat(outs, 0)


class _Manager:
    def __init__(self):
        self._cache: dict[tuple[str, str], _Extractor] = {}
        self._lock = threading.Lock()

    def get(self, spec: ModelSpec, device: str) -> _Extractor:
        key = (spec.key, device)
        with self._lock:
            ext = self._cache.get(key)
            if ext is None:
                ext = _Extractor(spec, device)
                self._cache[key] = ext
            return ext


MANAGER = _Manager()


def flush_vram() -> dict:
    """Release cached/unused GPU memory back to the driver.

    Models stay loaded: `empty_cache()` only frees the caching allocator's
    unused reserved blocks, never live allocations (the model weights), so this
    flushes the previous job's intermediate tensors without unloading models.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return {"ok": True, "freed_mb": 0.0}
    idxs = sorted({int(d.split(":")[1]) for d in DEVICES if d.startswith("cuda")})
    before = sum(torch.cuda.memory_reserved(i) for i in idxs)
    for i in idxs:
        torch.cuda.synchronize(i)
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    after = sum(torch.cuda.memory_reserved(i) for i in idxs)
    return {"ok": True,
            "reserved_before_mb": round(before / 1048576, 1),
            "reserved_after_mb": round(after / 1048576, 1),
            "freed_mb": round((before - after) / 1048576, 1)}


def _split(n: int, k: int, align: int = 1) -> list[tuple[int, int]]:
    k = max(1, min(k, n))
    base = n // k
    bounds, s = [], 0
    for i in range(k):
        e = n if i == k - 1 else s + base
        if align > 1 and i < k - 1:
            e = max(s + align, (e // align) * align)
        if e > s:
            bounds.append((s, e))
        s = e
        if s >= n:
            break
    if bounds and bounds[-1][1] < n:
        bounds[-1] = (bounds[-1][0], n)
    return bounds


def extract_features(spec: ModelSpec, frames_u8: torch.Tensor, devices: list[str],
                     hw: tuple[int, int], grid: tuple[int, int],
                     batch_size: int, progress) -> torch.Tensor:
    """Return dense features (T, grid_h, grid_w, D) on devices[0]."""
    n = frames_u8.shape[0]
    primary = devices[0]
    unit = spec.chunk_frames if spec.family == "vjepa" else _DINO_UNIT
    k = max(1, min(len(devices), -(-n // unit)))  # ceil(n/unit), capped at #gpus
    slices = _split(n, k, align=spec.tubelet if spec.family == "vjepa" else 1)

    results: list[torch.Tensor | None] = [None] * len(slices)
    lock = threading.Lock()
    done = [0]

    def tick(c):
        with lock:
            done[0] += c
            progress(min(1.0, done[0] / max(n, 1)))

    def work(i: int, dev: str, s: int, e: int):
        if dev.startswith("cuda"):
            torch.cuda.set_device(dev)
        ext = MANAGER.get(spec, dev)
        g = ext.process(frames_u8[s:e].to(dev, non_blocking=True), hw, grid, batch_size, tick)
        results[i] = g.to(primary)

    if len(slices) == 1:
        work(0, primary, *slices[0])
    else:
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            futs = [ex.submit(work, i, devices[i % len(devices)], s, e)
                    for i, (s, e) in enumerate(slices)]
            for f in futs:
                f.result()

    return torch.cat([r for r in results if r is not None], 0)
