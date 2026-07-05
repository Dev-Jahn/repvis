"""Dense feature extraction (DINOv2/v3 per-frame, V-JEPA 2.1 spatio-temporal).

Frames arrive as uint8 RGB tensors ALREADY on the extractor's GPU (NVDEC puts
them there) — there is no host round-trip anywhere in this module. Each call
runs on a single device; multi-GPU parallelism lives one level up in the
pipeline, which assigns whole video segments to devices.

All heavy work runs in bf16; preprocessing (resize + normalize) is fused on
the GPU per batch.
"""
from __future__ import annotations

import gc
import threading

import torch
import torch.nn.functional as F
from transformers import AutoModel

from .config import COMPILE, DEVICES, ModelSpec


class _Extractor:
    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device

        if spec.family == "dino":
            # DINO is numerically clean in pure bf16. transformers >= 5 interpolates
            # position embeddings automatically for non-native resolutions.
            self.model = AutoModel.from_pretrained(
                spec.resolve_source(), dtype=torch.bfloat16, attn_implementation="sdpa",
            ).to(device).eval()
            cfg = self.model.config
            self.prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
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
            self.autocast = True
            self.in_dtype = torch.float32

        # fp16 resize is ~30% faster than fp32 with negligible output diff (<=1e-3)
        self.mean = torch.tensor(spec.mean, device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(spec.std, device=device, dtype=torch.float16).view(1, 3, 1, 1)

        # torch.compile helps ONLY the RoPE models (spec.compile): measured wall
        # gain dinov3 +16%, V-JEPA +15% from fusing the per-layer rotary-embedding
        # elementwise chain. The DINOv2 models are already at cuBLAS GEMM peak
        # (+7% forward that the decode overlap eats, net -6% wall) — never compile
        # them. mode="default" matches max-autotune throughput at half the warmup.
        # The per-resolution cold build (~70s) is why this stays opt-in (REPVIS_COMPILE);
        # a persistent inductor cache (config.py) amortizes it across restarts.
        if COMPILE and spec.compile:
            try:
                self.model = torch.compile(self.model, mode="default")
            except Exception:  # noqa: BLE001
                pass

    def _pre(self, frames_u8: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """(n,C,H,W) uint8 on-device -> (n,3,*hw) normalized in self.in_dtype."""
        x = frames_u8.half().div_(255.0)
        x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return x.to(self.in_dtype)

    @torch.inference_mode()
    def process(self, frames_u8, hw, grid, batch_size) -> torch.Tensor:
        if self.spec.family == "dino":
            return self._dino(frames_u8, hw, grid, batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._vjepa(frames_u8, hw, grid, batch_size)

    def _dino(self, frames_u8, hw, grid, bs) -> torch.Tensor:
        gh, gw = grid
        n, npatch = frames_u8.shape[0], gh * gw
        outs = []
        for i in range(0, n, bs):
            x = self._pre(frames_u8[i:i + bs], hw)
            lhs = self.model(pixel_values=x).last_hidden_state  # (b, prefix+gh*gw, D)
            patches = lhs[:, self.prefix:self.prefix + npatch, :]
            outs.append(patches.reshape(x.shape[0], gh, gw, -1).to(torch.float16))
        return torch.cat(outs, 0)

    def _vjepa(self, frames_u8, hw, grid, clip_bs) -> torch.Tensor:
        """Batch several 32-frame clips per forward — one clip under-fills the GPU."""
        gh, gw = grid
        n = frames_u8.shape[0]
        tub, cf = self.spec.tubelet, self.spec.chunk_frames
        n_clips = -(-n // cf)
        outs = []
        for c0 in range(0, n_clips, clip_bs):
            clips = []
            for c in range(c0, min(c0 + clip_bs, n_clips)):
                clip = frames_u8[c * cf:(c + 1) * cf]
                pad = cf - clip.shape[0]
                if pad:   # tail clip: repeat last frame to full length
                    clip = torch.cat([clip, clip[-1:].repeat(pad, 1, 1, 1)], 0)
                clips.append(clip)
            x = self._pre(torch.cat(clips, 0), hw)                    # (B*cf,3,*hw)
            vid = x.reshape(len(clips), cf, 3, *hw).permute(0, 2, 1, 3, 4).contiguous()
            feats = self.model(pixel_values_videos=vid, skip_predictor=True).last_hidden_state
            t_tok = cf // tub
            gridf = feats.reshape(len(clips) * t_tok, gh, gw, -1)     # row-major [B*T,H,W]
            gridf = gridf.repeat_interleave(tub, dim=0)               # tubelet -> frames
            outs.append(gridf.to(torch.float16))
        return torch.cat(outs, 0)[:n]


class _Manager:
    def __init__(self):
        self._cache: dict[tuple[str, str], _Extractor] = {}
        self._lock = threading.Lock()        # guards _cache
        self._load_lock = threading.Lock()   # serializes construction (see get)

    def get(self, spec: ModelSpec, device: str) -> _Extractor:
        key = (spec.key, device)
        with self._lock:
            ext = self._cache.get(key)
        if ext is not None:
            return ext
        # `from_pretrained(dtype=...)` is NOT thread-safe: it flips torch's global
        # default dtype during construction, so two models loading concurrently
        # race and one comes out fp32 (-> "mat1 and mat2 must have the same dtype"
        # mid-forward). Serialize all construction; loads are one-time and cached.
        with self._load_lock:
            with self._lock:
                ext = self._cache.get(key)
            if ext is None:
                ext = _Extractor(spec, device)
                with self._lock:
                    self._cache[key] = ext
            return ext


MANAGER = _Manager()

_GRAY_LEVELS = (64, 128, 192)
_gray_cache: dict[tuple, torch.Tensor] = {}
_gray_lock = threading.Lock()


@torch.inference_mode()
def gray_field(spec: ModelSpec, device: str, proc: tuple[int, int],
               grid: tuple[int, int]) -> torch.Tensor:
    """The model's positional field: its mean response to uniform gray frames.

    Gray frames carry no content, so this isolates the position-dependent
    component of the features (RoPE/pos-emb leakage). Used by remove_bg to
    debias tokens before fg/bg clustering. Cached per (model, grid); (gh*gw, D)
    float32 on `device`.
    """
    key = (spec.key, tuple(grid))
    with _gray_lock:
        f = _gray_cache.get(key)
    if f is None:
        n = spec.chunk_frames if spec.family == "vjepa" else 1
        frames = torch.cat([torch.full((n, 3, *proc), v, dtype=torch.uint8, device=device)
                            for v in _GRAY_LEVELS])
        feats = MANAGER.get(spec, device).process(frames, tuple(proc), tuple(grid),
                                                  spec.batch_size)
        f = feats.float().mean(0).reshape(-1, feats.shape[-1]).cpu()
        with _gray_lock:
            _gray_cache[key] = f
    return f.to(device)


def warm_model(spec: ModelSpec, device: str):
    """Load (or reuse) the model on `device` before the pipeline starts."""
    MANAGER.get(spec, device)


def extract_unit_chunk(spec: ModelSpec, frames_u8: torch.Tensor, device: str,
                       proc: tuple[int, int], grid: tuple[int, int], bs: int) -> torch.Tensor:
    """Dense features (T, grid_h, grid_w, D) fp16 for an on-device frame chunk."""
    return MANAGER.get(spec, device).process(frames_u8, proc, grid, bs)


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
