"""Configuration, model registry and global performance flags."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REPVIS_DATA_DIR", str(ROOT)))  # where sources/ + runs/ live
SOURCES_DIR = DATA_ROOT / "sources"   # content-addressed uploaded videos (stored once)
RUNS_DIR = DATA_ROOT / "runs"         # per-run PCA outputs + meta (persist across restarts)
STATIC_DIR = ROOT / "static"
MODELS_HF_DIR = ROOT / "models_hf"
VENDOR_SRC = ROOT / "src"  # holds the vendored vjepa21_hf package

SOURCES_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

# ---- global inference perf knobs (Blackwell) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# torch.compile is opt-in: it gives the best steady-state throughput but adds a
# one-time warmup that can dominate short clips. Everything else (bf16, SDPA,
# multi-GPU sharding, GPU decode/encode, GPU PCA) is always on.
COMPILE = os.environ.get("REPVIS_COMPILE", "0") == "1"


def available_devices() -> list[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    env = os.environ.get("REPVIS_GPUS")
    if env:
        ids = [int(x) for x in env.split(",") if x.strip() != ""]
    else:
        ids = list(range(torch.cuda.device_count()))
    return [f"cuda:{i}" for i in ids]


DEVICES = available_devices()

# Streaming: the pipeline processes the video in chunks of this many frames so
# peak GPU memory is O(one chunk), independent of total video length.
STREAM_CHUNK = int(os.environ.get("REPVIS_STREAM_CHUNK", "256"))
# Only shard onto GPUs with at least this much free VRAM (shared box: skip busy ones).
_MIN_FREE_GB = float(os.environ.get("REPVIS_MIN_FREE_GB", "16"))


def select_devices(min_free_gb: float = _MIN_FREE_GB) -> list[str]:
    """Usable CUDA devices ordered emptiest-first; skips GPUs that are busy.

    Falls back to the single emptiest GPU if none clear the threshold, and to CPU
    if CUDA is unavailable. Re-evaluated per job so contention is handled live.
    """
    if not torch.cuda.is_available():
        return ["cpu"]
    stats = []
    for d in DEVICES:
        if not d.startswith("cuda"):
            continue
        i = int(d.split(":")[1])
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            continue
        stats.append((free, d))
    if not stats:
        return ["cpu"]
    stats.sort(reverse=True)
    usable = [d for free, d in stats if free / 1e9 >= min_free_gb]
    return usable or [stats[0][1]]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str            # "dino" (image, per-frame) | "vjepa" (video, spatio-temporal)
    source: str            # HF repo id (dino) or dir name under models_hf/ (vjepa)
    patch: int
    max_side: int          # cap on the LONGER processed side; aspect preserved, source never upscaled
    tubelet: int = 1       # temporal patch size (vjepa)
    chunk_frames: int = 32  # frames per forward clip (vjepa)
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    note: str = ""

    def resolve_source(self) -> str:
        # V-JEPA: prefer a locally converted copy, else the Hub repo (auto-downloaded).
        if self.family == "vjepa":
            local = MODELS_HF_DIR / self.source.split("/")[-1]
            if (local / "config.json").exists():
                return str(local)
        return self.source

    def is_available(self) -> bool:
        return True  # all models are downloadable from the Hub on first use


def proc_hw(h: int, w: int, patch: int, max_side: int) -> tuple[int, int, int, int]:
    """Aspect-preserving resize target snapped to a patch multiple on each side,
    capping the longer side at `max_side` (the source is never upscaled).

    Returns (proc_h, proc_w, grid_h, grid_w).
    """
    longer = max(int(h), int(w), 1)
    scale = min(1.0, max_side / longer)
    gh = max(1, round(h * scale / patch))
    gw = max(1, round(w * scale / patch))
    return gh * patch, gw * patch, gh, gw


REGISTRY: dict[str, ModelSpec] = {
    "dinov2-base": ModelSpec(
        "dinov2-base", "DINOv2 · ViT-B/14", "dino",
        "facebook/dinov2-base", patch=14, max_side=1024,
        note="Image SSL. Classic dense-feature rainbow PCA.",
    ),
    "dinov2-large": ModelSpec(
        "dinov2-large", "DINOv2 · ViT-L/14", "dino",
        "facebook/dinov2-large", patch=14, max_side=1024,
        note="Larger DINOv2 backbone, sharper parts.",
    ),
    "dinov3-vitb16": ModelSpec(
        "dinov3-vitb16", "DINOv3 · ViT-B/16", "dino",
        "facebook/dinov3-vitb16-pretrain-lvd1689m", patch=16, max_side=1024,
        note="Gated (license accepted). RoPE; CLS + 4 registers stripped.",
    ),
    "vjepa21-vitl": ModelSpec(
        "vjepa21-vitl", "V-JEPA 2.1 · ViT-L/16", "vjepa",
        "Dev-Jahn/vjepa2.1-vitl-fpc64-384", patch=16, max_side=640, tubelet=2, chunk_frames=32,
        note="Video SSL, dense spatio-temporal features. Heavier at high res.",
    ),
}
