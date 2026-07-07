#!/usr/bin/env python
"""Forward-throughput benchmark for repvis backbones (baseline / compile / FP8).

Purpose
-------
The FP8 + torch.compile forward speedup for the largest backbones
(``dinov2-giant``, ``dinov3-vith16plus``) was only ESTIMATED (~+15% for the
huge+ models under compile). This script MEASURES it on the real backbone, loaded
exactly the way the server loads it, so the numbers can replace the estimate.

It loads each backbone through repvis' own code path (``repvis.extract._Extractor``
+ ``repvis.config.REGISTRY``): same ``AutoModel.from_pretrained(dtype=bf16,
attn_implementation="sdpa")``, same preprocessing (``_pre``: fp16 bicubic resize +
ImageNet normalize), same forward (``process`` -> ``_dino`` / ``_vjepa``). The only
thing the script varies is the *config* under test:

  * ``baseline``  — eager bf16, exactly as the server runs with REPVIS_COMPILE unset.
  * ``compile``   — same model + ``torch.compile(mode="default")`` (identical to what
                    extract.py applies when REPVIS_COMPILE=1 and spec.compile).
  * ``fp8``       — torchao dynamic FP8 (per-row e4m3) quant of the nn.Linear layers,
                    then torch.compile so the scaled-mm actually fuses. Skipped with
                    a clear message if torchao is not installed or the arch/kernels
                    can't do FP8 (see PITFALLS).

Between every config the model is dropped and ``flush_vram()`` is called so reserved
VRAM from the previous config never contaminates the next measurement.

NOTE ON "fp16": the server runs the DINO backbones in **bf16**, not fp16 (see
extract.py: ``dtype=torch.bfloat16``). "baseline" here therefore means the server's
real bf16 eager path. This is deliberate — benchmarking a dtype the server never uses
would not answer the question. bf16 is the honest baseline.

Metrics (per config)
--------------------
  * latency: median / p10 / p90 wall time for ONE forward over ``--frames`` frames,
    timed with CUDA events (device-side, excludes Python dispatch jitter).
  * throughput: frames/sec = frames / median_latency.
  * speedup vs. baseline (throughput ratio).
  * peak VRAM: ``torch.cuda.max_memory_allocated`` during the timed region.

SWEEP PLAN (what to run on the GPU)
-----------------------------------
Primary targets (the ones whose speedup was only estimated):
    uv run python scripts/bench_backbone.py --model dinov2-giant      --device cuda:0
    uv run python scripts/bench_backbone.py --model dinov3-vith16plus --device cuda:0
Controls (to confirm the script reproduces the already-measured deltas in the code
comments: dinov3 +16% compile, dinov2 net-negative compile):
    uv run python scripts/bench_backbone.py --model dinov3-vitb16     --device cuda:0
    uv run python scripts/bench_backbone.py --model dinov2-large      --device cuda:0
Defaults sweep all three configs at the server's real resolution: a 1080x1920 source
capped to max_side=1024 via proc_hw (== what the pipeline feeds the model). Override
the source resolution with --height/--width to probe other aspect ratios.

Pick a FREE gpu with --device (the box is shared; GPU7 runs the live server — do NOT
use cuda:7). Add --configs baseline,compile to skip FP8, or --frames N to change the
per-forward batch (defaults to the spec's tuned batch_size = one real forward).

PITFALLS (read before trusting a number)
----------------------------------------
  * compile cold build: the FIRST compiled forward triggers an inductor build (~70s
    per resolution for these models). --warmup MUST be >=1 (default 3) so the timed
    iters are all warm; the build time is reported separately, not folded into latency.
  * compile is per-resolution: if you change --height/--width, compile rebuilds. Don't
    compare a warm run to a cold run.
  * FP8 on Blackwell (sm_120): FP8 tensor-core GEMM needs a torch/torchao build with
    sm_120 float8 kernels. If quant applies but the scaled-mm silently falls back to
    bf16 you'll see ~no speedup — that's a real (negative) result, report it as such.
    FP8 also only pays off when GEMM-bound; if a model is memory/attention-bound the
    win shrinks. FP8 dynamic-act quant is applied to nn.Linear only (attn qkv/proj +
    mlp) — LayerNorm/attention softmax stay bf16, matching how FP8 inference is done.
  * numerics: this measures SPEED only. Before adopting FP8 in the server, separately
    check PCA-video output parity — FP8 activations can shift feature statistics.
  * throughput saturation: DINO throughput is flat from bs~32 (see config.py), so
    per-forward latency at the tuned batch_size is the meaningful number; frames/sec
    is derived, not independently swept here.
  * warm allocator: max_memory_allocated is reset per config AFTER warmup so it
    reflects the steady-state forward, not one-off compile scratch.

This script does NOT modify the server or any global state that outlives the process.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# repvis is run in place, never installed (pyproject: tool.uv package=false), so add
# the repo root to sys.path — makes `uv run python scripts/bench_backbone.py` work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from repvis.config import REGISTRY, proc_hw  # noqa: E402
from repvis.extract import _Extractor, flush_vram  # noqa: E402


def _make_frames(n: int, src_h: int, src_w: int, device: str) -> torch.Tensor:
    """Random uint8 RGB frames (n, 3, src_h, src_w) on-device, exactly the shape
    NVDEC hands the extractor (source resolution; _pre resizes to proc)."""
    return torch.randint(0, 256, (n, 3, src_h, src_w), dtype=torch.uint8, device=device)


def _build(spec, device: str, compile_it: bool, fp8: bool) -> tuple[_Extractor, dict]:
    """Load the backbone via repvis' own _Extractor (COMPILE is off at import, so the
    constructor never auto-compiles), then apply the config under test on top."""
    info: dict = {}
    ext = _Extractor(spec, device)          # eager bf16, identical to server load
    if fp8:
        info["fp8"] = _apply_fp8(ext.model)
    if compile_it:
        ext.model = torch.compile(ext.model, mode="default")  # matches extract.py
    return ext, info


def _apply_fp8(model) -> str:
    """Dynamic FP8 (e4m3) quant of nn.Linear via torchao. Returns a status string.

    torchao's public factory has been renamed across releases; try the current name
    first, then the legacy one. If neither imports, FP8 is simply unavailable here.
    """
    try:
        from torchao.quantization import quantize_
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"torchao not available ({e}); install torchao to bench FP8") from e

    cfg = None
    err = None
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        cfg = Float8DynamicActivationFloat8WeightConfig()
    except Exception as e:  # noqa: BLE001
        err = e
    if cfg is None:
        try:
            from torchao.quantization import float8_dynamic_activation_float8_weight
            cfg = float8_dynamic_activation_float8_weight()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"torchao present but no FP8 dynamic-act config found ({err} / {e})"
            ) from e
    quantize_(model, cfg)
    return type(cfg).__name__


def _time_config(ext, frames, proc, grid, bs, iters, warmup, device) -> dict:
    """warmup (absorbs compile build) -> event-timed iters. One process() call per
    iter == one real forward over `frames`."""
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(warmup):
        ext.process(frames, proc, grid, bs)
    torch.cuda.synchronize(device)
    warmup_s = time.perf_counter() - t0

    dev_idx = int(device.split(":")[1])
    torch.cuda.reset_peak_memory_stats(dev_idx)
    ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ext.process(frames, proc, grid, bs)
        end.record()
        torch.cuda.synchronize(device)
        ms.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated(dev_idx) / 1048576
    ms.sort()

    def _pct(p: float) -> float:
        return ms[min(len(ms) - 1, int(p * len(ms)))]

    med = statistics.median(ms)
    n = frames.shape[0]
    return {
        "warmup_s": round(warmup_s, 2),
        "median_ms": round(med, 3),
        "p10_ms": round(_pct(0.10), 3),
        "p90_ms": round(_pct(0.90), 3),
        "fps": round(n / (med / 1000.0), 1),
        "peak_vram_mb": round(peak_mb, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure forward throughput for a repvis backbone across "
                    "baseline(bf16)/compile/FP8 configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY),
                    help="backbone key from repvis.config.REGISTRY")
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:N (pick a FREE gpu; NOT cuda:7 — live server)")
    ap.add_argument("--configs", default="baseline,compile,fp8",
                    help="comma list subset of baseline,compile,fp8")
    ap.add_argument("--height", type=int, default=1080, help="SOURCE frame height")
    ap.add_argument("--width", type=int, default=1920, help="SOURCE frame width")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames per forward (default: spec.batch_size = one real forward)")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup iterations (>=1 to absorb the compile cold build)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this benchmark needs a GPU.")
    if not args.device.startswith("cuda"):
        raise SystemExit("--device must be a cuda device.")

    spec = REGISTRY[args.model]
    proc_h, proc_w, gh, gw = proc_hw(args.height, args.width, spec.patch, spec.max_side)
    proc, grid = (proc_h, proc_w), (gh, gw)
    bs = args.frames or spec.batch_size
    n = bs  # feed exactly one batch so each process() is a single forward

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    valid = {"baseline", "compile", "fp8"}
    bad = set(configs) - valid
    if bad:
        raise SystemExit(f"unknown config(s): {sorted(bad)} (valid: {sorted(valid)})")

    print(f"# repvis backbone forward benchmark")
    print(f"model={args.model} ({spec.label})  family={spec.family}  device={args.device}")
    print(f"source={args.height}x{args.width} -> proc={proc_h}x{proc_w}  grid={gh}x{gw} "
          f"({gh * gw} tokens)  patch={spec.patch}")
    print(f"frames/forward={n}  iters={args.iters}  warmup={args.warmup}")
    print(f"torch={torch.__version__}  sm={torch.cuda.get_device_capability(int(args.device.split(':')[1]))}")
    print("-" * 78)

    frames = _make_frames(n, args.height, args.width, args.device)

    results: dict[str, dict] = {}
    for cfg in configs:
        compile_it = cfg in ("compile", "fp8")   # FP8 needs compile to fuse scaled-mm
        fp8 = cfg == "fp8"
        try:
            ext, info = _build(spec, args.device, compile_it=compile_it, fp8=fp8)
        except RuntimeError as e:
            print(f"[{cfg}] SKIPPED: {e}")
            continue
        try:
            r = _time_config(ext, frames, proc, grid, bs, args.iters, args.warmup, args.device)
        except Exception as e:  # noqa: BLE001 — surface OOM / kernel errors per-config
            print(f"[{cfg}] FAILED during timing: {type(e).__name__}: {e}")
            del ext
            flush_vram()
            continue
        results[cfg] = r
        extra = f"  ({info['fp8']})" if fp8 and "fp8" in info else ""
        print(f"[{cfg}]{extra}")
        print(f"    median={r['median_ms']} ms  (p10={r['p10_ms']} p90={r['p90_ms']})  "
              f"fps={r['fps']}  peak_vram={r['peak_vram_mb']} MB  warmup={r['warmup_s']} s")
        del ext
        flush_vram()

    if "baseline" in results:
        base_fps = results["baseline"]["fps"]
        print("-" * 78)
        print("speedup vs baseline (bf16 eager):")
        for cfg, r in results.items():
            print(f"    {cfg:>9}: {r['fps'] / base_fps:.3f}x  ({r['fps']} fps)")


if __name__ == "__main__":
    main()
