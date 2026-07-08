# repvis

Web tool to visualize the **dense patch features** of self-supervised vision
backbones as a PCA-RGB video, side-by-side with the original.

Each spatial (DINO) / spatio-temporal (V-JEPA) patch token is projected onto its
top-3 principal components (one PCA fit over the whole clip, so colors stay
temporally consistent) and shown as RGB. Results land in a **matrix workspace**:
rows are source videos, columns are models, so you can compare backbones on one
clip and (with **joint PCA**) compare clips under one shared basis — the same
color then means the same feature direction *across* videos.

## Models

| Key | Backbone | Kind | Notes |
|-----|----------|------|-------|
| `dinov2-base`  | DINOv2 ViT-B/14 | image, per-frame | classic rainbow PCA |
| `dinov2-large` | DINOv2 ViT-L/14 | image, per-frame | sharper parts |
| `dinov2-giant` | DINOv2 ViT-g/14 | image, per-frame | 1.1B params; best DINOv2 features, ~4× slower than large |
| `dinov3-vitb16`| DINOv3 ViT-B/16 | image, per-frame | gated; CLS+4 registers stripped |
| `dinov3-vith16plus` | DINOv3 ViT-H+/16 | image, per-frame | gated; 840M params |
| `vjepa21-vitl` | V-JEPA 2.1 ViT-L/16 @384 | video, spatio-temporal | tubelet=2, 24×24 grid |

All weights download from the HF Hub on first use. V-JEPA 2.1 pulls
[`Dev-Jahn/vjepa2.1-vitl-fpc64-384`](https://huggingface.co/Dev-Jahn/vjepa2.1-vitl-fpc64-384),
loaded via the vendored port in `src/vjepa21_hf` (rebuild locally with
`scripts/fetch_vjepa21.sh` if preferred). DINOv3 is gated — accept its license and
`huggingface-cli login` first.

### Resolution

Frames are resized **aspect-preserving** to the nearest patch multiple, capping
the longer side at the model's `max_side` (DINO 1024, V-JEPA 640; never upscaled).
So the dense grid stays close to the source resolution instead of a tiny square,
and the PCA video is rendered back at the **source resolution**. Override the cap
per job in the UI's *Advanced › Max resolution*. Higher = finer grid but more
compute (V-JEPA is heaviest — it attends over all spatio-temporal tokens at once).
Note: these backbones were pretrained at lower/fixed resolutions, so very large
inputs trade train/test resolution match for spatial detail.

## Setup & run

```bash
uv sync                        # create .venv + install deps (PyTorch cu130 by default)

./run.sh                       # http://127.0.0.1:8000
HOST=0.0.0.0 ./run.sh          # bind all interfaces (no auth — use with care)
PORT=9000 ./run.sh             # custom port
REPVIS_GPUS=0,1 ./run.sh       # restrict GPUs
REPVIS_COMPILE=1 ./run.sh      # torch.compile (max throughput, slow warmup)
REPVIS_DATA_DIR=/data ./run.sh # keep sources/ + runs/ on another volume
```

For a different CUDA build, change the `pytorch-cu130` index URL in `pyproject.toml`
(e.g. `…/whl/cu128` or `…/whl/cpu`) and re-run `uv sync`.

### Using the workspace

1. **Drop video(s)** onto the tray. Uploads are content-addressed, so the same
   file is stored (and processed) once — re-running never re-uploads or grows disk.
   The **×** on a chip deletes the source and every result derived from it.
2. **Select** one or more sources, pick a **model**, hit **Run**. Selecting ≥2
   sources runs a shared-basis **joint PCA** (cross-video colors) automatically.
3. Each result lands in the **matrix** (rows = sources, cols = models). Re-run a
   source with another model to add a column and compare side by side; everything
   plays from one synced transport.
4. Per PCA cell, the **PC→RGB** control swaps which principal component drives each
   color channel (the 6 permutations) plus optional per-channel invert — applied
   live as a browser filter, no re-encode. **⬇** downloads the PCA video (encoded
   canonically as PC1→R PC2→G PC3→B; the swizzle is display-only).
5. Each PCA cell has its **foreground isolated**: a lightweight **SAM2** pass segments the
   subject (auto-seeded from feature saliency) and bakes the mask into the video (background
   → black), pixel-accurate and temporally consistent. Click the cell to add a **foreground
   point (+)**, **Alt/Option-click** for a **background point (−)** and it re-segments; **↺**
   resets to the auto-seed. **Refit** re-fits the PCA colors over the current foreground so
   within-subject color contrast stands out.

The workspace is **persistent**: completed results live on disk (`runs/`, or
`$REPVIS_DATA_DIR`) and the matrix is rebuilt on page reload — even reloading
mid-run re-attaches to the in-flight job's progress. Re-running a (source, model)
cell replaces its previous result. **Clear** deletes all results from the server
(sources are kept).

## Security

Access control is off by default (a solo, LAN/localhost tool). Set **`REPVIS_TOKEN`**
to a shared secret to gate everything — every API route, source video, run video and
the SSE stream require the token (via `Authorization: Bearer`, an `X-Repvis-Token`
header, or the `repvis_token` cookie that `POST /api/login` sets). With it unset the
server runs open and prints a startup warning that all content is publicly reachable.

The token and its cookie travel **in cleartext over plain HTTP**, so `REPVIS_TOKEN`
alone does not make the server safe to expose directly on an untrusted network. For
remote access, keep it bound to localhost and reach it through an **SSH tunnel**
(`ssh -L 8000:127.0.0.1:8000 host`) or put it behind a **TLS-terminating reverse
proxy** (nginx/caddy). Repeated wrong tokens at `POST /api/login` are throttled per
client IP: after 5 failures the endpoint returns **429** for an exponentially growing
window (a correct login clears it), so online password-guessing is impractical.

## Performance & memory

**Everything stays on the GPU end to end** — there is no CPU pixel path. Frames
are decoded straight onto the GPU with NVDEC (torchcodec CUDA), preprocessed and
run through the backbone in bf16, PCA-projected, upsampled, and handed to the
NVENC encoder (torchcodec CUDA) still as GPU tensors. The only host traffic is
the fp16 feature cache (one D2H per chunk) and the tiny PCA basis.

The work is **fully pipelined** so the GPU never waits on I/O. A run is split into
*units* (contiguous frame segments) bin-packed across GPUs; within each unit,
three stages run concurrently on separate CUDA streams — NVDEC decode, model
forward, and the feature D2H — handed off by CUDA events, never a device-wide
sync. Multiple NVDEC sessions decode in parallel per GPU (the card has several
decode engines). The default CUDA stream is reserved for torchcodec, which
synchronizes against it; model math runs on side streams so decode and compute
truly overlap. Phase 2 (project → upsample → NVENC) is likewise streamed, with
each source rendered on its own GPU when several are available.

On one RTX PRO 6000 (Blackwell), a 600-frame 1080p clip went from **~47 s (~6%
GPU util, CPU-bound) to ~5.5 s** for DINOv2-B and **~40 s → ~6.3 s** for
V-JEPA 2.1 — roughly **8×** and **6×**, with feature cosine ≥ 0.995 and PCA-RGB
PSNR ≥ 39 dB vs the old CPU-decode path (the residual is NVDEC's 1–2 LSB color
conversion + fp16 resize, both visually indistinguishable). Peak GPU memory is
O(one chunk), independent of video length. Host RAM holds the fp16 feature cache
(bounded by `max_frames`); for **joint** runs it holds every source's features
until the shared basis is fit — keep `max_frames` modest when comparing many long
clips (the box has ample host RAM; nothing spills to disk).

For a single source, decode+extract scale across GPUs but the render/encode is one
NVENC stream on one GPU, so multi-GPU mainly speeds phase 1; joint runs render
every source in parallel across GPUs. GPUs are chosen per job by free VRAM
(`REPVIS_MIN_FREE_GB`, default 16), emptiest first, so busy GPUs on a shared box
are skipped (never hardcodes `cuda:0`).

Always on: NVDEC decode, GPU preprocess (fp16 resize), bf16 (DINO) /
bf16-autocast (V-JEPA), SDPA attention, TF32, multi-GPU + multi-stream overlap,
GPU-side PCA (SVD), NVENC encode.

`REPVIS_COMPILE=1` adds `torch.compile` — but **only for the RoPE models**
(DINOv3, V-JEPA), and this is deliberate, measured, not laziness. The DINOv2
models are already at cuBLAS tensor-core-GEMM peak: `torch.compile` and TensorRT
both fall back to the same cuBLAS kernels and win only ~7% on the forward, which
the NVDEC overlap then eats — net **−6% wall**, so they are never compiled.
DINOv3/V-JEPA instead spend a large fraction of each layer on rotary-embedding
elementwise ops that fuse well: forward 1.28–1.41×, and since forward is the
pipeline's critical path there, **+15–16% wall** at 0.9999 feature cosine.
It stays opt-in because the first clip of each new resolution pays a ~70 s cold
build (fused into a persistent `inductor-cache/` so it is built once per
resolution ever, not per process) — worth it for batch-processing many clips,
not for a single short one. FP8 (E4M3) was measured too: the GEMMs do hit
1.6–1.9× in isolation, but the unquantized bf16 attention dilutes it to ~1.2×
forward (≈ compile), for a torchao dependency and lower fidelity — not worth it.

## Tests

```bash
uv run pytest                      # API tests (no GPU)
REPVIS_TEST_GPU=1 uv run pytest    # + full joint-PCA pipeline run on GPU
```

Tests run against a throwaway `REPVIS_DATA_DIR`, never the repo's `sources/`/`runs/`.

## Rebuilding V-JEPA 2.1 weights (optional)

```bash
bash scripts/fetch_vjepa21.sh          # download official ckpt + convert ViT-L
# or another size from a downloaded checkpoints/*.pt:
.venv/bin/python convert_vjepa21_to_hf.py --model_name vit_base --output_dir models_hf/vjepa2.1-vitb-384
```

## License

Apache-2.0. The V-JEPA 2.1 port under `src/vjepa21_hf/` derives from
[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
