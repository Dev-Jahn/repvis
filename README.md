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
| `dinov3-vitb16`| DINOv3 ViT-B/16 | image, per-frame | gated; CLS+4 registers stripped |
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
```

For a different CUDA build, change the `pytorch-cu130` index URL in `pyproject.toml`
(e.g. `…/whl/cu128` or `…/whl/cpu`) and re-run `uv sync`.

### Using the workspace

1. **Drop video(s)** onto the tray. Uploads are content-addressed, so the same
   file is stored (and processed) once — re-running never re-uploads or grows disk.
2. **Select** one or more sources, pick a **model**, hit **Run**. Selecting ≥2
   sources runs a shared-basis **joint PCA** (cross-video colors) automatically.
3. Each result lands in the **matrix** (rows = sources, cols = models). Re-run a
   source with another model to add a column and compare side by side; everything
   plays from one synced transport.
4. Per PCA cell, the **PC→RGB** control swaps which principal component drives each
   color channel (the 6 permutations) plus optional per-channel invert — applied
   live as a browser filter, no re-encode.

## Performance & memory

The pipeline is **streamed in chunks of `REPVIS_STREAM_CHUNK` frames** (default 256)
so peak GPU memory is O(one chunk), independent of video length — a 1-hour 1080p
clip at 6000 frames runs in the same ~6 GB as a 256-frame clip. Per chunk:
decode→extract→offload features to host RAM; the PCA basis is fit once on a capped
token subsample (temporally consistent colors); then project+encode chunk-by-chunk.
Features for the whole job live in CPU RAM (bounded by `max_frames`). A **joint**
run must hold every source's features until the shared basis is fit, so host RAM
there scales with the *total* frames across the selected sources — keep `max_frames`
modest when jointly comparing many long clips.

GPUs are chosen per job by free VRAM (`REPVIS_MIN_FREE_GB`, default 16), emptiest
first, so busy GPUs on a shared box are skipped (never hardcodes `cuda:0`).

Always on: bf16 (DINO) / bf16-autocast (V-JEPA), SDPA attention, TF32, multi-GPU
sharding per chunk, GPU-side PCA (SVD), NVENC encode. `torch.compile` is opt-in.

## Rebuilding V-JEPA 2.1 weights (optional)

```bash
bash scripts/fetch_vjepa21.sh          # download official ckpt + convert ViT-L
# or another size from a downloaded checkpoints/*.pt:
.venv/bin/python convert_vjepa21_to_hf.py --model_name vit_base --output_dir models_hf/vjepa2.1-vitb-384
```

## License

Apache-2.0. The V-JEPA 2.1 port under `src/vjepa21_hf/` derives from
[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
