#!/usr/bin/env bash
# Download the official V-JEPA 2.1 ViT-L checkpoint and convert it to HF format.
# Optional: the app downloads the converted model from the Hub by default.
set -euo pipefail
cd "$(dirname "$0")/.."
VPY="$PWD/.venv/bin/python"
CKPT=checkpoints/vjepa2_1_vitl_dist_vitG_384.pt
URL=https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
OUT=models_hf/vjepa2.1-vitl-fpc64-384

echo "=== [1/2] download ViT-L checkpoint (~4.9GB, resumable) ==="
curl -L -C - --retry 5 --retry-delay 3 -o "$CKPT" "$URL"
ls -la "$CKPT"

echo "=== [2/2] convert -> $OUT ==="
"$VPY" convert_vjepa21_to_hf.py --model_name vit_large --output_dir "$OUT"
echo "=== files ==="
ls -la "$OUT"
echo "=== DONE vjepa21 vit-l ==="
