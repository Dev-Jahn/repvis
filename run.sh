#!/usr/bin/env bash
# Launch the repvis web app.
#   ./run.sh                  -> http://127.0.0.1:8000
#   HOST=0.0.0.0 ./run.sh     -> bind all interfaces (server is unauthenticated — use with care)
#   PORT=9000 ./run.sh        -> custom port
#   REPVIS_COMPILE=1 ./run.sh -> enable torch.compile (best throughput, slow first run)
#   REPVIS_GPUS=0,1 ./run.sh  -> restrict to specific GPUs
# Gated models (DINOv3) need `huggingface-cli login` first.
set -e
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn repvis.server:app \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
