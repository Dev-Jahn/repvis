"""Shared model-load lock.

`from_pretrained` is NOT thread-safe: it flips torch's global default dtype
during construction, so two models loading concurrently — even from different
families (extractor vs. SAM) — race and one comes out fp32 (-> "mat1 and mat2
must have the same dtype" mid-forward). Serialize ALL construction across
model families through this one lock.
"""
from __future__ import annotations

import threading

LOAD_LOCK = threading.Lock()
