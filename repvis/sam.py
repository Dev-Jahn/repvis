"""Foreground segmentation via SAM2 (lightweight, video, temporally consistent).

Replaces the old feature-clustering remove_bg, which could only carve blobby
patch-grid (e.g. 64x36) regions and shaved subjects from the edges inward. SAM2
runs on the decoded RGB frames and yields PIXEL-accurate masks, propagated across
the clip from a few point prompts by its streaming memory. An automatic seed is
derived from DINO saliency; the user refines with +/- point clicks (see server).

sam2.1-hiera-tiny: ~1 GB VRAM, ~33 ms/frame at 1080p, Apache-2.0, no extra deps
(shipped in transformers). Weights download from the HF Hub on first use.
"""
from __future__ import annotations

import threading

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from . import modelload

_MODEL_ID = "facebook/sam2.1-hiera-tiny"


class _SamManager:
    """Cache one (model, processor) per device; serialize construction.

    `from_pretrained` flips torch's global default dtype during build (same race
    the extractor hits), so all construction is serialized behind the shared
    `modelload.LOAD_LOCK` (across model families, not just SAM).
    """

    def __init__(self):
        self._cache: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, device: str) -> tuple[Sam2VideoModel, Sam2VideoProcessor]:
        with self._lock:
            ent = self._cache.get(device)
        if ent is not None:
            return ent
        with modelload.LOAD_LOCK:
            with self._lock:
                ent = self._cache.get(device)
            if ent is None:
                model = Sam2VideoModel.from_pretrained(_MODEL_ID).to(device).eval()
                proc = Sam2VideoProcessor.from_pretrained(_MODEL_ID)
                ent = (model, proc)
                with self._lock:
                    self._cache[device] = ent
            return ent


MANAGER = _SamManager()


def warm_model(device: str):
    """Load (or reuse) SAM2 on `device` before it is first needed."""
    MANAGER.get(device)


@torch.inference_mode()
def segment(frames: list[np.ndarray], points: list[tuple[float, float, int, int]], device: str,
            *, obj_id: int = 1) -> torch.Tensor:
    """Segment one foreground object across `frames` from point prompts.

    frames: list of (H, W, 3) uint8 RGB frames (source resolution), frame order.
    points: [(x, y, label, frame)] in source pixel coords, label 1 = foreground
            (+), 0 = background (-); `frame` is the frame index the click sits on.
            Points on different frames all condition the SAME object (multi-frame
            refinement). Must be non-empty.
    Returns per-frame binary masks (T, H, W) bool on CPU (True = foreground).

    Points are grouped by frame and each conditioning frame is added-then-run in
    ascending order: SAM2 only consumes a frame's pending points when `model` is
    called on that frame (a single `model(frame_idx=min)` silently drops points
    on every other frame). Propagation then runs forward from the earliest
    conditioning frame, plus reverse when it is not frame 0, so the whole clip is
    covered regardless of where the clicks land.
    """
    if not points:
        raise ValueError("segment() needs at least one point prompt")
    grouped: dict[int, list[tuple[float, float, int, int]]] = {}
    for p in points:
        grouped.setdefault(int(p[3]), []).append(p)
    cond_frames = sorted(grouped)

    model, proc = MANAGER.get(device)
    sess = proc.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
    for f in cond_frames:
        pts_f = grouped[f]
        proc.add_inputs_to_inference_session(
            inference_session=sess, frame_idx=f, obj_ids=obj_id,
            input_points=[[[[float(x), float(y)] for (x, y, _l, _fr) in pts_f]]],
            input_labels=[[[int(lab) for (_x, _y, lab, _fr) in pts_f]]])
        model(inference_session=sess, frame_idx=f)

    T = len(frames)
    H, W = int(sess.video_height), int(sess.video_width)
    out = torch.zeros(T, H, W, dtype=torch.bool)

    def _fill(reverse: bool):
        for o in model.propagate_in_video_iterator(sess, reverse=reverse):
            m = proc.post_process_masks(
                [o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            out[int(o.frame_idx)] = (m[0, 0] > 0).to("cpu")

    _fill(reverse=False)
    if cond_frames[0] > 0:            # cover frames before the earliest click
        _fill(reverse=True)
    sess.reset_inference_session()
    return out
