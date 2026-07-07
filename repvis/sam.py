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


def _run_on_session(model, proc, sess, points: list, obj_id: int) -> torch.Tensor:
    """Condition `sess` on `points` and propagate to per-frame masks (T, H, W) bool
    on CPU. Assumes `sess` carries no stale prompts (fresh, or reset_tracking_data'd).

    Points are grouped by frame and each conditioning frame is added-then-run in
    ascending order: SAM2 only consumes a frame's pending points when `model` is
    called on that frame (a single `model(frame_idx=min)` silently drops points
    on every other frame). Propagation then runs forward from the earliest
    conditioning frame, plus reverse when it is not frame 0, so the whole clip is
    covered regardless of where the clicks land.
    """
    grouped: dict[int, list[tuple[float, float, int, int]]] = {}
    for p in points:
        grouped.setdefault(int(p[3]), []).append(p)
    cond_frames = sorted(grouped)

    for f in cond_frames:
        pts_f = grouped[f]
        proc.add_inputs_to_inference_session(
            inference_session=sess, frame_idx=f, obj_ids=obj_id,
            input_points=[[[[float(x), float(y)] for (x, y, _l, _fr) in pts_f]]],
            input_labels=[[[int(lab) for (_x, _y, lab, _fr) in pts_f]]])
        model(inference_session=sess, frame_idx=f)

    T = int(sess.num_frames)
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
    return out


@torch.inference_mode()
def segment(frames: list[np.ndarray], points: list[tuple[float, float, int, int]], device: str,
            *, obj_id: int = 1) -> torch.Tensor:
    """Segment one foreground object across `frames` from point prompts (one-shot).

    frames: list of (H, W, 3) uint8 RGB frames (source resolution), frame order.
    points: [(x, y, label, frame)] in source pixel coords, label 1 = foreground
            (+), 0 = background (-); `frame` is the frame index the click sits on.
            Points on different frames all condition the SAME object (multi-frame
            refinement). Must be non-empty.
    Returns per-frame binary masks (T, H, W) bool on CPU (True = foreground).

    Builds a throwaway session (default cache size 1, GPU-resident) and drops it —
    use build_session/segment_session when the same clip is re-segmented (clicks).
    """
    if not points:
        raise ValueError("segment() needs at least one point prompt")
    model, proc = MANAGER.get(device)
    sess = proc.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
    out = _run_on_session(model, proc, sess, points, obj_id)
    sess.reset_inference_session()
    return out


def build_session(frames: list[np.ndarray], device: str):
    """Create a REUSABLE video session for a clip that will be re-segmented across
    +/- clicks. Decoded frames and the vision-feature cache are kept on CPU (so an
    idle session costs host RAM, not VRAM) and the cache is sized to hold EVERY
    frame's features, so after the first segment a re-segment skips the vision
    encoder entirely (only prompt conditioning + propagation re-run).

    The caller owns the session lifetime and its invalidation (see pipeline._SegCache).
    """
    _model, proc = MANAGER.get(device)
    return proc.init_video_session(
        video=frames, inference_device=device,
        inference_state_device="cpu", video_storage_device="cpu",
        max_vision_features_cache_size=len(frames), dtype=torch.float32)


@torch.inference_mode()
def segment_session(sess, points: list, device: str, *, obj_id: int = 1) -> torch.Tensor:
    """Re-segment a session built by build_session from new point prompts.

    Clears the previous click's prompts/outputs (reset_tracking_data) but KEEPS the
    vision-feature cache, so a subsequent click reuses the persisted decoded frames
    and vision features and only re-runs conditioning + propagation. Never resets
    the session's cache — invalidation is the run cache's job.
    """
    if not points:
        raise ValueError("segment_session() needs at least one point prompt")
    model, proc = MANAGER.get(device)
    sess.reset_tracking_data()       # drop prior prompts/outputs, keep vision cache
    return _run_on_session(model, proc, sess, points, obj_id)
