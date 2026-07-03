"""GPU video IO via torchcodec: NVDEC decode in, NVENC encode out.

Everything stays on the GPU — decode lands uint8 RGB on-device (NV12->RGB on
GPU), and the encoder consumes uint8 RGB CUDA tensors directly (RGB->NV12 on
GPU inside NVENC's input path). There is no CPU pixel work and no host copy
anywhere. Measured on RTX PRO 6000: sampled 1080p decode ~850 fps, streaming
h264_nvenc encode ~510 fps.
"""
from __future__ import annotations

import numpy as np
import torch


def compute_indices(n_total: int, fps: float, target_fps: float, max_frames: int) -> list[int]:
    n_total = max(1, int(n_total))
    stride = max(1, round(fps / target_fps)) if (fps and target_fps) else 1
    idx = list(range(0, n_total, stride))
    if len(idx) > max_frames:
        sel = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[int(i)] for i in sel]
    return idx


def probe_video(path) -> dict:
    """Container metadata (no frames decoded)."""
    from torchcodec.decoders import VideoDecoder
    md = VideoDecoder(str(path), device="cpu").metadata
    fps = float(md.average_fps or 30.0)
    duration = float(md.duration_seconds or 0.0)
    n_total = int(md.num_frames or 0) or int(duration * fps) or 1
    if duration <= 0:
        duration = n_total / max(fps, 1.0)
    return {"width": int(md.width), "height": int(md.height),
            "duration": duration, "src_fps": fps, "n_total": n_total}


class GpuVideoSource:
    """NVDEC-decoded frames for an explicit index list, served in GPU chunks.

    The decoder is created on first iteration — torchcodec decoders are not
    thread-safe, so it must be built and driven by the same (decode) thread.

    seek_mode='approximate' is deliberate, not a default: 'exact' is marginally
    faster on pristine clips but CRASHES ("no more frames left to decode") on
    real-world uploads with imperfect container metadata / open-GOP structure
    (e.g. anything produced by `ffmpeg -ss -c copy`), where it walks past the
    true stream end mid-batch. 'approximate' decodes those same clips fully.
    We subsample to a few hundred frames anyway, so frame-exact seeking buys
    nothing — robustness is the only thing that matters here.
    """

    def __init__(self, path, device: str, indices: list[int]):
        self.path = str(path)
        self.device = device
        self.indices = indices
        self.n = len(indices)
        self._dec = None

    def iter_chunks(self, chunk: int, align: int = 1):
        """Yield (start, frames_u8 (k,3,H,W) uint8 RGB on self.device).

        Chunk sizes are multiples of `align` (V-JEPA clip length) so temporal
        windows never straddle a chunk boundary. The first chunk is a single
        `align` unit so the consumer starts working with minimal fill latency;
        full chunks follow. Only the final chunk may be a partial tail.
        """
        from torchcodec.decoders import VideoDecoder
        if self._dec is None:
            self._dec = VideoDecoder(self.path, device=self.device, seek_mode="approximate")
        chunk = max(align, (chunk // align) * align)
        first = align if align > 1 else min(16, chunk)
        i, size = 0, first
        while i < self.n:
            sub = self.indices[i:i + size]
            yield i, self._dec.get_frames_at(indices=sub).data
            i += len(sub)
            size = chunk

    def close(self):
        self._dec = None


class GpuEncoder:
    """Streaming NVENC mp4 writer for uint8 RGB CUDA chunks (in frame order)."""

    def __init__(self, out_path, width: int, height: int, fps: float, device: str):
        from torchcodec.encoders import Encoder
        self._enc = Encoder()
        self._stream = self._enc.add_video(
            height=height, width=width, frame_rate=max(1.0, fps), device=device,
            codec="h264_nvenc", preset="p4",
            extra_options={"movflags": "+faststart"})
        self._enc.open_file(str(out_path))
        self._open = True

    def submit(self, rgb_u8: torch.Tensor):
        """rgb_u8: (k,3,H,W) uint8 contiguous, on the encoder's device."""
        self._stream.add_frames(rgb_u8)

    def close(self):
        if self._open:
            self._open = False
            self._enc.close()

    def abort(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort teardown on a failed run
            pass
