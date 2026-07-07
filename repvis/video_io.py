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


def iter_frames_at(dec, indices: list[int]):
    """Yield the frames at ascending POSITIONS `indices` — (3, H, W) uint8 RGB on
    the decoder's device — by decoding sequentially from the start, never seeking.

    This is the single definition of "frame index" shared by phase-1 feature
    extraction (NVDEC) and the SAM re-decode (CPU): index i = the i-th frame in
    presentation order, exactly what ffmpeg emits. Index-seek APIs
    (get_frames_at & co) cannot provide that: under seek_mode='approximate' they
    map index -> index/avg_fps -> timestamp, which resolves DIFFERENT frames on
    VFR clips (and per backend), and on NVDEC they crash outright on open-GOP /
    stream-copy-trimmed uploads ("no more frames left to decode"); 'exact' seek
    crashes on those same trims. Sequential decode has neither failure mode.
    `_core.get_next_frame` is torchcodec's only no-seek stepping primitive
    (version pinned in uv.lock); the public surface offers no equivalent.

    Two decoder quirks are papered over, keeping both backends identical:
      * NVDEC emits pre-roll frames (pts < stream begin) on `-ss -c copy` trims
        that CPU/ffmpeg drop — those are skipped and not counted;
      * if container metadata overcounts frames (imperfect uploads), the tail
        clamps to the last real frame, mirroring approximate-seek's end-clamp,
        so the caller always receives exactly len(indices) frames.
    """
    from torchcodec import _core
    begin = float(dec.metadata.begin_stream_seconds or 0.0)
    pos, last, eof = 0, None, False
    for want in indices:
        while not eof and pos <= want:
            try:
                data, pts, _dur = _core.get_next_frame(dec._decoder)
            except IndexError:      # stream ended before metadata said it would
                eof = True
                break
            if float(pts) < begin - 1e-6:   # NVDEC pre-roll junk on trimmed clips
                continue
            last, pos = data, pos + 1
        if last is None:
            raise RuntimeError(f"no decodable frames in {dec.metadata.duration_seconds}s stream")
        yield last


class GpuVideoSource:
    """NVDEC-decoded frames for an explicit index list, served in GPU chunks.

    The decoder is created on first iteration — torchcodec decoders are not
    thread-safe, so it must be built and driven by the same (decode) thread.

    Frames are selected POSITIONALLY via a sequential decode (iter_frames_at):
    no per-index seeking, so the index -> frame mapping is identical to the CPU
    SAM re-decode on every clip class (VFR, open-GOP, stream-copy trims) and
    NVDEC's seek crashes on imperfect uploads never trigger. seek_mode is
    'approximate' only so decoder CREATION skips the full-file scan ('exact'
    scans, and its index walk crashes on trims anyway) — no seek ever happens.
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
        i, size, buf = 0, first, []
        for frame in iter_frames_at(self._dec, self.indices):
            buf.append(frame)
            if len(buf) == size:
                yield i, torch.stack(buf)
                i += size
                size, buf = chunk, []
        if buf:
            yield i, torch.stack(buf)

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
