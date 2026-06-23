"""Chunked video decode (torchcodec, PyAV fallback) and NVENC encode.

`VideoSource` decodes to CPU and yields frames in bounded chunks so the pipeline
never holds the whole video in memory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

_NVENC: bool | None = None


def _compute_indices(n_total: int, fps: float, target_fps: float, max_frames: int) -> list[int]:
    n_total = max(1, int(n_total))
    stride = max(1, round(fps / target_fps)) if (fps and target_fps) else 1
    idx = list(range(0, n_total, stride))
    if len(idx) > max_frames:
        sel = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[int(i)] for i in sel]
    return idx


class VideoSource:
    """Sampled frames of a video, served in chunks (frames stay on CPU)."""

    def __init__(self, path, target_fps: float = 24.0, max_frames: int = 900):
        self.path = str(path)
        self._tc = None        # torchcodec decoder (chunked random access)
        self._frames = None    # PyAV fallback: full CPU tensor of sampled frames
        self.indices: list[int] = []
        self.meta: dict = {}
        self._setup(target_fps, max_frames)
        self.n = len(self.indices)

    def _setup(self, target_fps, max_frames):
        try:
            from torchcodec.decoders import VideoDecoder
            dec = VideoDecoder(self.path, device="cpu")
            md = dec.metadata
            fps = float(md.average_fps or 30.0)
            duration = float(md.duration_seconds or 0.0)
            n_total = int(md.num_frames or 0) or int(duration * fps) or 1
            if duration <= 0:
                duration = n_total / max(fps, 1.0)
            self.indices = _compute_indices(n_total, fps, target_fps, max_frames)
            self.meta = {"width": int(md.width), "height": int(md.height),
                         "duration": duration, "src_fps": fps,
                         "fps_out": len(self.indices) / duration if duration > 0 else fps,
                         "n_total": n_total, "n_sampled": len(self.indices)}
            self._tc = dec
        except Exception as e:  # noqa: BLE001
            self._setup_av(target_fps, max_frames, e)

    def _setup_av(self, target_fps, max_frames, prev):
        import av
        container = av.open(self.path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        n_total = int(stream.frames or 0)
        duration = float((stream.duration or 0) * stream.time_base) if stream.duration else 0.0
        if n_total == 0:
            n_total = int(duration * fps) if duration else 0
        want = set(_compute_indices(max(n_total, 1), fps, target_fps, max_frames)) if n_total else None
        out, i, w, h = [], 0, 0, 0
        for frame in container.decode(video=0):
            if want is None or i in want:
                arr = frame.to_ndarray(format="rgb24")
                h, w = arr.shape[:2]
                out.append(torch.from_numpy(arr))
                if want is not None and len(out) >= len(want):
                    break
                if want is None and len(out) >= max_frames:
                    break
            i += 1
        container.close()
        if not out:
            raise RuntimeError(f"video decode failed (torchcodec: {prev}; pyav: no frames)")
        self._frames = torch.stack(out).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W) CPU
        if not duration:
            duration = len(out) / max(fps, 1.0)
        self.indices = list(range(len(out)))
        self.meta = {"width": w, "height": h, "duration": duration, "src_fps": fps,
                     "fps_out": len(out) / duration if duration > 0 else fps,
                     "n_total": n_total or len(out), "n_sampled": len(out)}

    def iter_chunks(self, chunk: int) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield (start_index, frames_u8 [k,C,H,W] uint8 on CPU)."""
        for i in range(0, self.n, max(1, chunk)):
            if self._tc is not None:
                sub = self.indices[i:i + chunk]
                yield i, self._tc.get_frames_at(indices=sub).data
            else:
                yield i, self._frames[i:i + chunk]

    def close(self):
        self._tc = None
        self._frames = None


def _has_nvenc() -> bool:
    global _NVENC
    if _NVENC is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=20).stdout
            _NVENC = "h264_nvenc" in out
        except Exception:  # noqa: BLE001
            _NVENC = False
    return _NVENC


def encode_rgb_frames(frames: Iterable[np.ndarray], out_path: Path, width: int, height: int, fps: float):
    """Encode an iterable of contiguous uint8 RGB (H,W,3) frames to mp4 via ffmpeg."""
    if _has_nvenc():
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", f"{max(fps, 1.0):.6f}", "-i", "-",
           *vcodec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encode failed (exit {ret})")
