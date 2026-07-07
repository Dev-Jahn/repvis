"""Frame-alignment exactness checks for the SAM CPU re-decode path (CPU-only).

Background
----------
`repvis/pipeline.py` builds dense features from a phase-1 NVDEC (CUDA) decode and,
in `segment_and_render` / `_render_source`, re-decodes the SAME `frame_indices` on
the CPU (`_decode_source_frames`, seek_mode="approximate") to feed SAM2. The claim
under test is that "frame alignment is exact" between those two decodes so the SAM
masks line up 1:1 with feats.f16 / the encoded video. Both decode paths use
seek_mode="approximate".

`approximate` seek converts a frame *index* to a timestamp using the stream's
*average* fps and then decodes from the nearest preceding keyframe. That mapping is
correct for constant-frame-rate (CFR) clips but breaks on variable-frame-rate (VFR)
clips, where the real presentation timestamps do not follow the average rate.

These tests build ffmpeg fixtures that stress the risk and pin down, on CPU:
  * determinism of the production re-decode (`_decode_source_frames` twice),
  * that phase-1-style chunked decode and single-call re-decode agree (batching
    does not change the approximate result within one backend),
  * that approximate == exact on CFR / open-GOP but DIVERGES on VFR,
  * that seek_mode="exact" CRASHES on a stream-copy-trimmed clip (why the code
    cannot simply switch to exact).

The remaining risk — whether the CUDA (NVDEC) approximate decode resolves the same
index->frame mapping as the CPU approximate decode on a VFR source — needs a GPU and
is documented in the accompanying finding, not exercised here.

    uv run pytest tests/test_frame_alignment.py -q
"""
import subprocess

import numpy as np
import pytest

from torchcodec.decoders import VideoDecoder  # noqa: E402

from repvis import pipeline as pl  # noqa: E402

SIZE = "96x64"


def _run(cmd):
    subprocess.run(cmd, check=True)


def _closed_gop_cfr(path):
    """Baseline: constant frame rate, closed GOP (keyint 12, no cross-GOP refs)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={SIZE}:rate=10", "-t", "3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-x264-params", "keyint=12:min-keyint=12:scenecut=0:open_gop=0:bframes=2",
          "-fps_mode", "cfr", str(path)])


def _open_gop_cfr(path):
    """Open-GOP: B-frames before an I-frame reference across the GOP boundary."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={SIZE}:rate=10", "-t", "3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-x264-params", "keyint=12:min-keyint=12:scenecut=0:open_gop=1:bframes=3:b-pyramid=normal",
          "-fps_mode", "cfr", str(path)])


def _vfr(path):
    """Variable frame rate: irregular per-frame durations (real PTS != avg-rate PTS)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={SIZE}:rate=30", "-t", "3",
          "-vf", r"select='not(mod(n\,2))+not(mod(n\,5))'", "-fps_mode", "vfr",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-x264-params", "keyint=12:min-keyint=12:scenecut=0", str(path)])


def _stream_copy_trim(src, path):
    """Trim from a mid-GOP point with -c copy -> imperfect metadata / dangling refs."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.75", "-i", str(src),
          "-c", "copy", str(path)])


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp("align")
    paths = {"baseline": d / "baseline.mp4", "opengop": d / "opengop.mp4",
             "vfr": d / "vfr.mp4", "sscopy": d / "sscopy.mp4"}
    _closed_gop_cfr(paths["baseline"])
    _open_gop_cfr(paths["opengop"])
    _vfr(paths["vfr"])
    _stream_copy_trim(paths["opengop"], paths["sscopy"])
    return paths


def _nframes(path):
    return int(VideoDecoder(str(path), device="cpu").metadata.num_frames)


def _decode(path, indices, mode):
    """Single-call CPU decode with a chosen seek mode -> (T,H,W,3) uint8."""
    d = VideoDecoder(str(path), device="cpu", seek_mode=mode)
    data = d.get_frames_at(indices=[int(i) for i in indices]).data
    return data.permute(0, 2, 3, 1).contiguous().numpy()


def _decode_chunked(path, indices, mode, first=5, chunk=8):
    """Phase-1 style: ONE decoder driven with sequential index sublists, mirroring
    GpuVideoSource.iter_chunks (first small chunk, then full chunks)."""
    d = VideoDecoder(str(path), device="cpu", seek_mode=mode)
    out, i, size = [], 0, first
    while i < len(indices):
        sub = indices[i:i + size]
        out.append(d.get_frames_at(indices=[int(x) for x in sub]).data
                   .permute(0, 2, 3, 1).contiguous().numpy())
        i += len(sub)
        size = chunk
    return np.concatenate(out, 0)


CFR = ["baseline", "opengop"]
ALL = ["baseline", "opengop", "vfr"]


@pytest.mark.parametrize("kind", ALL)
def test_redecode_is_deterministic(fixtures, kind):
    """The production SAM re-decode (`_decode_source_frames`, approximate) is
    byte-for-byte reproducible on every fixture — a necessary condition for masks
    to be stable across re-segment calls."""
    path = fixtures[kind]
    idx = list(range(0, _nframes(path), 2))
    f1 = pl._decode_source_frames(path, idx)
    f2 = pl._decode_source_frames(path, idx)
    assert len(f1) == len(f2) == len(idx)
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))


@pytest.mark.parametrize("kind", ALL)
def test_chunked_and_single_approx_agree(fixtures, kind):
    """Phase-1-style chunked decode and the single-call re-decode return identical
    pixels under approximate seek on every fixture: within one backend the approximate
    result does not depend on how indices are batched. (This isolates the remaining
    risk to the CUDA-vs-CPU *backend* difference, which needs a GPU to test.)"""
    path = fixtures[kind]
    idx = list(range(_nframes(path)))
    single = _decode(path, idx, "approximate")
    chunked = _decode_chunked(path, idx, "approximate")
    assert single.shape == chunked.shape
    assert np.array_equal(single, chunked)


@pytest.mark.parametrize("kind", CFR)
def test_approx_matches_exact_on_cfr(fixtures, kind):
    """On constant-frame-rate clips (closed- AND open-GOP) approximate seek lands on
    the exact same frames as exact seek: the average-rate index->timestamp mapping is
    correct, so these sources carry no alignment risk."""
    path = fixtures[kind]
    idx = [0, 3, 7, 11, 12, 13, 17, 23, 24, 27]
    idx = [i for i in idx if i < _nframes(path)]
    a = _decode(path, idx, "approximate")
    e = _decode(path, idx, "exact")
    assert np.array_equal(a, e)


def test_approx_diverges_from_exact_on_vfr(fixtures):
    """CHARACTERIZATION of the bug: on a VFR clip approximate seek returns the WRONG
    frames (different content) versus exact seek for the majority of sampled indices,
    because real PTS do not follow the average frame rate. This is the concrete failure
    the alignment claim ignores."""
    path = fixtures["vfr"]
    n = _nframes(path)
    idx = [i for i in [0, 3, 7, 11, 12, 13, 17, 23, 24, 27, n - 1] if i < n]
    a = _decode(path, idx, "approximate")
    e = _decode(path, idx, "exact")
    mismatched = sum(0 if np.array_equal(a[i], e[i]) else 1 for i in range(len(idx)))
    # Not merely off-by-a-little: most frames are an entirely different picture.
    assert mismatched >= len(idx) // 2, f"expected VFR divergence, got {mismatched}/{len(idx)}"


def test_exact_seek_crashes_on_stream_copy_trim(fixtures):
    """Why the code cannot simply switch to seek_mode='exact': on a `-ss -c copy`
    trimmed clip (imperfect container metadata, the shape of many real uploads) exact
    seek RAISES 'no more frames left to decode', while approximate decodes it fully.
    A safe fix must therefore be single-decode-path, not exact seek."""
    path = fixtures["sscopy"]
    idx = list(range(_nframes(path)))
    # approximate succeeds
    ok = _decode(path, idx, "approximate")
    assert ok.shape[0] == len(idx)
    # exact blows up
    with pytest.raises(RuntimeError):
        _decode(path, idx, "exact")
