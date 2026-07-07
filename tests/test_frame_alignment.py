"""Frame-alignment exactness checks for the decode paths (CPU-only by default).

Background
----------
`repvis/pipeline.py` builds dense features from a phase-1 NVDEC (CUDA) decode
(`video_io.GpuVideoSource`) and, in `segment_and_render` / `_render_source`,
re-decodes the SAME `frame_indices` on the CPU (`_decode_source_frames`) to feed
SAM2. The invariant under test: both paths resolve frame index i to the IDENTICAL
source frame — the i-th frame in presentation order (what ffmpeg emits) — on every
clip class, so SAM masks align 1:1 with feats.f16 / the encoded video.

Both paths now share `video_io.iter_frames_at`: a sequential decode with NO
per-index seeking, selecting frames by POSITION. The characterization tests at the
bottom pin down why seeking was abandoned: seek_mode='approximate' maps index ->
avg-fps timestamp, which resolves WRONG frames on VFR clips (and, on NVDEC,
crashes on open-GOP / stream-copy trims); seek_mode='exact' crashes on `-ss -c
copy` trimmed clips. Positional sequential decode has neither failure mode.

Ground truth is an ffmpeg atlas: `-fps_mode passthrough` rawvideo emits every
container frame once, in presentation order.

The cross-BACKEND check (NVDEC vs CPU per-index identity) is GPU-gated:

    uv run pytest tests/test_frame_alignment.py -q                     # CPU
    REPVIS_TEST_GPU=1 uv run pytest tests/test_frame_alignment.py -q   # + NVDEC
"""
import os
import subprocess

import numpy as np
import pytest

from torchcodec.decoders import VideoDecoder  # noqa: E402

from repvis import pipeline as pl  # noqa: E402
from repvis.video_io import GpuVideoSource  # noqa: E402

GPU = pytest.mark.skipif(os.environ.get("REPVIS_TEST_GPU") != "1",
                         reason="set REPVIS_TEST_GPU=1 to run the NVDEC cross-backend test")

SIZE = "96x64"
W, H = 96, 64


def _run(cmd):
    subprocess.run(cmd, check=True)


def _closed_gop_cfr(path, size=SIZE):
    """Baseline: constant frame rate, closed GOP (keyint 12, no cross-GOP refs)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={size}:rate=10", "-t", "3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-x264-params", "keyint=12:min-keyint=12:scenecut=0:open_gop=0:bframes=2",
          "-fps_mode", "cfr", str(path)])


def _open_gop_cfr(path, size=SIZE):
    """Open-GOP: B-frames before an I-frame reference across the GOP boundary."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={size}:rate=10", "-t", "3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-x264-params", "keyint=12:min-keyint=12:scenecut=0:open_gop=1:bframes=3:b-pyramid=normal",
          "-fps_mode", "cfr", str(path)])


def _vfr(path, size=SIZE):
    """Variable frame rate: irregular per-frame durations (real PTS != avg-rate PTS)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"testsrc2=size={size}:rate=30", "-t", "3",
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


def _atlas(path, w=W, h=H):
    """Positional ground truth: every container frame once, presentation order."""
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path),
                          "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    n = len(out) // (w * h * 3)
    return np.frombuffer(out, dtype=np.uint8).reshape(n, h, w, 3)


def _decode(path, indices, mode):
    """Single-call CPU decode with a chosen seek mode -> (T,H,W,3) uint8."""
    d = VideoDecoder(str(path), device="cpu", seek_mode=mode)
    data = d.get_frames_at(indices=[int(i) for i in indices]).data
    return data.permute(0, 2, 3, 1).contiguous().numpy()


def _phase1_frames(path, indices, device="cpu"):
    """Decode `indices` exactly the way phase-1 does: GpuVideoSource.iter_chunks
    (small first chunk, then full chunks, with a V-JEPA-style align)."""
    vs = GpuVideoSource(path, device, list(indices))
    out = [c.cpu().permute(0, 2, 3, 1).contiguous().numpy()
           for _s, c in vs.iter_chunks(8, align=4)]
    vs.close()
    return np.concatenate(out, 0)


ALL = ["baseline", "opengop", "vfr", "sscopy"]
CFR = ["baseline", "opengop"]
EXACT_OK = ["baseline", "opengop", "vfr"]   # exact seek crashes on sscopy


@pytest.mark.parametrize("kind", ALL)
def test_sam_redecode_is_positional(fixtures, kind):
    """THE invariant: `_decode_source_frames` returns byte-for-byte the i-th
    presentation-order frame (ffmpeg atlas) for every requested index, on every
    fixture class — including the stream-copy trim where exact seek can't run."""
    path = fixtures[kind]
    atl = _atlas(path)
    idx = list(range(0, _nframes(path), 3))
    frames = pl._decode_source_frames(path, idx)
    assert len(frames) == len(idx)
    assert all(np.array_equal(f, atl[i]) for f, i in zip(frames, idx))


@pytest.mark.parametrize("kind", ALL)
def test_phase1_and_sam_decode_identical(fixtures, kind):
    """Cross-PATH invariant: the phase-1 chunked decode (GpuVideoSource, here on
    CPU) and the SAM re-decode resolve the IDENTICAL source frame per index —
    same shared positional decode, so masks align with features by construction."""
    path = fixtures[kind]
    idx = list(range(0, _nframes(path), 2))
    p1 = _phase1_frames(path, idx)
    sam = np.stack(pl._decode_source_frames(path, idx))
    assert p1.shape == sam.shape
    assert np.array_equal(p1, sam)


@pytest.mark.parametrize("kind", EXACT_OK)
def test_positional_matches_exact_seek(fixtures, kind):
    """Where exact seek works at all, the positional decode agrees with it —
    positional selection implements true frame identity, not a new convention."""
    path = fixtures[kind]
    idx = [i for i in [0, 3, 7, 11, 12, 13, 17, 23, 24, 27] if i < _nframes(path)]
    e = _decode(path, idx, "exact")
    frames = pl._decode_source_frames(path, idx)
    assert all(np.array_equal(f, e[k]) for k, f in enumerate(frames))


@pytest.mark.parametrize("kind", ALL)
def test_redecode_is_deterministic(fixtures, kind):
    """The SAM re-decode is byte-for-byte reproducible on every fixture — a
    necessary condition for masks to be stable across re-segment calls."""
    path = fixtures[kind]
    idx = list(range(0, _nframes(path), 2))
    f1 = pl._decode_source_frames(path, idx)
    f2 = pl._decode_source_frames(path, idx)
    assert len(f1) == len(f2) == len(idx)
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))


def test_tail_clamps_when_metadata_overcounts(fixtures):
    """Imperfect containers can report more frames than the stream decodes
    (compute_indices trusts metadata). Positions past the real end clamp to the
    last real frame — same end-clamp approximate seek had — so both decode paths
    still return exactly len(indices) frames and stay aligned."""
    path = fixtures["sscopy"]
    n = _nframes(path)
    atl = _atlas(path)
    idx = list(range(0, n, 2)) + [n + 3, n + 7]
    frames = pl._decode_source_frames(path, idx)
    assert len(frames) == len(idx)
    assert np.array_equal(frames[-1], atl[-1]) and np.array_equal(frames[-2], atl[-1])


# ---- characterization: why the decode is positional, not seek-based ---------

@pytest.mark.parametrize("kind", CFR)
def test_approx_seek_matches_exact_on_cfr(fixtures, kind):
    """On constant-frame-rate clips approximate seek lands on the same frames as
    exact seek — which is why the old seek-based decode looked correct on clean
    CFR sources and the misalignment only surfaced on VFR uploads."""
    path = fixtures[kind]
    idx = [i for i in [0, 3, 7, 11, 12, 13, 17, 23, 24, 27] if i < _nframes(path)]
    assert np.array_equal(_decode(path, idx, "approximate"), _decode(path, idx, "exact"))


def test_approx_seek_diverges_from_exact_on_vfr(fixtures):
    """CHARACTERIZATION of the original bug: on a VFR clip approximate seek
    (index -> avg-fps timestamp) returns entirely different frames than true frame
    identity for most sampled indices. This is why get_frames_at was abandoned for
    frame selection."""
    path = fixtures["vfr"]
    n = _nframes(path)
    idx = [i for i in [0, 3, 7, 11, 12, 13, 17, 23, 24, 27, n - 1] if i < n]
    a = _decode(path, idx, "approximate")
    e = _decode(path, idx, "exact")
    mismatched = sum(0 if np.array_equal(a[i], e[i]) else 1 for i in range(len(idx)))
    assert mismatched >= len(idx) // 2, f"expected VFR divergence, got {mismatched}/{len(idx)}"


def test_exact_seek_crashes_on_stream_copy_trim(fixtures):
    """Why the fix could not simply be seek_mode='exact': on a `-ss -c copy`
    trimmed clip (the shape of many real uploads) exact seek RAISES 'no more
    frames left to decode', while the positional decode handles it (above)."""
    path = fixtures["sscopy"]
    idx = list(range(_nframes(path)))
    with pytest.raises(RuntimeError):
        _decode(path, idx, "exact")


# ---- GPU: cross-BACKEND identity (NVDEC phase-1 vs CPU SAM re-decode) -------

@GPU
@pytest.mark.parametrize("kind", ALL)
def test_nvdec_phase1_matches_cpu_sam_decode(tmp_path_factory, kind):
    """For a fixed frame_indices list, the NVDEC phase-1 decode returns the same
    REAL source frame per index as the CPU SAM re-decode, on every fixture class.
    Fixtures are 640x360 (realistic size; at toy sizes like 96x64 NVDEC itself has
    a deterministic wrong-pixel artifact unrelated to frame selection). NVDEC
    pixels differ from CPU by YUV->RGB rounding, so identity is asserted by
    nearest-match against the CPU atlas."""
    d = tmp_path_factory.mktemp("align-gpu")
    path = d / f"{kind}.mp4"
    size = "640x360"
    if kind == "baseline":
        _closed_gop_cfr(path, size)
    elif kind == "opengop":
        _open_gop_cfr(path, size)
    elif kind == "vfr":
        _vfr(path, size)
    else:
        src = d / "opengop-src.mp4"
        _open_gop_cfr(src, size)
        _stream_copy_trim(src, path)
    atl = _atlas(path, 640, 360).astype(np.int16)
    idx = list(range(0, _nframes(path), 2))
    gpu = _phase1_frames(path, idx, device="cuda")
    assert gpu.shape[0] == len(idx)
    for k, i in enumerate(idx):
        dist = np.abs(atl - gpu[k].astype(np.int16)).reshape(atl.shape[0], -1).mean(1)
        assert int(dist.argmin()) == i, f"index {i}: NVDEC returned frame {int(dist.argmin())}"
        assert float(dist.min()) < 2.0   # YUV->RGB rounding only
