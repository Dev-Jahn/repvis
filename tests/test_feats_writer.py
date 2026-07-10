"""Fault-injection test for the async feats.f16 writer's failure branch.

    uv run pytest tests/test_feats_writer.py -q

No GPU needed: this drives `_dump_feats_async` directly on tiny CPU fp16
tensors and injects an I/O fault into the background writer thread, exercising
the least-travelled path — the writer raising *mid-dump*, after some bytes have
already landed on the temp file but before the atomic os.replace().

What the real contract is (read off repvis/pipeline.py, not assumed):

  * writer `_write()`  — on any BaseException while writing the temp file it
    (1) `tmp.unlink(missing_ok=True)`  -> the partial temp is removed,
    (2) `err.append(e)`               -> the failure is captured, and
    (3) `g.fail(e)`                   -> the group is marked failed + cancelled;
    os.replace() is never reached, so the final feats.f16 is never created.
  * caller `_render_source` — `finally: writer.join()` then `if werr: raise
    werr[0]`, i.e. it joins the thread (no leak) and re-raises, so a writer
    failure surfaces as a failed run instead of a silent success.

These tests pin down all four guarantees: the failure surfaces, no final
feats.f16 (nor a torn one) is left behind, the temp is cleaned up, and the
writer thread is joined rather than leaked.
"""
import builtins
import os
import tempfile

# Redirect sources/ + runs/ away from the repo before repvis.config is imported.
os.environ.setdefault("REPVIS_DATA_DIR", tempfile.mkdtemp(prefix="repvis-test-"))

import pytest  # noqa: E402
import torch  # noqa: E402

from repvis import pipeline as pl  # noqa: E402


def _make_chunks(n=4, d=8):
    """n tiny fp16 CPU tensors — each supports `.numpy().tobytes()`, exactly
    what the writer reads, so a dump produces `n * d * 2` bytes total."""
    return [torch.arange(i * d, i * d + d, dtype=torch.float16) for i in range(n)]


class _FaultyFile:
    """Wraps a real file object but makes `.write()` raise once it has been
    called `fail_after` times — i.e. after a chunk or two are truly on disk, so
    the injected fault lands *mid-dump*, not before the temp file exists."""

    def __init__(self, fh, counter, fail_after):
        self._fh = fh
        self._counter = counter
        self._fail_after = fail_after

    def write(self, b):
        self._counter["n"] += 1
        if self._counter["n"] > self._fail_after:
            raise OSError("injected disk write failure mid-dump")
        return self._fh.write(b)

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)


def _reraise_like_caller(werr):
    """Exactly what _render_source does after joining the writer:
    `if werr: raise werr[0]`."""
    if werr:
        raise werr[0]


def test_writer_failure_mid_dump_surfaces_and_leaves_no_final_file(tmp_path,
                                                                   monkeypatch):
    """Writer raises partway through the dump: the failure must surface to the
    caller, no final feats.f16 (torn or otherwise) may exist, the temp file must
    be gone, and the writer thread must be joined — not leaked."""
    run_dir = tmp_path
    chunks = _make_chunks(n=4)
    g = pl._Group(emit=lambda **_kw: None, total_frames=len(chunks))

    # Inject the fault at a natural seam: the writer's `open(tmp, "wb")`.
    # Setting `open` in the pipeline module namespace shadows the builtin for
    # pipeline code only, so the real temp file is created and the first write
    # lands for real; the SECOND write raises -> a genuine mid-dump failure with
    # a real partial temp on disk.
    counter = {"n": 0}

    def faulty_open(*a, **kw):
        return _FaultyFile(builtins.open(*a, **kw), counter, fail_after=1)

    monkeypatch.setattr(pl, "open", faulty_open, raising=False)

    writer, werr = pl._dump_feats_async(chunks, run_dir, g)

    # ---- caller contract: join (finally) then re-raise (`if werr: raise ...`) ----
    writer.join()
    assert not writer.is_alive(), "writer thread must be joined, not leaked"

    # (a) the failure was captured and surfaces when the caller re-raises.
    assert werr, "writer failure must be recorded for the caller to re-raise"
    assert isinstance(werr[0], OSError)
    with pytest.raises(OSError):
        _reraise_like_caller(werr)

    # the fault really fired mid-dump: >=1 chunk was written before it raised.
    assert counter["n"] >= 2, "fault must fire after some bytes are on disk"

    # (b) no final feats.f16 (os.replace never ran -> no torn/partial final file).
    assert not (run_dir / "feats.f16").exists()
    # (c/d) the temp file was cleaned up by the writer's except branch.
    assert not (run_dir / "feats.f16.tmp").exists()

    # the group is marked failed + cancelled, so the whole run fails.
    assert g.error is not None and g.cancel.is_set()


def test_writer_success_control_writes_full_file_and_removes_temp(tmp_path):
    """Control: with no injected fault the same harness succeeds — feats.f16
    holds the exact concatenated chunk bytes, the temp is gone, nothing is
    recorded as an error, and the group stays healthy. Proves the failure test's
    outcome is caused by the injected fault, not by a broken harness."""
    run_dir = tmp_path
    chunks = _make_chunks(n=4)
    g = pl._Group(emit=lambda **_kw: None, total_frames=len(chunks))

    writer, werr = pl._dump_feats_async(chunks, run_dir, g)
    writer.join()

    assert not writer.is_alive()
    assert not werr
    assert g.error is None and not g.cancel.is_set()
    assert not (run_dir / "feats.f16.tmp").exists()

    expected = b"".join(c.numpy().tobytes() for c in chunks)
    assert (run_dir / "feats.f16").read_bytes() == expected
