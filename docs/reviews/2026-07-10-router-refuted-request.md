# Review Request — 2026-07-10-router-refuted

The reviewer has the repository via git. This is a domain/code review, not a workflow audit —
keep the jahns-workflow harness out of scope unless asked.

- Project / Branch: repvis / main
- Reviewing: `c5c0ef66cfd993268318e05f18a3ab2120fa2e3f`   (diff against `9c8184dfb2528bafed93cf4ea6cefc03c32c5ab8` — the previous round's tip)

## What changed and why

The previous round's gpt-5.5-pro review found 1 Major + 2 minor (all verified REAL). This round
processed all three, and the Major's candidate fix died an instructive death:

1. **feats-writer fault injection** (minor) — the async feats.f16 writer's mid-dump failure
   branch now has a fault-injection test; all four contract points (error surfaces to the
   caller, no torn final file, tmp cleanup, thread joined) held on the real code — zero bugs.
2. **auto-seed coverage** (minor, task stays open) — degenerate uniform frames could plant the
   (+) primary and the (−) border negative on the SAME cell: fixed (the negative now skips
   cells holding a positive; the standing "negative is otherwise UNGATED" ruling untouched).
   The norm-outlier real-subject exclusion risk is now pinned by a strict xfail (a coherent
   2×2 subject blob at |z|≈11.2 IS excluded today); the policy fix needs a GPU eval and stays
   in the backlog.
3. **sparse-decode Major — implemented, then REFUTED** (the headline). A GPU measurement
   campaign established: NVDEC exact-seek's crash axis is open-GOP (not trims), undetectable
   from metadata; exact-scan creation is ~free; where exact runs it appeared byte-identical to
   the positional walk. On that basis a probe-gated hybrid router shipped (probe = crash check
   {0,1,mid,tail} + head byte-identity vs the positional walk, per (file, backend), persisted
   sidecar, mid-run demotion) and passed every suite (CPU 61, GPU 39). An adversarial review
   then produced a REPRODUCIBLE refutation: on mid-stream-spliced sources (concat/reconnect),
   torchcodec exact `get_frames_at` resolution is **index-set-dependent** — a DENSE [0..n)
   request resolves every frame correctly while SPARSE strided requests silently collapse
   post-splice indices onto the pre-splice frame (no exception; the probe's own sample set
   resolves correctly). No fixed probe can certify the production `compute_indices` set, and
   the risky clip class (dashcam/VOD long videos) is exactly the optimization's target class.
   The router was removed from main (preserved on `wip/exact-decode-router` with all
   measurements); main is back to the positional walk as the ONLY frame-selection mechanism,
   now with a splice fixture in the alignment invariant suite and a characterization test
   recording the refutation. The Major (`perf/sparse-decode-full-walk`) is re-scoped to three
   sound directions: (a) full-comparison certification keyed by (file, indices, chunking) for
   REPEAT SAM re-decodes only, (b) one sequential decode pass per source fanned out to units,
   (c) overlapping the SAM CPU walk under phase-1 GPU work.

## Read these first

1. `tests/test_frame_alignment.py` — the new `_midstream_splice` fixture (now in `ALL`, so
   every positional-invariant test covers the splice class) and
   `test_exact_seek_silently_misaligns_on_splice` (dense-correct AND sparse-wrong asserted).
2. `git diff 9c8184d..c5c0ef6 -- repvis/pipeline.py` — the `_auto_seed` same-cell guard (the
   only production-code change that SURVIVED this round).
3. `tests/test_feats_writer.py`, `tests/test_autoseed.py` — the two minors' tests.
4. `docs/reviews/2026-07-08-backlog-zero-feedback.md` — the triage this round executed.
5. Optional context: branch `wip/exact-decode-router` (the refuted implementation + its probe
   docstrings recording the measured crash matrix).

## Claims to attack

1. **Containment completeness**: main contains NO exact-seek frame selection anywhere — the
   positional walk is the only mechanism; the splice class satisfies every alignment invariant
   test (SAM re-decode positional, phase-1↔SAM identity, determinism, NVDEC cross-backend).
2. **The characterization test actually pins the refuted mechanism**: dense-correct +
   sparse-wrong on the same file, no exception — i.e. it would catch anyone re-introducing
   index-seek selection naively.
3. **Same-cell guard**: cannot regress the standing ungated-negative ruling (nearby-cell and
   on-subject negatives remain allowed; only exact cell collision is excluded), cannot alter
   positive selection, and the k≤5 "no border cell left" corner is handled sanely.
4. **Fault-injection honesty**: the feats-writer test injects at a real seam (writer's own
   `open`/write path), proves the fault fired mid-dump, and asserts the actual caller
   contract — not a tautology that would pass on broken code.
5. **Re-scoped directions are sound**: in particular (a) — given index-set-dependent
   resolution, is (file, indices, chunking, torchcodec-version) a SUFFICIENT certification
   key for replaying exact fetches, or is there a residual (decoder state, threading) that
   could make a certified replay diverge?
6. **xfail honesty**: the norm-outlier-subject xfail is strict and reproduces the exclusion
   (would XPASS-fail if the filter changed), not a vacuous fixture.

## Evidence already produced (mine — inspect, don't trust)

| Claim | Command / artifact | My reading | Where it lives |
|---|---|---|---|
| 1,2 | `uv run pytest tests/test_frame_alignment.py -q` (24/5skip) + `REPVIS_TEST_GPU=1` (29 pass) | splice in ALL; characterization pins dense-ok/sparse-wrong pairs (20→18, 22→18, 24→18 @stride2) | `c5c0ef6` |
| 3 | RED proof in commit body (uniform frame: `{(0,0)}` carried both labels pre-fix) | guard = surgical border-mask subtraction | `6ab758a` |
| 4 | `uv run pytest tests/test_feats_writer.py -q` (2 pass; counter proves 2nd write raised) | all four contract points real | `1ddd767` |
| 6 | `uv run pytest tests/test_autoseed.py -q` (2 pass, 1 xfail strict) | blob |z|≈11.2 flagged, primary lands on background | `d1bf08f` |
| refutation | reproduced twice by me from the adversarial scripts before ruling | index-set dependence confirmed at byte level | test in `c5c0ef6`; full campaign on `wip/exact-decode-router` |

## Known weak spots

- The splice fixture is a synthetic ffmpeg concat (`-c copy` of a clean + mid-GOP-trimmed
  open-GOP segment); real-world reconnect VODs/dashcam footage were not sampled. The
  mechanism argument (dangling-ref discontinuity) generalizes, but field prevalence is
  unmeasured.
- The round REMOVED a wrong fix; the Major itself (O(n_total) walk, per-unit prefix
  re-decode, second full CPU walk for SAM) is still open. Long-video latency is unchanged.
- The adversarial campaign produced more fixture recipes (closed-GOP splices, probe-set-
  correct variants) than were upstreamed; only the minimal reproducing fixture is in tests.
- The autoseed policy question (norm-outlier real subjects) is documented, not solved.
- GPU-validated results this round ran on GPUs 4/5 of a shared box; CPU-suite numbers are the
  stable reference.

## Domain lens

Decode semantics first (is the containment airtight; could any index-seek path survive),
then test honesty (do the new tests prove what they claim), then the re-scoped solution
directions for the still-open Major.

## Out of scope

The jahns-workflow harness; `wip/exact-decode-router` and `wip/parallel-joint-encode`
internals (context only); the four open minors (multi-gpu-sam, codec coverage, stale-VFR
migration, segcache byte budget); live-server deployment.

## Response wanted

Major / critical issues only. For each: a concrete failure mechanism and where you confirmed
it. Separate confirmed findings, open domain questions, and residual risks from unavailable
GPU / data / environment.
