# Review Request — 2026-07-07-review-fixes

The reviewer has the repository via git. Domain/code review — keep the jahns-workflow harness out
of scope. This is a **remediation re-review**: the previous round's SAM2 findings, now fixed.

- Project / Branch: repvis / main
- Reviewing: `1262e1572bd1daf56b4d5cf34497ffce47891cfe`   (diff against `071502ac85de130d551521f3e29d7735c5796c08` — the previous round's tip; the diff is the fixes + a shared model-load module)

## What changed and why

All five REAL findings from `2026-07-07-sam2-foreground-feedback.md` are fixed, plus the refit
`decision/refit-mask-grid-threshold` ruling. Implemented in parallel, one owner per file, so the
change set is the fixes only.

## Read these first

1. `repvis/sam.py` — the blocker fix. Points are now `(x,y,label,frame)`; conditioning is
   **add-then-run per frame** (a single `model(frame_idx=min)` silently drops other frames' points —
   verified against the real model), with reverse propagation when the earliest click isn't frame 0.
2. `repvis/pipeline.py` — `_run_sam` (raises) vs `_segment` (initial-run wrapper, sets
   `seg.available/error/empty`); `segment_and_render` calls `_run_sam` directly so a refine failure
   does NOT clobber `pca.mp4`/`masks.u1`/`meta`; per-frame routing; point bounds check; weighted refit.
3. `repvis/server.py` — 4-tuple point validation (finite → 400, out-of-bounds → 422),
   `ACTIVE_RUN_MUTATIONS` mutex gating DELETE runs/sources + supersede.
4. `repvis/pca.py` — weighted `refit_display` (weighted PCA over grid tokens by fg fraction).
5. `repvis/modelload.py` (new) + `repvis/extract.py` — shared `LOAD_LOCK` across model families.
6. `static/app.js` — `frame=floor(currentTime/duration*seg.frames)`; controls shown whenever a run
   rendered (not only `seg.available`).

## Claims to attack

1. **Multi-frame conditioning is correct**: clicks on frames 0 and N both condition the same object;
   the mask on every frame reflects all points; nothing is silently dropped; reverse propagation
   covers frames before the earliest click. (The prior single-`model(min)` bug is gone.)
2. **No-clobber on refine failure**: if SAM raises during `/segment` or `/refit`, the existing
   artifacts are byte-identical afterward and the endpoint returns 5xx (test asserts this).
3. **Mutex actually excludes**: a `DELETE /api/runs` or `DELETE /api/sources` during a segment/refit
   cannot rmtree that run's dir (409 / skip). Attack the window between check and rmtree.
4. **Point validation**: NaN/Inf/out-of-range/bad-frame points are rejected (400 shape, 422 bounds)
   before reaching SAM2; nothing pathological reaches the model.
5. **Weighted refit is a proper weighted PCA** (weighted mean, `sqrt(w)`-scaled SVD, weighted
   quantiles) and reduces to the unweighted fit at equal weights; thin structures now contribute.
6. **Shared load lock closes the race**: extractor and SAM2 `from_pretrained` are now mutually
   serialized (single module-level `LOAD_LOCK`).

## Evidence already produced (mine — inspect, don't trust)

| Claim | Command / artifact | My reading | Where |
|---|---|---|---|
| frame-idx | live E2E: click at 50% playback | `meta.seg.points` gets `[...,frame 24]`, not 0 | PROGRESS §Rounds |
| no-clobber + validation | `REPVIS_TEST_GPU=1 uv run pytest` | 6 pass (monkeypatch SAM raise → 5xx, bytes intact; NaN→400, oob→422) | `tests/test_api.py` |
| weighted PCA | numeric checks | equal-weights == unweighted; thin ridge favored | agent report / `pca.py` |

## Known weak spots

- **Frame alignment exactness** (prior open question) is only partially closed — a re-decode
  determinism test was added, but GPU-phase1-vs-CPU-reseg exactness on open-GOP/VFR clips is still
  open (`spike/frame-alignment-check`).
- Reverse-propagation cost: a late-only click re-propagates twice (fwd+rev) — correctness over speed.
- Auto-seed quality unchanged (`feat/sam-autoseed-quality`).

## Domain lens

Correctness of the interactive-segmentation contract (per-frame prompting, failure isolation,
concurrency) and the weighted-PCA math.

## Out of scope

The harness; the unchanged extract/NVDEC/NVENC pipeline; auto-seed quality; the frame-alignment
spike (tracked).

## Response wanted

Major / critical only. For each: a concrete failure mechanism and where you confirmed it. Separate
confirmed findings, open questions, and residual risks from unavailable GPU/data.
