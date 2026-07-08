# Review Request — 2026-07-08-parallel-sweep

The reviewer has the repository via git. This is a domain/code review, not a workflow audit —
keep the jahns-workflow harness out of scope unless asked.

- Project / Branch: repvis / main
- Reviewing: `5776596a0cd993fc1af6ef70f92d2434a87ec520`   (diff against `1262e1572bd1daf56b4d5cf34497ffce47891cfe` — the previous round's tip)

## What changed and why

This round drained the entire task backlog in parallel and adjudicated every deliverable with GPU
measurements before merging. Five thrusts:

1. **Concurrency correctness** — the previous round's run-mutation mutex was shown (by re-review)
   to race on a stale snapshot: destructive paths computed skip-sets under `LOCK` then rmtree'd
   OUTSIDE it. All three destructive paths (`delete_runs`, `delete_source`, `_persist_run`
   supersede) now select victims AND rmtree inside ONE `with LOCK:`; segment/refit existence checks
   moved inside the same lock. A same-shape upload-vs-delete phantom race was closed the same way,
   and `segment_and_render`'s meta.json write became atomic (`os.replace`). A follow-on security
   review then caught that the *async* `upload_source` and SSE generator acquired that blocking
   LOCK on the event-loop thread (loop starvation while a delete holds LOCK across rmtree) — both
   now hop to worker threads via `asyncio.to_thread`.
2. **Access control** — shared-token auth (`REPVIS_TOKEN`): a single HTTP middleware gates every
   route (API, source video, run pca, SSE, /static); browser flow bootstraps a `repvis_token`
   httpOnly cookie via `POST /api/login` so `<video>`/EventSource authenticate; unset token = open
   mode + startup warning.
3. **Decode index semantics** — a spike proved "frame alignment is exact" FALSE on VFR, and GPU
   validation showed phase-1 itself was timestamp-based (NVDEC `get_frames_at` approximate), plus
   NVDEC crashes outright on sparse open-GOP/trimmed clips. Both consumers (phase-1 NVDEC, SAM CPU
   re-decode) now share one positional no-seek primitive `video_io.iter_frames_at` (sequential
   `get_next_frame`, pre-roll skip, positional select, end-clamp): frame index i ≡ i-th
   presentation-order frame.
4. **Interactive latency** — per-run SAM2 session/vision-feature cache (`pipeline._SegCache`,
   CPU-resident, LRU=2, dropped under LOCK on delete/supersede/refit): warm click 2.7x faster,
   byte-identical masks.
5. **Seed quality + perf ground truth** — multi-point auto-seed gated by DINO feature similarity
   (extra positives only when cosine-closer to peak 1 than to the patch mean); backbone bench
   replaced the ~+15% estimate with measured numbers (giant compile 1.09x / fp8 1.39x; vith16plus
   1.29x / 1.63x); the parallel-NVENC-encode task was DROPPED on a measured negative (encode is
   ~3% of phase-2 wall; experiment parked on `wip/parallel-joint-encode`).

## Read these first

1. `repvis/server.py` — the whole lock discipline: `LOCK`, `ACTIVE_RUN_MUTATIONS`, single-LOCK
   delete paths, `_materialize_source`, `_events_tick`, the auth middleware `_auth_gate` + login.
2. `repvis/video_io.py` — `iter_frames_at` (the positional primitive) and its use in
   `GpuVideoSource.iter_chunks`.
3. `repvis/pipeline.py` — `_SegCache` + `segment_and_render` (cache reuse/build, atomic meta
   write, no-clobber on SAM failure), `_decode_source_frames`, gated `_auto_seed`, weighted refit.
4. `tests/test_api.py` (barrier race tests, auth tests) and `tests/test_frame_alignment.py`
   (positional invariant + NVDEC/CPU cross-backend identity).
5. `repvis/pca.py` — `_weighted_quantile` (weighted type-7).

## Claims to attack

1. **No delete/mutation interleaving survives**: with the single-LOCK discipline there is no
   thread interleaving in which a run/source dir is rmtree'd while a segment/refit reads it, nor
   one where segment/refit proceeds against an already-deleted dir (it 404s under the lock).
2. **No event-loop blocking**: no code path acquires `LOCK` on the asyncio event-loop thread
   anymore (upload + SSE go through `asyncio.to_thread`); a long delete cannot freeze SSE/uploads.
3. **Auth completeness**: with `REPVIS_TOKEN` set, every content-bearing route (including /static,
   media, SSE, /docs) returns 401 without a valid credential; the exempt set is exactly
   `/`, `/api/login`, `/api/auth`; comparisons are constant-time; unset token preserves the old
   open behavior exactly.
4. **Positional decode identity**: for every clip class we can fixture (closed-GOP CFR, open-GOP,
   VFR, stream-copy trim), phase-1 NVDEC and the SAM CPU re-decode resolve the SAME source frame
   per index (pixel-rounding ≤2 aside), and the definition (i-th presentation-order frame) matches
   ffmpeg passthrough. `seek_mode=exact` remains unusable (crashes on trims).
5. **Cache correctness**: a warm cached re-segment is byte-identical (`masks.u1`) to a cold one
   for the same points; different points re-segment (no stale replay); every delete/supersede/refit
   path drops the cache entry, so a stale session can never produce wrong masks.
6. **Seed gating**: the gated multi-point seed never yields worse coverage than the single-point
   seed on any clip class; the border negative cannot carve a hole in a frame-filling subject.
7. **Weighted quantile**: `_weighted_quantile` with equal weights equals `torch.quantile`
   (type-7) within fp tolerance for all n ≥ 2 and q ∈ [0,1]; with non-negative weights the CDF is
   monotone (searchsorted stays valid).
8. **Encode drop justification**: at 1080p/1200 frames the encode is ~3% of phase-2 wall; no
   plausible workload makes NVENC the joint-run bottleneck, so dropping the fan-out task was right.

## Evidence already produced (mine — inspect, don't trust)

| Claim | Command / artifact | My reading | Where it lives |
|---|---|---|---|
| 1 | `uv run pytest tests/test_api.py -k "race" -q` (3 barrier tests, RED on reverted server) | interleaving forced; fix excludes | `tests/test_api.py`, commit `d39ee3b` |
| 3 | auth tests + live 401 sweep | every route gated; exempt exact | `tests/test_api.py`, commit `afefcb6` |
| 4 | `REPVIS_TEST_GPU=1 uv run pytest tests/test_frame_alignment.py -q` (24 pass) + E2E VFR centroid <4px | cross-backend index identity on 4 fixture classes | `tests/test_frame_alignment.py`, commit `34ce9dd` |
| 5 | cold/warm sha256 of masks.u1 ×3 regimes + diff-points guard | byte-match + true re-segment | commit `e09c7d5`, PROGRESS 2026-07-08 |
| 6 | 3-arm (OLD/DRAFT/NEW) 5-clip GPU eval | no regression; fill_frame 0.407→0.9995 | commit `9ed7a52`, PROGRESS |
| 7 | `uv run pytest tests/test_pca.py -q` (8 pass, incl. n=16 q=0.02 regression) | reduces exactly at equal weights | `tests/test_pca.py`, commit `04991fd` |
| 8 | phase-2 profile: 87.7s wall = SAM 56.1 + decode 15.2 + dump ~14 + encode 2.5–3.6 | encode ~3% | task notes `perf/parallel-joint-encode`, branch `wip/parallel-joint-encode` |

## Known weak spots

- `iter_frames_at` uses a **private torchcodec API** (`torchcodec._core.get_next_frame`); pinned
  at 0.14.0 and guarded by a GPU test, but an upgrade could break it silently on CPU-only CI.
- VFR runs created BEFORE the positional fix have masks/feats under the old (timestamp) index
  definition; nothing migrates them.
- The `_SegCache` has a documented benign drop/re-put race (in-flight segment can re-cache a
  just-deleted run_id); argued harmless via LRU + never-requested-again.
- Auth cookie has no `Secure` flag (HTTP deployment); token travels cleartext on the LAN; login
  has no rate limit yet (tracked: `fix/auth-hardening`).
- `upload_source` holds LOCK across `shutil.move` + small writes on a worker thread — same-fs
  rename assumed (mkstemp inside SOURCES_DIR).
- Multi-unit phase-1 decode now pays a prefix re-decode (-28% decode throughput at 4 units,
  still above extraction speed).

## Domain lens

Concurrency first (lock discipline, event-loop safety, cache invalidation), then decode index
semantics (VFR/open-GOP/trim edge cases), then the numerics (weighted quantile, PCA color
stability). Web-security of the token gate matters but the deployment is a single-user LAN box.

## Out of scope

The jahns-workflow harness; work in flight on branches (fp8 fidelity gate, phase-2 tail perf,
auth hardening, saliency artifact-token fix) — they get their own round.

## Response wanted

Major / critical issues only. For each: a concrete failure mechanism and where you confirmed it.
Separate confirmed findings, open domain questions, and residual risks from unavailable
GPU / data / environment.
