# Review Request — 2026-07-08-backlog-zero

The reviewer has the repository via git. This is a domain/code review, not a workflow audit —
keep the jahns-workflow harness out of scope unless asked.

- Project / Branch: repvis / main
- Reviewing: `9c8184dfb2528bafed93cf4ea6cefc03c32c5ab8`   (diff against `5776596a0cd993fc1af6ef70f92d2434a87ec520` — the previous round's tip)

## What changed and why

The previous round's fable review found 2 Major + 1 Minor; all three were fixed the same day, and
the remaining four backlog tasks were drained in parallel. Five thrusts:

1. **Review fixes (concurrency)** — same-rid mutation overlap now 409s inside the LOCK (a plain
   set couldn't tell two same-rid mutations apart, so the first finisher's `discard` stripped the
   second's protection); `_SegCache` sig gained the device (`(source_id, T, dev)`) so device drift
   is a cold miss instead of a cross-device 500; `create_runs` validates sources, mkdirs run dirs
   and registers the group in ONE LOCK critical section.
2. **FP8 spike closed with NO-GO** — the fidelity gate (Phase 0 noise floor vs Phase 1 torchao
   full-FP8 probe through the real extract→PCA-render path) failed decisively on real footage:
   PCA subspace rotation 4.7–7.5° (gate 2°, fp16 floor 0.7–1.9°), shared-basis ΔE p95 5.4–9.4
   (gate 3). Committed as `scripts/fp8_gate_dinov2.py` + results in `docs/spikes/fp8-attention.md`.
   No production code touched.
3. **Phase-2 tail** — the multi-GB `feats.f16` dump now overlaps SAM segmentation via a background
   writer thread (tmp file + atomic `os.replace`); dump remainder 5–6.5s → ~1s. Reusing phase-1
   NVDEC frames for SAM was REJECTED on measurement (NVDEC vs CPU decode not byte-identical).
4. **Saliency artifact filter** — robust median/MAD modified z-score (|z| > 3.5, two-sided)
   excludes DINO norm-outlier tokens from the positive-peak argmax; fixes the flat-scene hijack;
   CPU unit test committed (`tests/test_autoseed.py`).
5. **Auth hardening** — per-IP exponential backoff on `/api/login` (5 tries → 429 + Retry-After,
   doubling window), unconditional `#t=` fragment strip, README Security section.

Plus the previous review's open question A was RULED: the border negative prompt stays ungated;
the "never worse than single-point" claim is downgraded to an empirical 5-clip statement
(docstring records the residual).

## Read these first

1. `git diff 5776596a..9c8184d -- repvis/server.py` — the same-rid 409 guards, single-LOCK
   create_runs, and the login throttle (`_login_state`).
2. `repvis/pipeline.py` — `_dump_feats_async` + the reordered `_render_source` (writer thread
   lifetime, atomic replace, failure paths); `_SegCache` sig; `_auto_seed` norm filter.
3. `tests/test_api.py` — `test_segment_same_rid_overlap_409s`, `test_segcache_device_in_sig`,
   `test_login_bruteforce_throttled`; `tests/test_autoseed.py`.
4. `scripts/fp8_gate_dinov2.py` + the results section of `docs/spikes/fp8-attention.md`.

## Claims to attack

1. **Same-rid guard completeness**: with the 409 guard, no interleaving of two mutations (any mix
   of /segment and /refit) on the same or different rids can leave a run dir unprotected while a
   mutation is in flight; the marker's add/discard lifetime is now airtight.
2. **Async dump safety**: the background feats writer cannot produce a partial/torn `feats.f16`
   (tmp + atomic replace), cannot deadlock or leak the thread on SAM/encode failure, does not
   change masks or video, and does not raise peak host RAM.
3. **Device-sig sufficiency**: `(source_id, T, dev)` closes the cross-device staleness class; no
   other session-affecting parameter (model choice, processor size, dtype) can vary per-run
   without also being caught.
4. **Norm-filter soundness**: the MAD filter cannot exclude a REAL subject in realistic footage
   (object tokens sit |z|≈1); the MAD=0 guard means uniform-norm frames are untouched; the filter
   only affects the seed, never the masks/render directly.
5. **Login throttle correctness**: the per-IP backoff cannot lock out a legitimate user
   permanently (resets on success, window capped), cannot be bypassed by header spoofing beyond
   the documented proxy caveat, and adds no timing side-channel.
6. **FP8 NO-GO validity**: the gate methodology (bf16 self-noise 0, fp16 floor, real-footage
   fixed-basis ΔE + subspace angle) is sound; the NO-GO is not an artifact of the harness (e.g.
   the synthetic-clip basis degeneracy is quarantined from the verdict).

## Evidence already produced (mine — inspect, don't trust)

| Claim | Command / artifact | My reading | Where it lives |
|---|---|---|---|
| 1 | `uv run pytest tests/test_api.py -k same_rid -q` (RED on reverted server) | overlap 409s; pre-fix blocked instead | `tests/test_api.py`, commit `628becd` |
| 2 | 3× new-path runs: masks byte-identical; async-vs-sync diff == sync-vs-sync baseline magnitude | change introduces nothing | task notes `perf/phase2-sam-decode-tail`, commit `c78ea48` |
| 3 | `uv run pytest tests/test_api.py -k segcache -q` | dev mismatch = miss | commit `628becd` |
| 4 | 3-arm GPU eval on 4 clips + `uv run pytest tests/test_autoseed.py -q` | flat-clip hijack fixed; object |z|≈1 vs artifact 3.6–19.7 | commit `f30136e`, `9c8184d` |
| 5 | `uv run pytest tests/test_api.py -k bruteforce -q` (monkeypatched clock) | 5×401 → 429 → recovery | commit `e0f3089` |
| 6 | `scripts/fp8_gate_dinov2.py` reproduced 2× bit-identical | real-footage failure decisive | `docs/spikes/fp8-attention.md`, commit `6dde2f1` |

## Known weak spots

- The async feats writer holds the full fp16 cache alive ~5–14s longer than before (no new peak,
  but a longer plateau); its failure path (writer raises mid-run) is the least-exercised branch.
- The saliency filter regressed the synthetic fill_frame control (0.9995 → 0.286) — accepted as a
  degenerate no-object clip, but a REAL textureless close-up has not been tested.
- fp8 gate measured dinov2-base only; giant/large inferred by family numerics.
- Login throttle state is per-process (restart clears it) and keys on `request.client.host`.
- GPU-only claims (2's masks-identity, 4's 3-arm eval, 6) were validated on GPUs 4/5 while the
  box was under heavy external load; wall-clock numbers have stated variance.

## Domain lens

Concurrency again first (marker lifetime, writer-thread failure paths), then the numerics verdicts
(fp8 gate methodology, MAD filter edge cases), then auth throttle logic.

## Out of scope

The jahns-workflow harness; the four newly registered minor tasks (multi-gpu-sam, codec coverage,
stale-VFR migration, segcache byte budget); anything on `wip/parallel-joint-encode`.

## Response wanted

Major / critical issues only. For each: a concrete failure mechanism and where you confirmed it.
Separate confirmed findings, open domain questions, and residual risks from unavailable
GPU / data / environment.
