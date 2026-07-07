# Review Request — 2026-07-07-sam2-foreground

The reviewer has the repository via git. This is a domain/code review, not a workflow audit —
keep the jahns-workflow harness (tasks.yaml, ROADMAP.md, docs/) out of scope.

- Project / Branch: repvis / main
- Reviewing: `071502ac85de130d551521f3e29d7735c5796c08`   (diff against `(root)` — first round; review the whole history, focus on the SAM2 rework)

## What changed and why

The background-removal subsystem was scrapped and rebuilt. The old `remove_bg` clustered dense
patch features on a coarse grid (e.g. 64×36) and carved fg/bg from that — it could only produce
blobby, patch-quantized regions and visibly shaved subjects from the edges inward, never a clean
object boundary. The replacement runs **SAM2** (`facebook/sam2.1-hiera-tiny`, Apache-2.0, shipped in
`transformers`) on the decoded RGB frames: pixel-accurate masks propagated across the whole clip
from point prompts, **baked into the PCA video server-side**. An automatic seed is derived from
DINO-feature saliency; the user refines with `+` (click) / `−` (Alt-click) points. Colors are
unchanged by masking; the per-cell **Refit** re-fits the PCA basis over the (now clean) foreground.

## Read these first

1. `repvis/sam.py` — the SAM2 wrapper: `segment(frames, points, device)` builds a
   `Sam2VideoInferenceSession`, seeds one object on frame 0, propagates, returns `(T,H,W)` bool.
2. `repvis/pipeline.py` — the integration: auto-seed from `chunks[0][0]` saliency, `_bake_encode`
   (mask multiplied onto upsampled rgb before NVENC), and `segment_and_render` / `refit_and_render`
   which reuse persisted `feats.f16` + `masks.u1` (no backbone re-run). **`frame_indices` alignment**
   is the load-bearing invariant.
3. `repvis/pca.py` — now mask-free (color mapping + `refit_display` only).
4. `repvis/server.py` — `POST /segment {points}` (empty = auto reset), `POST /refit` (no body).
5. `static/app.js` — click(+)/alt-click(−) → source-pixel mapping → `/segment` → reload; markers.

## Claims to attack

1. **Frame alignment is exact.** `meta.frame_indices` (length == `frames`) are the same source
   indices phase-1 decoded, so the SAM masks, `feats.f16`, and the encoded video are frame-for-frame
   aligned in `segment_and_render` / `refit_and_render` (which re-decode exactly those indices).
   Attack: any path where re-decode order, count, or a dropped/duplicated frame desyncs mask↔frame.
2. **The bake is geometrically correct.** The SAM mask (decoded source res) is resized to the even
   render size `(oh,ow)` = `(result.height, result.width)` and multiplied per frame — no transpose,
   no off-by-one, no aspect/letterbox mismatch vs the upsampled PCA rgb.
3. **Refit uses the right foreground.** `refit_and_render` downsamples the pixel mask to the feature
   grid via `adaptive_avg_pool2d(...) > 0.5` and refits colors on exactly those fg grid tokens.
4. **No backbone re-run on refine/refit.** `/segment` and `/refit` reuse `feats.f16`; only SAM2
   (segment) and NVENC (bake) run. `masks.u1` packbits round-trips at the stored `(T,oh,ow)`.
5. **Points are validated & mapped correctly.** Client maps click `clientX/Y` → source pixels under
   `object-fit: contain`; server rejects malformed points (400); label 1=fg / 0=bg matches SAM.
6. **SAM2 in fp32 on Blackwell (sm_120) is numerically fine** and single-object propagation is
   stable across the clip.

## Evidence already produced (mine — inspect, don't trust)

| Claim | Command / artifact | My reading | Where it lives |
|---|---|---|---|
| align + bake + reuse | `REPVIS_TEST_GPU=1 uv run pytest` | 6 pass (joint run + segment + refit) | `tests/test_api.py::test_full_joint_run_and_persistence` |
| mask baked | decode `runs/<rid>/pca.mp4`, near-black ≈ 0.82 | background zeroed, subject kept | PROGRESS §Rounds |
| clean pixel fg | live browser E2E, +click | subject segmented crisply, reload works | PROGRESS §Rounds |
| masks.u1 size | `stat masks.u1` == `ceil(T·oh·ow/8)` | packbits shape correct | run dir |

## Known weak spots

- **Auto-seed quality**: a single DINO-saliency argmax point misses on some frames (registered
  `feat/sam-autoseed-quality`); it is refinable by click but the default can be a small/off region.
- **Long clips**: `init_video_session` holds all sampled frames (host RAM) + SAM propagates the whole
  clip per click (~33 ms/frame) — click latency and memory grow with `max_frames` (`perf/sam-session-cache`).
- **Single object only** (obj_id=1); multi-subject scenes need multiple objects.
- Joint (multi-source) runs: each source segments independently — confirm masks/points don't cross.

## Domain lens

Video-segmentation + GPU-pipeline correctness: mask↔frame↔feature alignment, the pixel-mask bake,
and the reuse-without-re-extract contract. Not UI polish.

## Out of scope

The jahns-workflow harness; the pre-existing DINO/V-JEPA extract + NVDEC/NVENC pipeline (unchanged);
auto-seed *quality* (tracked separately — correctness of the mechanism is in scope, not its picks).

## Response wanted

Major / critical issues only. For each: a concrete failure mechanism and where you confirmed it.
Separate confirmed findings, open domain questions, and residual risks from unavailable GPU/data.
