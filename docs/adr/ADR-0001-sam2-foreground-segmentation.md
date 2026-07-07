# ADR-0001: Replace feature-clustering remove_bg with SAM2 foreground segmentation

- Status: accepted
- Date: 2026-07-07
- Round: 2026-07-07-sam2-foreground
- SSOT sections affected: none (SSOT disabled)
- Tasks: feat/sam2-foreground-segmentation, feat/per-cell-bg-threshold-refit (mask half superseded),
  fix/refit-soft-weight-mask, decision/refit-mask-grid-threshold

## Context

Background removal was built on the dense patch features themselves: PC1/Otsu, then a
gray-field debias + k-means(k=4) + top-border prior over the ~64×36 patch grid. Because the
features are coarse (one token per patch), the mask could only ever be blobby and
patch-quantized — it shaved subjects from the edges inward and could not produce a clean object
boundary. A short-lived per-cell threshold slider (round `2026-07-07-sam2-foreground`, commit
`3aff231`) made the coarseness tunable but not the boundary; user feedback was that it "smears
from the edge" rather than separating the foreground.

## Decision

Scrap the feature-clustering remove_bg and segment the foreground with **SAM2**
(`facebook/sam2.1-hiera-tiny`, Apache-2.0, shipped in `transformers`, ~1 GB VRAM, ~33 ms/frame)
running on the decoded RGB frames — pixel-accurate masks propagated across the clip by SAM2's
streaming memory, **baked into the PCA video server-side**.

- **Auto-seed** each run from the frame-0 DINO-saliency peak (a positive point); the mask bakes
  automatically (background → black).
- **Refine** with point prompts: click a cell for a `+` point, Alt/Option-click for a `−` point;
  points carry their frame index and SAM2 is conditioned per frame; `↺` resets to the auto-seed.
- **Refit** re-fits the display PCA basis over the (now clean) foreground; the fit weights grid
  tokens by their foreground fraction (soft-weighted PCA, no hard threshold) so thin structures
  survive — see `decision/refit-mask-grid-threshold`.
- Persist `feats.f16` + `masks.u1` + `state.pt` + `meta{frame_indices,seg}` so refine/refit reuse
  cached features (no backbone re-run).

## Consequences

- Clean, pixel-accurate, temporally consistent foreground; the "edge shaving" is gone.
- New dependency surface: a segmentation model + an interactive prompting UX and its endpoints
  (`/segment`, `/refit`) and failure modes (a subsequent review added: per-frame prompting,
  failure isolation, point validation, a run-mutation mutex, a shared model-load lock).
- Cost: SAM2 re-propagates the whole clip per click (~33 ms/frame) and the run persists the full
  fp16 feature cache (~2–4 GB/cell) for refit. Open follow-ups: `feat/sam-autoseed-quality`,
  `perf/sam-session-cache`, `spike/frame-alignment-check`.
- Invalidated if a lighter unprompted matting model (e.g. BiRefNet) were preferred — rejected
  below.

## Alternatives considered

- **Keep tuning feature-clustering remove_bg** — rejected: the patch grid can't yield clean
  boundaries no matter the threshold; the limitation is fundamental.
- **MobileSAM** — image-only (no video memory), would flicker frame-to-frame; SAM2's propagation
  gives temporal consistency.
- **FastSAM** — AGPL, incompatible with this Apache-2.0 repo.
- **Unprompted matting / SOD (BiRefNet, RMBG-2.0)** — more direct for "background removal" but not
  SAM-family (chosen for interactive refinement + video propagation); RMBG-2.0 is non-commercial.
- **Fully automatic (no clicks)** — kept the auto-seed as the default but exposed `+`/`−` clicks,
  since a single saliency seed misses on some frames.
