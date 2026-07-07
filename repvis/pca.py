"""PCA -> RGB rendering of dense patch features (all on GPU).

Split into a fit step (on a capped token subsample gathered while streaming) and
a per-chunk projection step, so we never materialize all features at once.

Background removal ("remove_bg") — chosen by side-by-side evaluation of 12
strategies on real clips (see repo history for the study):
  1. Subtract the model's *positional field*: its response to uniform gray
     frames, centered across positions. DINOv3/V-JEPA (RoPE) leak a strong
     smooth position gradient into patch features that otherwise dominates any
     fg/bg split (masks degenerate into left/right halves); the gray response
     isolates exactly that content-free component, so subtracting it can never
     remove actual objects. For DINOv2 the centered field is ~0, a no-op.
  2. k-means (k=4) on the top-8 PCs of the debiased, L2-normalized tokens.
     A single PC1 threshold (the classic DINOv2-paper trick) breaks on scenes
     with several objects; 4 clusters + a border prior degrade gracefully.
  3. Background = clusters over-represented on the top/upper-side image border
     (bottom border excluded: subjects are routinely cut off by it). This
     replaces "foreground = minority side", which inverted the mask on most
     real clips. Worst case is keeping an extra object — never losing the
     subject.
  4. At projection, the binary mask is cleaned with a 3x3 majority filter
     (kills single-token salt-and-pepper speckle).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

_MASK_K = 4          # fg/bg clusters
_MASK_PCS = 8        # cluster space dimensionality
_BORDER_SLACK = 1.25  # bg iff cluster's border share > slack * expected share


def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]


@torch.inference_mode()
def _fit(x: torch.Tensor, k: int, fit_max: int):
    """Return top-k principal components (k, D) and the mean (D,)."""
    xs = _subsample(x, fit_max)
    mean = xs.mean(0)
    xc = xs - mean
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:k].clone()
    for i in range(k):  # deterministic sign -> stable colors run-to-run
        j = torch.argmax(comps[i].abs())
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return comps, mean


@torch.inference_mode()
def _quantile(x: torch.Tensor, qs) -> torch.Tensor:
    x = _subsample(x, 200_000)
    return torch.quantile(x, torch.tensor(qs, device=x.device, dtype=x.dtype))


def _kmeans(z: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    """k-means on rows of z with deterministic farthest-point init -> (k, dim)."""
    cent = [z[torch.argmax((z * z).sum(1))]]
    for _ in range(k - 1):
        d = torch.cdist(z, torch.stack(cent)).amin(1)
        cent.append(z[torch.argmax(d)])
    cent = torch.stack(cent)
    for _ in range(iters):
        a = torch.cdist(z, cent).argmin(1)
        new = torch.stack([z[a == j].mean(0) if (a == j).any() else cent[j]
                           for j in range(k)])
        if torch.allclose(new, cent):
            break
        cent = new
    return cent


def _top_border(gh: int, gw: int, device) -> torch.Tensor:
    """Flat (gh*gw,) bool: top row + upper halves of the side columns."""
    m = torch.zeros(gh, gw, dtype=torch.bool, device=device)
    m[0, :] = True
    m[: gh // 2, 0] = m[: gh // 2, -1] = True
    return m.reshape(-1)


@dataclass
class BgCtx:
    """Spatial context for the fit-token sample, needed only for remove_bg."""
    pos: torch.Tensor      # (M,) flat grid position of each fit token
    grid_id: torch.Tensor  # (M,) index into grids/fields per token's source
    grids: list            # [(gh, gw)] per source
    fields: dict           # {(gh, gw): (gh*gw, D) float32 gray positional field}


@dataclass
class PCAState:
    l2norm: bool
    mean: torch.Tensor    # (D,)
    comps: torch.Tensor   # (3, D)
    lo: torch.Tensor      # (3,)
    hi: torch.Tensor      # (3,)
    remove_bg: bool = False
    mask_mean: torch.Tensor | None = None   # (D,) cluster-space centering
    mask_comps: torch.Tensor | None = None  # (_MASK_PCS, D)
    cent: torch.Tensor | None = None        # (_MASK_K, _MASK_PCS)
    fg_clusters: torch.Tensor | None = None  # (_MASK_K,) bool
    fields: dict | None = None              # {(gh, gw): (gh*gw, D) centered field}
    # continuous background score (higher = more background-like), for the live
    # threshold slider: score = ((z_masked . bg_axis) - bg_lo)/(bg_hi - bg_lo).
    bg_axis: torch.Tensor | None = None     # (_MASK_PCS,) fg->bg direction
    bg_lo: torch.Tensor | None = None       # () score normalization low
    bg_hi: torch.Tensor | None = None       # () score normalization high
    bg_threshold: float = 0.5               # default keep-cutoff: fg iff score < t

    def to(self, device) -> "PCAState":
        """Non-mutating: returns a copy on `device` (states are shared across
        concurrent render threads, one per GPU)."""
        def opt(t):
            return None if t is None else t.to(device)
        return PCAState(
            self.l2norm, self.mean.to(device), self.comps.to(device),
            self.lo.to(device), self.hi.to(device), self.remove_bg,
            opt(self.mask_mean), opt(self.mask_comps), opt(self.cent),
            opt(self.fg_clusters),
            None if self.fields is None else {g: f.to(device) for g, f in self.fields.items()},
            opt(self.bg_axis), opt(self.bg_lo), opt(self.bg_hi), self.bg_threshold)


@torch.inference_mode()
def _fit_mask(raw: torch.Tensor, bg: BgCtx):
    """Cluster debiased tokens and mark fg clusters via the border prior.

    Returns (mask_mean, mask_comps, cent, fg_clusters, fg_token_mask, fields,
    bg_axis, bg_lo, bg_hi, bg_threshold) or None when the split is degenerate
    (then everything is kept, mask off).
    """
    dev = raw.device
    fields = {tuple(g): (f.to(dev) - f.to(dev).mean(0)) for g, f in bg.fields.items()}
    xd = raw.clone()
    border = torch.zeros(raw.shape[0], dtype=torch.bool, device=dev)
    for gi, (gh, gw) in enumerate(bg.grids):
        m = bg.grid_id == gi
        if m.any():
            xd[m] -= fields[(gh, gw)][bg.pos[m]]
            border[m] = _top_border(gh, gw, dev)[bg.pos[m]]
    zn = F.normalize(xd, dim=1)
    mask_comps, mask_mean = _fit(zn, _MASK_PCS, zn.shape[0])
    z = (zn - mask_mean) @ mask_comps.T
    cent = _kmeans(z, _MASK_K)
    assign = torch.cdist(z, cent).argmin(1)
    bf = border.float().mean()
    fg_clusters = torch.tensor(
        [bool(border[assign == j].float().mean() <= _BORDER_SLACK * bf)
         if (assign == j).any() else False for j in range(_MASK_K)], device=dev)
    fg = fg_clusters[assign]
    if not fg_clusters.any() or fg_clusters.all() or fg.float().mean() < 0.02:
        return None

    # Continuous background score: project mask-space coords onto the fg->bg axis
    # (bg cluster centroids minus fg cluster centroids). The binary border-prior
    # split above becomes the DEFAULT cut on this axis; the slider then moves the
    # cut smoothly, keeping a patch as foreground iff its score < threshold.
    pf = cent[fg_clusters].mean(0)
    pb = cent[~fg_clusters].mean(0)
    bg_axis = F.normalize(pb - pf, dim=0)                 # (_MASK_PCS,)
    s = z @ bg_axis                                       # (M,) higher = more bg
    q = _quantile(s, [0.01, 0.99])
    bg_lo, bg_hi = q[0], q[1]
    mid = 0.5 * (pf + pb) @ bg_axis                       # fg/bg prototype midpoint
    t0 = float(((mid - bg_lo) / (bg_hi - bg_lo).clamp_min(1e-6)).clamp(0.0, 1.0))
    return mask_mean, mask_comps, cent, fg_clusters, fg, fields, bg_axis, bg_lo, bg_hi, t0


@torch.inference_mode()
def fit_pca_state(fit_tokens: torch.Tensor, *, remove_bg: bool = False, l2norm: bool = False,
                  percentiles=(2.0, 98.0), bg: BgCtx | None = None) -> PCAState:
    """Fit the PCA basis + display range from a (M, D) token sample (capped upstream)."""
    x = fit_tokens.float()
    if l2norm:
        x = F.normalize(x, dim=1)

    mask_mean = mask_comps = cent = fg_clusters = fields = fg = None
    bg_axis = bg_lo = bg_hi = None
    bg_threshold = 0.5
    if remove_bg and bg is not None:
        fitted = _fit_mask(fit_tokens.float(), bg)
        if fitted is not None:
            (mask_mean, mask_comps, cent, fg_clusters, fg, fields,
             bg_axis, bg_lo, bg_hi, bg_threshold) = fitted

    src = x[fg] if fg is not None else x       # display basis fit on fg only
    comps, mean = _fit(src, 3, src.shape[0])
    z = (x - mean) @ comps.T
    ref = z[fg] if fg is not None else z

    p = [percentiles[0] / 100.0, percentiles[1] / 100.0]
    lo = torch.empty(3, device=x.device)
    hi = torch.empty(3, device=x.device)
    for c in range(3):
        q = _quantile(ref[:, c], p)
        lo[c], hi[c] = q[0], q[1]

    return PCAState(l2norm, mean, comps, lo, hi, remove_bg,
                    mask_mean, mask_comps, cent, fg_clusters, fields,
                    bg_axis, bg_lo, bg_hi, bg_threshold)


@torch.inference_mode()
def project_chunk(feats: torch.Tensor, st: PCAState) -> torch.Tensor:
    """feats (T,H,W,D) -> UNMASKED rgb (T,H,W,3) in [0,1], using a fixed basis.

    Background removal is no longer baked here — it is applied as a live,
    thresholdable mask at display time (client) or on bake (`apply_mask`), so the
    same encoded video serves any threshold. See `score_chunk`.
    """
    T, H, W, D = feats.shape
    x = feats.reshape(-1, D).float()
    disp = F.normalize(x, dim=1) if st.l2norm else x
    z = (disp - st.mean) @ st.comps.T
    rgb = ((z - st.lo) / (st.hi - st.lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return rgb.reshape(T, H, W, 3)


def _has_mask(st: PCAState) -> bool:
    return (st.remove_bg and st.fields is not None and st.mask_mean is not None
            and st.mask_comps is not None and st.bg_axis is not None
            and st.bg_lo is not None and st.bg_hi is not None)


@torch.inference_mode()
def score_chunk(feats: torch.Tensor, st: PCAState) -> torch.Tensor:
    """feats (T,H,W,D) -> per-patch background score (T,H,W) in [0,1].

    Higher = more background-like. A patch is foreground iff score < threshold.
    Requires mask material (`_has_mask`); the caller checks first.
    """
    assert (st.fields is not None and st.mask_mean is not None and st.mask_comps is not None
            and st.bg_axis is not None and st.bg_lo is not None and st.bg_hi is not None)
    T, H, W, D = feats.shape
    x = feats.reshape(-1, D).float()
    xd = (x.reshape(T, H * W, D) - st.fields[(H, W)]).reshape(-1, D)
    zc = (F.normalize(xd, dim=1) - st.mask_mean) @ st.mask_comps.T
    s = zc @ st.bg_axis
    s = ((s - st.bg_lo) / (st.bg_hi - st.bg_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return s.reshape(T, H, W)


@torch.inference_mode()
def apply_mask(rgb: torch.Tensor, score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Black out background patches in rgb (T,H,W,3) given score (T,H,W).

    Keep a patch iff score < threshold, then 2x 3x3 majority filter to de-speckle.
    Mirrors the client-side compositor exactly (must stay in sync).
    """
    T, H, W, _ = rgb.shape
    m = (score < threshold).float().reshape(T, 1, H, W)
    for _ in range(2):
        m = (F.avg_pool2d(m, 3, 1, 1) > 0.5).float()
    return rgb * m.reshape(T, H, W, 1)


@torch.inference_mode()
def refit_display(fg_tokens: torch.Tensor, st: PCAState,
                  percentiles=(2.0, 98.0)) -> PCAState:
    """Re-fit ONLY the display basis (mean/comps/lo/hi) on `fg_tokens`, keeping
    the mask material — the per-cell 'enhance' refit over the current foreground.
    """
    x = fg_tokens.float()
    if st.l2norm:
        x = F.normalize(x, dim=1)
    comps, mean = _fit(x, 3, x.shape[0])
    z = (x - mean) @ comps.T
    p = [percentiles[0] / 100.0, percentiles[1] / 100.0]
    lo = torch.empty(3, device=x.device)
    hi = torch.empty(3, device=x.device)
    for c in range(3):
        q = _quantile(z[:, c], p)
        lo[c], hi[c] = q[0], q[1]
    return replace(st, mean=mean, comps=comps, lo=lo, hi=hi)
