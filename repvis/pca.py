"""PCA -> RGB rendering of dense patch features (all on GPU).

Split into a fit step (on a capped token subsample gathered while streaming) and
a per-chunk projection step, so we never materialize all features at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


@torch.inference_mode()
def _otsu_threshold(v: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Otsu threshold: the value that best splits a (bimodal) 1-D distribution.

    Used to separate foreground/background along PC1 — a data-driven valley, not
    a fixed 50% median split.
    """
    v = _subsample(v, 200_000)
    lo, hi = v.min(), v.max()
    if (hi - lo) < 1e-9:
        return lo
    hist = torch.histc(v, bins=bins, min=float(lo), max=float(hi))
    p = hist / hist.sum().clamp_min(1.0)
    centers = torch.linspace(float(lo), float(hi), bins, device=v.device)
    w0 = torch.cumsum(p, 0)                       # weight of "below" class
    w1 = (1.0 - w0).clamp_min(1e-12)
    csum = torch.cumsum(p * centers, 0)
    mu0 = csum / w0.clamp_min(1e-12)
    mu1 = (csum[-1] - csum) / w1
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2          # between-class variance
    return centers[int(torch.argmax(sigma_b))]


@dataclass
class PCAState:
    l2norm: bool
    mean: torch.Tensor    # (D,)
    comps: torch.Tensor   # (3, D)
    lo: torch.Tensor      # (3,)
    hi: torch.Tensor      # (3,)
    remove_bg: bool = False
    c1: torch.Tensor | None = None   # (1, D)
    m1: torch.Tensor | None = None   # (D,)
    thr: torch.Tensor | None = None  # scalar
    fg_above: bool = True

    def to(self, device) -> "PCAState":
        """Non-mutating: returns a copy on `device` (states are shared across
        concurrent render threads, one per GPU)."""
        return PCAState(
            self.l2norm, self.mean.to(device), self.comps.to(device),
            self.lo.to(device), self.hi.to(device), self.remove_bg,
            None if self.c1 is None else self.c1.to(device),
            None if self.m1 is None else self.m1.to(device),
            None if self.thr is None else self.thr.to(device),
            self.fg_above)


@torch.inference_mode()
def fit_pca_state(fit_tokens: torch.Tensor, *, remove_bg: bool = False, l2norm: bool = False,
                  percentiles=(2.0, 98.0)) -> PCAState:
    """Fit the PCA basis + display range from a (M, D) token sample (capped upstream)."""
    x = fit_tokens.float()
    if l2norm:
        x = F.normalize(x, dim=1)

    c1 = m1 = thr = None
    fg_above = True
    fg = None
    if remove_bg:
        _c1, _m1 = _fit(x, 1, x.shape[0])
        pc1 = (x - _m1) @ _c1[0]
        _thr = _otsu_threshold(pc1)                    # data-driven fg/bg split (not 50%)
        above = pc1 > _thr
        fg_above = bool(above.float().mean() <= 0.5)   # foreground = minority side
        cand = above if fg_above else ~above
        # only enable masking on a non-degenerate split; else keep everything
        if cand.float().mean() >= 0.02:
            fg, c1, m1, thr = cand, _c1, _m1, _thr

    src = x[fg] if fg is not None else x
    comps, mean = _fit(src, 3, src.shape[0])
    z = (x - mean) @ comps.T
    ref = z[fg] if fg is not None else z

    p = [percentiles[0] / 100.0, percentiles[1] / 100.0]
    lo = torch.empty(3, device=x.device)
    hi = torch.empty(3, device=x.device)
    for c in range(3):
        q = _quantile(ref[:, c], p)
        lo[c], hi[c] = q[0], q[1]

    return PCAState(l2norm, mean, comps, lo, hi, remove_bg, c1, m1, thr, fg_above)


@torch.inference_mode()
def project_chunk(feats: torch.Tensor, st: PCAState) -> torch.Tensor:
    """feats (T,H,W,D) -> rgb (T,H,W,3) in [0,1], using a fixed PCA state."""
    T, H, W, D = feats.shape
    x = feats.reshape(-1, D).float()
    if st.l2norm:
        x = F.normalize(x, dim=1)
    z = (x - st.mean) @ st.comps.T
    rgb = ((z - st.lo) / (st.hi - st.lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    if st.remove_bg and st.c1 is not None:
        pc1 = (x - st.m1) @ st.c1[0]
        above = pc1 > st.thr
        fg = above if st.fg_above else ~above
        rgb[~fg] = 0.0
    return rgb.reshape(T, H, W, 3)
